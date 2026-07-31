

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	digest "github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/vbauerster/mpb/v8"
)

// hfGGUFMediaType is the standard Docker AI media type for GGUF model layers.
const hfGGUFMediaType = "application/vnd.docker.ai.gguf.v3"

// ---------------------------------------------------------------------------
// Registry detection
// ---------------------------------------------------------------------------

// isKnownOCIHost returns true for registries that are definitely OCI-compliant,
// skipping the network probe entirely.
func isKnownOCIHost(host string) bool {
	switch host {
	case "ghcr.io", "docker.io", "index.docker.io", "registry-1.docker.io",
		"quay.io", "gcr.io", "mcr.microsoft.com", "public.ecr.aws":
		return true
	}
	return false
}

// isKnownHFHost returns true for known HuggingFace-compatible hosts.
func isKnownHFHost(host string) bool {
	switch host {
	case "hf.co", "huggingface.co", "modelscope.cn":
		return true
	}
	return false
}

// isOCIRegistry probes the OCI Distribution /v2/ endpoint and returns true if
// the server advertises itself as an OCI registry via the standard header.
func isOCIRegistry(ctx context.Context, client *http.Client, host string) bool {
	probeCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(probeCtx, "GET", "https://"+host+"/v2/", nil)
	if err != nil {
		return false
	}
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	// OCI registries advertise registry/2.0 on both 200 and 401 responses.
	return resp.Header.Get("Docker-Distribution-Api-Version") != ""
}

// ---------------------------------------------------------------------------
// HuggingFace API types and helpers
// ---------------------------------------------------------------------------

// hfFile is one entry returned by the HuggingFace tree API.
type hfFile struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
	OID  string `json:"oid"`
	Type string `json:"type"` // "file" or "directory"
}

// hfEndpoint returns the HuggingFace API base URL for the host.
// Mirrors llama.cpp's MODEL_ENDPOINT / HF_ENDPOINT override logic.
func hfEndpoint(host string) string {
	for _, env := range []string{"MODEL_ENDPOINT", "HF_ENDPOINT"} {
		if v := os.Getenv(env); v != "" {
			return strings.TrimRight(v, "/") + "/"
		}
	}
	if host == "hf.co" {
		return "https://huggingface.co/"
	}
	return "https://" + host + "/"
}

// hfToken resolves the HuggingFace bearer token to use for authenticated
// requests, mirroring huggingface_hub's own resolution order: the HF_TOKEN
// environment variable (falling back to the legacy
// HUGGING_FACE_HUB_TOKEN), then the on-disk active-token file written by
// `llmman login` — see the Rust `hf` module's `token_path`, which uses the
// exact same path, so either tool's login is honored by the other.
func hfToken() string {
	for _, env := range []string{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"} {
		if v := strings.TrimSpace(os.Getenv(env)); v != "" {
			return v
		}
	}
	data, err := os.ReadFile(hfTokenPath())
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

// hfTokenPath returns the path to the active HuggingFace token file:
// $HF_TOKEN_PATH if set, else "$HF_HOME/token", else
// "~/.cache/huggingface/token".
func hfTokenPath() string {
	if p := os.Getenv("HF_TOKEN_PATH"); p != "" {
		return p
	}
	if home := os.Getenv("HF_HOME"); home != "" {
		return filepath.Join(home, "token")
	}
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".cache", "huggingface", "token")
	}
	return filepath.Join(".cache", "huggingface", "token")
}

// hfHeadMetadata performs a HEAD request against a HuggingFace file's
// /resolve/ URL and reports the file's real content digest and size,
// without downloading the body — mirroring huggingface_hub's own
// get_hf_file_metadata(). This is what makes streaming a HuggingFace file
// straight into a registry push possible at all: containerd/OCI registry
// pushes require the blob's digest to be known *before* any bytes are
// sent (see backend_docker.go's llmman_transfer), and for a large,
// LFS-tracked file (virtually every real GGUF/safetensors weight file)
// the true sha256 of the content is exposed via the X-Linked-Etag header
// on this cheap HEAD request — the same field huggingface_hub prefers
// over the plain ETag for exactly this reason (LFS pointer vs. real
// object). ok is false when the digest can't be determined this way
// (small, non-LFS files, where the ETag is a git blob sha1, not a sha256
// of the content) — callers should fall back to a normal buffered
// download for those; they're tiny (config/tokenizer files), so buffering
// them in memory costs nothing.
func hfHeadMetadata(ctx context.Context, client *http.Client, url, token string) (dgst digest.Digest, size int64, ok bool, err error) {
	req, err := http.NewRequestWithContext(ctx, "HEAD", url, nil)
	if err != nil {
		return "", 0, false, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	req.Header.Set("Accept-Encoding", "identity") // force the real, uncompressed size
	resp, err := client.Do(req)
	if err != nil {
		return "", 0, false, fmt.Errorf("HEAD %s: %w", url, err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		return "", 0, false, fmt.Errorf("HEAD %s: HTTP %d", url, resp.StatusCode)
	}

	etag := resp.Header.Get("X-Linked-Etag")
	if etag == "" {
		etag = resp.Header.Get("ETag")
	}
	etag = strings.TrimPrefix(etag, "W/")
	etag = strings.Trim(etag, `"`)
	if len(etag) != 64 {
		return "", 0, false, nil // not a sha256 — not LFS, caller should buffer instead
	}

	sizeStr := resp.Header.Get("X-Linked-Size")
	if sizeStr == "" {
		sizeStr = resp.Header.Get("Content-Length")
	}
	size, convErr := parseInt64(sizeStr)
	if convErr != nil {
		return "", 0, false, nil
	}

	return digest.NewDigestFromEncoded(digest.SHA256, strings.ToLower(etag)), size, true, nil
}

func parseInt64(s string) (int64, error) {
	var n int64
	_, err := fmt.Sscanf(s, "%d", &n)
	return n, err
}

// hfGet issues an authenticated GET and decodes JSON into dst.
func hfGet(ctx context.Context, client *http.Client, url, token string, dst any) error {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("GET %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("GET %s: HTTP %d", url, resp.StatusCode)
	}
	return json.NewDecoder(resp.Body).Decode(dst)
}

// hfFetchCommit returns the current commit SHA for owner/repo.
func hfFetchCommit(ctx context.Context, client *http.Client, endpoint, owner, repo, token string) (string, error) {
	var info struct {
		SHA string `json:"sha"`
	}
	url := endpoint + "api/models/" + owner + "/" + repo
	if err := hfGet(ctx, client, url, token, &info); err != nil {
		return "", fmt.Errorf("HF model info: %w", err)
	}
	if info.SHA == "" {
		return "main", nil // graceful fallback
	}
	return info.SHA, nil
}

// hfFetchFiles returns the recursive file listing for owner/repo at commit.
func hfFetchFiles(ctx context.Context, client *http.Client, endpoint, owner, repo, commit, token string) ([]hfFile, error) {
	var files []hfFile
	url := endpoint + "api/models/" + owner + "/" + repo + "/tree/" + commit + "?recursive=true"
	if err := hfGet(ctx, client, url, token, &files); err != nil {
		return nil, fmt.Errorf("HF file list: %w", err)
	}
	return files, nil
}

// ---------------------------------------------------------------------------
// GGUF file selection (mirrors llama.cpp find_best_model)
// ---------------------------------------------------------------------------

// quantPreference is the default quantization preference order, matching llama.cpp.
var quantPreference = []string{"Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0", "Q4_0", "Q6_K", "Q2_K"}

// isModelGGUF returns true for GGUF files that are primary model weights
// (not mmproj projectors or imatrix importance files).
func isModelGGUF(path string) bool {
	lower := strings.ToLower(path)
	return strings.HasSuffix(lower, ".gguf") &&
		!strings.Contains(lower, "mmproj") &&
		!strings.Contains(lower, "imatrix")
}

// selectGGUF picks the best GGUF from the file listing.
// tag is the user-supplied quantization hint (e.g. "Q4_K_M") or empty for auto.
func selectGGUF(files []hfFile, tag string) (hfFile, error) {
	var models []hfFile
	for _, f := range files {
		if f.Type == "file" && isModelGGUF(f.Path) {
			models = append(models, f)
		}
	}
	if len(models) == 0 {
		return hfFile{}, fmt.Errorf("no GGUF model files found in repository")
	}

	// Explicit tag: user asked for a specific quantization.
	if tag != "" && tag != "latest" {
		upper := strings.ToUpper(tag)
		for _, f := range models {
			if strings.Contains(strings.ToUpper(f.Path), upper) {
				return f, nil
			}
		}
		return hfFile{}, fmt.Errorf("no GGUF file matching %q found; available:\n%s",
			tag, ggufList(models))
	}

	// Auto-select by preference list (Q4_K_M first, then Q8_0, …).
	for _, pref := range quantPreference {
		for _, f := range models {
			if strings.Contains(strings.ToUpper(f.Path), pref) {
				return f, nil
			}
		}
	}

	// Fallback: smallest file (most compressed).
	sort.Slice(models, func(i, j int) bool { return models[i].Size < models[j].Size })
	return models[0], nil
}

func ggufList(files []hfFile) string {
	var b strings.Builder
	for _, f := range files {
		b.WriteString("  " + f.Path + "\n")
	}
	return b.String()
}

// ---------------------------------------------------------------------------
// parseHFRef
// ---------------------------------------------------------------------------

// parseHFRef splits a (possibly `:latest`-normalized) HF reference
// "host/owner/repo[:tag]" into its four components.
func parseHFRef(ref string) (host, owner, repo, tag string, err error) {
	if idx := strings.LastIndex(ref, ":"); idx > strings.LastIndex(ref, "/") {
		tag = ref[idx+1:]
		ref = ref[:idx]
	}
	parts := strings.SplitN(ref, "/", 3)
	if len(parts) != 3 {
		return "", "", "", "", fmt.Errorf("invalid HuggingFace reference %q: expected host/owner/repo", ref)
	}
	return parts[0], parts[1], parts[2], tag, nil
}

// ---------------------------------------------------------------------------
// pullHF — top-level HuggingFace pull
// ---------------------------------------------------------------------------

// cachedLayerName returns the GGUF filename for ref if it is fully cached in
// the local OCI store (manifest blob + all layer blobs present), or "" if not.
func cachedLayerName(layoutDir, ref string) string {
	idx, err := readIndex(layoutDir)
	if err != nil {
		return ""
	}
	for _, m := range idx.Manifests {
		if m.Annotations[ocispec.AnnotationRefName] != ref {
			continue
		}
		if !blobExists(layoutDir, m) {
			return ""
		}
		data, err := readBlob(layoutDir, m.Digest)
		if err != nil {
			return ""
		}
		var manifest ocispec.Manifest
		if err := json.Unmarshal(data, &manifest); err != nil {
			return ""
		}
		for _, layer := range manifest.Layers {
			if !blobExists(layoutDir, layer) {
				return ""
			}
		}
		// All blobs present — return a filename from the first layer annotation.
		if len(manifest.Layers) > 0 {
			ann := manifest.Layers[0].Annotations
			for _, key := range []string{"org.cncf.model.filepath", ocispec.AnnotationTitle} {
				if name := ann[key]; name != "" {
					return filepath.Base(name)
				}
			}
		}
		return ref
	}
	return ""
}

func pullHF(ctx context.Context, ref, layoutDir string) error {
	host, owner, repo, tag, err := parseHFRef(ref)
	if err != nil {
		return err
	}

	if err := ensureLayout(layoutDir); err != nil {
		return fmt.Errorf("init OCI layout: %w", err)
	}

	// Fast path: skip all network I/O if the ref is fully cached locally.
	if name := cachedLayerName(layoutDir, ref); name != "" {
		fmt.Fprintf(os.Stderr, "Cached   %s\n", name)
		return nil
	}

	endpoint := hfEndpoint(host)
	token := hfToken()

	// apiClient: short total timeout for metadata requests (commit, file list).
	apiClient := &http.Client{Timeout: 120 * time.Second}

	// dlClient: no body read timeout so large files can download without a
	// deadline, but connection and header timeouts prevent hanging on stalls.
	// Mirrors llama.cpp common/download.cpp approach.
	dlClient := &http.Client{
		Transport: &http.Transport{
			DialContext: (&net.Dialer{
				Timeout:   30 * time.Second,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			TLSHandshakeTimeout:   30 * time.Second,
			ResponseHeaderTimeout: 60 * time.Second,
		},
	}
	_ = dlClient // used below

	commit, err := hfFetchCommit(ctx, apiClient, endpoint, owner, repo, token)
	if err != nil {
		return err
	}

	files, err := hfFetchFiles(ctx, apiClient, endpoint, owner, repo, commit, token)
	if err != nil {
		return err
	}

	// Try GGUF first; fall back to safetensors if the repo has none.
	chosen, err := selectGGUF(files, tag)
	if err == nil {
		downloadURL := endpoint + owner + "/" + repo + "/resolve/" + commit + "/" + chosen.Path
		ggufDesc, err := downloadHFBlob(ctx, dlClient, downloadURL, token, layoutDir, owner, repo, commit, chosen)
		if err != nil {
			return err
		}
		return storeHFAsOCI(layoutDir, ref, owner+"/"+repo, chosen.Path, ggufDesc)
	}

	// No GGUF found — pull safetensors files as a CNCF model-spec image.
	return pullHFSafetensors(ctx, dlClient, ref, layoutDir, endpoint, owner, repo, commit, token, files)
}

// safetensorsMediaType maps a file extension to the appropriate CNCF layer media type.
func safetensorsMediaType(path string) string {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".safetensors", ".bin", ".pt", ".pth":
		return "application/vnd.cncf.model.weight.v1.raw"
	case ".json", ".model", ".txt", ".tiktoken":
		return "application/vnd.cncf.model.weight.config.v1.raw"
	default:
		return "application/vnd.cncf.model.doc.v1.raw"
	}
}

// shouldDownloadSafetensors returns true for files that belong in a local model directory.
func shouldDownloadSafetensors(path string) bool {
	base := strings.ToLower(filepath.Base(path))
	ext := strings.ToLower(filepath.Ext(path))
	// Skip hidden files, large non-model binaries, and git internals.
	if strings.HasPrefix(base, ".") {
		return false
	}
	switch ext {
	case ".safetensors", ".bin", ".pt", ".pth": // weights
		return true
	case ".json", ".model", ".txt", ".tiktoken": // config / tokeniser
		return true
	}
	// README and licence are useful but optional.
	switch base {
	case "readme.md", "license", "licence", "license.txt", "licence.txt":
		return true
	}
	return false
}

func pullHFSafetensors(
	ctx context.Context,
	client *http.Client,
	ref, layoutDir, endpoint, owner, repo, commit, token string,
	files []hfFile,
) error {
	var toDownload []hfFile
	for _, f := range files {
		if f.Type == "file" && shouldDownloadSafetensors(f.Path) {
			toDownload = append(toDownload, f)
		}
	}
	if len(toDownload) == 0 {
		return fmt.Errorf("no model files found in repository %s/%s", owner, repo)
	}

	var layers []ocispec.Descriptor
	for _, f := range toDownload {
		url := endpoint + owner + "/" + repo + "/resolve/" + commit + "/" + f.Path
		desc, err := downloadHFBlob(ctx, client, url, token, layoutDir, owner, repo, commit, f)
		if err != nil {
			return fmt.Errorf("download %s: %w", f.Path, err)
		}
		// Override media type and use the full relative path as the filepath annotation.
		desc.MediaType = safetensorsMediaType(f.Path)
		desc.Annotations = map[string]string{
			"org.cncf.model.filepath": f.Path,
		}
		layers = append(layers, desc)
	}

	return storeSafetensorsAsOCI(layoutDir, ref, owner+"/"+repo, layers)
}

func storeSafetensorsAsOCI(layoutDir, ref, modelRepo string, layers []ocispec.Descriptor) error {
	var cfg cncfModelConfig
	cfg.Config.Format = "safetensors"
	cfg.ModelFS.Type = "layers"
	for _, l := range layers {
		cfg.ModelFS.DiffIDs = append(cfg.ModelFS.DiffIDs, l.Digest.String())
	}
	cfgData, err := json.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("marshal CNCF config: %w", err)
	}
	configDesc, err := writeBlob(layoutDir, "application/vnd.cncf.model.config.v1+json", cfgData)
	if err != nil {
		return fmt.Errorf("write CNCF config: %w", err)
	}
	manifest := ocispec.Manifest{
		Versioned:    specs.Versioned{SchemaVersion: 2},
		MediaType:    ocispec.MediaTypeImageManifest,
		ArtifactType: "application/vnd.cncf.model.manifest.v1+json",
		Config:       configDesc,
		Layers:       layers,
		Annotations:  map[string]string{"ai.model.repo": modelRepo},
	}
	manifestData, err := json.Marshal(manifest)
	if err != nil {
		return fmt.Errorf("marshal manifest: %w", err)
	}
	manifestDesc, err := writeBlob(layoutDir, ocispec.MediaTypeImageManifest, manifestData)
	if err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}
	return updateIndex(layoutDir, ref, manifestDesc)
}

// ---------------------------------------------------------------------------
// stallReader — cancels the context if no bytes arrive within timeout.
// Mirrors llama.cpp's implicit stall detection via cpp-httplib timeouts.
// ---------------------------------------------------------------------------

type stallReader struct {
	r      io.Reader
	timer  *time.Timer
	cancel context.CancelFunc
}

func newStallReader(r io.Reader, timeout time.Duration, cancel context.CancelFunc) *stallReader {
	sr := &stallReader{r: r, cancel: cancel}
	sr.timer = time.AfterFunc(timeout, cancel)
	return sr
}

func (sr *stallReader) Read(p []byte) (int, error) {
	n, err := sr.r.Read(p)
	if n > 0 {
		sr.timer.Reset(60 * time.Second) // bytes arrived, reset stall clock
	}
	return n, err
}

func (sr *stallReader) stop() { sr.timer.Stop() }

// ---------------------------------------------------------------------------
// downloadHFBlob — HTTP download with resume, retry, and stall detection.
// Mirrors llama.cpp common/download.cpp: 3 attempts, 2s/4s backoff.
// ---------------------------------------------------------------------------

const (
	dlMaxAttempts  = 3
	dlRetryBase    = 2 * time.Second // doubles each retry: 2s, 4s
	dlStallTimeout = 60 * time.Second
)

func downloadHFBlob(ctx context.Context, client *http.Client, url, token, layoutDir, owner, repo, commit string, file hfFile) (ocispec.Descriptor, error) {
	if err := os.MkdirAll(filepath.Join(layoutDir, "blobs"), 0o755); err != nil {
		return ocispec.Descriptor{}, err
	}

	sanitize := strings.NewReplacer("/", "_", ":", "_", ".", "_")
	tmpKey := sanitize.Replace(owner + "_" + repo + "_" + commit[:12] + "_" + filepath.Base(file.Path))
	tmpPath := filepath.Join(layoutDir, "blobs", "hf-"+tmpKey+".part")

	label := "Pulling  " + filepath.Base(file.Path)
	doneLbl := "Pulled   " + filepath.Base(file.Path)
	prog := mpb.New(mpb.WithWidth(80), mpb.WithOutput(os.Stderr), mpb.WithRefreshRate(180*time.Millisecond))
	bar := addLayerBar(prog, label, doneLbl, file.Size)

	var lastErr error
	for attempt := 0; attempt < dlMaxAttempts; attempt++ {
		if attempt > 0 {
			delay := dlRetryBase * time.Duration(1<<uint(attempt-1)) // 2s, 4s
			fmt.Fprintf(os.Stderr, "\n[llmman] retrying %s (attempt %d/%d, wait %v)\n",
				filepath.Base(file.Path), attempt+1, dlMaxAttempts, delay)
			select {
			case <-ctx.Done():
				bar.Abort(false)
				prog.Wait()
				return ocispec.Descriptor{}, ctx.Err()
			case <-time.After(delay):
			}
		}

		// Re-read partial file size in case previous attempt downloaded some bytes.
		startOffset := int64(0)
		if fi, err := os.Stat(tmpPath); err == nil && fi.Size() > 0 && fi.Size() < file.Size {
			startOffset = fi.Size()
		}
		bar.SetCurrent(startOffset)

		desc, err := downloadAttempt(ctx, client, url, token, layoutDir, tmpPath, startOffset, file, bar)
		if err == nil {
			prog.Wait()
			return desc, nil
		}

		lastErr = err
		// 4xx errors are permanent — no point retrying.
		if isHTTP4xx(err) {
			break
		}
		// Network/5xx error: keep partial file, retry with resume.
		fmt.Fprintf(os.Stderr, "[llmman] download error: %v\n", err)
	}

	bar.Abort(false)
	prog.Wait()
	os.Remove(tmpPath) // exhausted retries
	return ocispec.Descriptor{}, fmt.Errorf("download %s failed after %d attempts: %w",
		filepath.Base(file.Path), dlMaxAttempts, lastErr)
}

// isHTTP4xx returns true for permanent HTTP client errors (no point retrying).
func isHTTP4xx(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	for _, code := range []string{"HTTP 400", "HTTP 401", "HTTP 403", "HTTP 404"} {
		if strings.Contains(s, code) {
			return true
		}
	}
	return false
}

// downloadAttempt performs one download attempt with stall detection.
func downloadAttempt(ctx context.Context, client *http.Client, url, token, layoutDir, tmpPath string, startOffset int64, file hfFile, bar *mpb.Bar) (ocispec.Descriptor, error) {
	// Per-attempt context with stall cancellation.
	attemptCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	req, err := http.NewRequestWithContext(attemptCtx, "GET", url, nil)
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	if startOffset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", startOffset))
	}

	resp, err := client.Do(req)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("download %s: %w", file.Path, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 && resp.StatusCode != 206 {
		return ocispec.Descriptor{}, fmt.Errorf("download %s: HTTP %d", file.Path, resp.StatusCode)
	}
	if startOffset > 0 && resp.StatusCode == 200 {
		// Server ignored Range header — restart from zero.
		startOffset = 0
		bar.SetCurrent(0)
	}

	digester := digest.Canonical.Digester()
	var f *os.File
	if startOffset > 0 {
		if pf, err := os.Open(tmpPath); err == nil {
			_, hashErr := io.Copy(digester.Hash(), pf)
			pf.Close()
			if hashErr == nil {
				f, _ = os.OpenFile(tmpPath, os.O_APPEND|os.O_WRONLY, 0o644)
			}
		}
		if f == nil {
			digester = digest.Canonical.Digester()
			startOffset = 0
		}
	}
	if f == nil {
		if f, err = os.Create(tmpPath); err != nil {
			return ocispec.Descriptor{}, err
		}
	}

	// Wrap with stall detector: cancel attemptCtx if no bytes for 60s.
	sr := newStallReader(resp.Body, dlStallTimeout, cancel)
	defer sr.stop()

	proxyRC := bar.ProxyReader(sr)
	if proxyRC == nil {
		proxyRC = io.NopCloser(sr)
	}
	written, copyErr := io.Copy(io.MultiWriter(f, digester.Hash()), proxyRC)
	proxyRC.Close()
	f.Close()

	if copyErr != nil {
		// Partial file kept for resume on next attempt — do NOT remove it here.
		return ocispec.Descriptor{}, fmt.Errorf("write %s: %w", file.Path, copyErr)
	}
	total := startOffset + written
	dgst := digester.Digest()

	// Move to content-addressed path.
	dir := filepath.Join(layoutDir, "blobs", dgst.Algorithm().String())
	if err := os.MkdirAll(dir, 0o755); err != nil {
		os.Remove(tmpPath)
		return ocispec.Descriptor{}, err
	}
	dest := filepath.Join(dir, dgst.Hex())
	if fi, err := os.Stat(dest); err == nil && fi.Size() == total {
		os.Remove(tmpPath) // already exists (idempotent)
	} else if err := os.Rename(tmpPath, dest); err != nil {
		os.Remove(tmpPath)
		return ocispec.Descriptor{}, err
	}

	return ocispec.Descriptor{
		// Use the CNCF model-spec weight media type so the stored manifest is
		// spec-compliant.  llmman's serve layer detection falls back to checking
		// the org.cncf.model.filepath annotation for ".gguf", so old manifests
		// (application/vnd.docker.ai.gguf.v3) still work via the other check.
		MediaType: "application/vnd.cncf.model.weight.v1.raw",
		Digest:    dgst,
		Size:      total,
		Annotations: map[string]string{
			"org.cncf.model.filepath": filepath.Base(file.Path),
		},
	}, nil
}

// ---------------------------------------------------------------------------
// storeHFAsOCI — wrap the GGUF blob in a CNCF model-spec OCI manifest
// ---------------------------------------------------------------------------

// cncfModelConfig is the required structure for application/vnd.cncf.model.config.v1+json.
type cncfModelConfig struct {
	Descriptor struct{} `json:"descriptor"`
	Config     struct {
		Format string `json:"format,omitempty"`
	} `json:"config"`
	ModelFS struct {
		Type    string   `json:"type"`
		DiffIDs []string `json:"diffIds"`
	} `json:"modelfs"`
}

func storeHFAsOCI(layoutDir, ref, modelRepo, filename string, ggufDesc ocispec.Descriptor) error {
	// Build a conformant CNCF model-spec config blob.
	var cfg cncfModelConfig
	cfg.Config.Format = "gguf"
	cfg.ModelFS.Type = "layers"
	cfg.ModelFS.DiffIDs = []string{ggufDesc.Digest.String()}

	cfgData, err := json.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("marshal CNCF model config: %w", err)
	}
	configDesc, err := writeBlob(layoutDir, "application/vnd.cncf.model.config.v1+json", cfgData)
	if err != nil {
		return fmt.Errorf("write CNCF model config: %w", err)
	}

	manifest := ocispec.Manifest{
		Versioned:    specs.Versioned{SchemaVersion: 2},
		MediaType:    ocispec.MediaTypeImageManifest,
		ArtifactType: "application/vnd.cncf.model.manifest.v1+json",
		Config:       configDesc,
		Layers:       []ocispec.Descriptor{ggufDesc},
		Annotations: map[string]string{
			"org.cncf.model.filepath": filepath.Base(filename),
			"ai.model.repo":           modelRepo,
		},
	}
	manifestData, err := json.Marshal(manifest)
	if err != nil {
		return fmt.Errorf("marshal OCI manifest: %w", err)
	}
	manifestDesc, err := writeBlob(layoutDir, ocispec.MediaTypeImageManifest, manifestData)
	if err != nil {
		return fmt.Errorf("write OCI manifest: %w", err)
	}
	return updateIndex(layoutDir, ref, manifestDesc)
}

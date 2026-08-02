

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

// isOCIHost reports whether host should be treated as an OCI Distribution
// registry (true) or a HuggingFace-compatible host (false): known-host
// shortcuts first, then a live /v2/ probe as the fallback for anything else.
// This is the single decision table `llmman pull`'s docker and podman
// backends (backend_docker.go, backend_podman.go) and `llmman transfer`'s
// source classification (transfer_common.go) all need to agree on.
func isOCIHost(ctx context.Context, host string) bool {
	if isKnownHFHost(host) {
		return false
	}
	if isKnownOCIHost(host) {
		return true
	}
	probeClient := &http.Client{Timeout: 5 * time.Second}
	return isOCIRegistry(ctx, probeClient, host)
}

// classifyPullRef runs the URI-scheme dispatch and OCI-vs-HuggingFace host
// classification shared by both backends' pullToLayout (backend_docker.go,
// backend_podman.go). If handled is true, dispatchPull has already fully
// processed ref (via one of hf://, ms://, ngc://, s3://, gs://, or a local
// path) and the caller should return dispatchErr immediately without doing
// anything else. Otherwise normalizedRef is ref with a ":latest" tag
// defaulted in, and isOCI reports whether normalizedRef's host should be
// pulled via the OCI registry protocol (true) or the shared HF path (false).
func classifyPullRef(ctx context.Context, ref, layoutDir string) (normalizedRef string, isOCI, handled bool, dispatchErr error) {
	if handled, err := dispatchPull(ctx, ref, layoutDir); handled {
		return ref, false, true, err
	}

	// Normalize: append :latest if reference has no tag or digest.
	if strings.LastIndex(ref, ":") <= strings.LastIndex(ref, "/") {
		ref = ref + ":latest"
	}

	host := strings.SplitN(ref, "/", 2)[0]
	return ref, isOCIHost(ctx, host), false, nil
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

// hfAPIClient returns the HTTP client used for HuggingFace metadata requests
// (commit lookup, file listing, HEAD digest probes) — a short total timeout
// suffices since these responses are small. Shared by pullHF (this file) and
// dockerTransferHF (transfer_docker.go), which both need one.
func hfAPIClient() *http.Client {
	return &http.Client{Timeout: 120 * time.Second}
}

// hfDownloadClient returns the HTTP client used for actually downloading (or
// streaming) HuggingFace file content: no body read timeout so large files
// can transfer without a deadline, but connection and header timeouts still
// prevent hanging on a stalled server. Mirrors llama.cpp's
// common/download.cpp approach. Shared by pullHF (this file) and
// dockerTransferHF (transfer_docker.go).
func hfDownloadClient() *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			DialContext: (&net.Dialer{
				Timeout:   30 * time.Second,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			TLSHandshakeTimeout:   30 * time.Second,
			ResponseHeaderTimeout: 60 * time.Second,
		},
	}
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

	// Do NOT follow the redirect: huggingface.co sets X-Linked-Etag/
	// X-Linked-Size on its own redirecting response (pointing at the real
	// content's sha256/size before it hands off to a CDN); the CDN's own
	// response has neither header and sets an unrelated ETag of its own
	// (its storage object's identifier, not a content hash we can trust)
	// — using that instead silently produces a wrong digest that a
	// registry push then rejects as DIGEST_INVALID after fully uploading
	// the (correct) bytes under the (wrong) declared name.
	noRedirect := &http.Client{
		Transport: client.Transport,
		Timeout:   client.Timeout,
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
	resp, err := noRedirect.Do(req)
	if err != nil {
		return "", 0, false, fmt.Errorf("HEAD %s: %w", url, err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 && (resp.StatusCode < 300 || resp.StatusCode >= 400) {
		return "", 0, false, fmt.Errorf("HEAD %s: HTTP %d", url, resp.StatusCode)
	}

	// Read size first and independently of digest validity below: callers
	// that fall back to buffering (small, non-LFS files) still want an
	// accurate progress-bar size even though the digest can't be trusted
	// yet — see transfer_docker.go's streamHFFileToRegistry.
	sizeStr := resp.Header.Get("X-Linked-Size")
	if sizeStr == "" && resp.StatusCode == 200 {
		// Only trust a plain Content-Length when there was no redirect —
		// a redirect response's Content-Length describes its own (tiny)
		// body, not the file being redirected to.
		sizeStr = resp.Header.Get("Content-Length")
	}
	if sizeStr != "" {
		if n, convErr := parseInt64(sizeStr); convErr == nil {
			size = n
		}
	}

	etag := resp.Header.Get("X-Linked-Etag")
	if etag == "" {
		etag = resp.Header.Get("ETag")
	}
	etag = strings.TrimPrefix(etag, "W/")
	etag = strings.Trim(etag, `"`)
	if len(etag) != 64 {
		return "", size, false, nil // not a sha256 — not LFS, caller should buffer instead
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

// reportCached prints "Cached <label>" and returns true if ref is already
// fully cached in layoutDir (manifest blob + all layer blobs present) — the
// signal every pull entry point in this package uses to skip all network
// I/O. label defaults to the cached name cachedLayerName itself resolved
// (e.g. a GGUF filename) when the empty string is passed.
func reportCached(layoutDir, ref, label string) bool {
	name := cachedLayerName(layoutDir, ref)
	if name == "" {
		return false
	}
	if label == "" {
		label = name
	}
	fmt.Fprintf(os.Stderr, "Cached   %s\n", label)
	return true
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
	if reportCached(layoutDir, ref, "") {
		return nil
	}

	endpoint := hfEndpoint(host)
	token := hfToken()

	apiClient := hfAPIClient()
	dlClient := hfDownloadClient()

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

// selectDownloadableHFFiles filters files down to the plain files that
// shouldDownloadSafetensors accepts, ignoring directories. Shared by
// pullHFSafetensors (this file) and dockerTransferHF (transfer_docker.go).
func selectDownloadableHFFiles(files []hfFile) []hfFile {
	var out []hfFile
	for _, f := range files {
		if f.Type == "file" && shouldDownloadSafetensors(f.Path) {
			out = append(out, f)
		}
	}
	return out
}

func pullHFSafetensors(
	ctx context.Context,
	client *http.Client,
	ref, layoutDir, endpoint, owner, repo, commit, token string,
	files []hfFile,
) error {
	toDownload := selectDownloadableHFFiles(files)
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
	manifestDesc, err := buildCNCFManifest(layoutBlobSink(layoutDir), "safetensors", modelRepo, "", layers)
	if err != nil {
		return err
	}
	return updateIndex(layoutDir, ref, manifestDesc)
}

// ---------------------------------------------------------------------------
// downloadHFBlob — HTTP download with resume, retry, and stall detection.
// Mirrors llama.cpp common/download.cpp: 3 attempts, 2s/4s backoff.
//
// dlMaxAttempts/dlRetryBase/dlStallTimeout/stallReader/isHTTP4xx/retryStream
// now live in shared_oci.go — they're used here for the local-disk pull
// path (which can resume a partial download with a Range request against
// its own .part file) and by transfer_docker.go's streaming push path
// (which, lacking a resumable registry upload — see that file's own
// comment on containerd's docker Pusher — can only retry a failed blob
// from scratch, not resume it, but still benefits from the same
// backoff/stall/permanent-vs-transient logic).
// ---------------------------------------------------------------------------

func downloadHFBlob(ctx context.Context, client *http.Client, url, token, layoutDir, owner, repo, commit string, file hfFile) (ocispec.Descriptor, error) {
	if err := os.MkdirAll(filepath.Join(layoutDir, "blobs"), 0o755); err != nil {
		return ocispec.Descriptor{}, err
	}

	sanitize := strings.NewReplacer("/", "_", ":", "_", ".", "_")
	tmpKey := sanitize.Replace(owner + "_" + repo + "_" + commit[:12] + "_" + filepath.Base(file.Path))
	tmpPath := filepath.Join(layoutDir, "blobs", "hf-"+tmpKey+".part")

	label := "Pulling  " + filepath.Base(file.Path)
	doneLbl := "Pulled   " + filepath.Base(file.Path)
	prog := newProgressPool(80)
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

	f, digester, startOffset, err := openForResume(tmpPath, startOffset)
	if err != nil {
		return ocispec.Descriptor{}, err
	}

	// Wrap with stall detector: cancel attemptCtx if no bytes for 60s.
	sr := newStallReader(resp.Body, dlStallTimeout, cancel)
	defer sr.stop()

	proxyRC := proxyOrNop(bar, sr)
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
	manifestDesc, err := buildCNCFManifest(layoutBlobSink(layoutDir), "gguf", modelRepo, filepath.Base(filename), []ocispec.Descriptor{ggufDesc})
	if err != nil {
		return err
	}
	return updateIndex(layoutDir, ref, manifestDesc)
}

// ---------------------------------------------------------------------------
// buildCNCFManifest — shared CNCF model-spec manifest+config construction,
// used by both the local-OCI-layout store path above (storeHFAsOCI,
// storeSafetensorsAsOCI) and transfer_docker.go's direct-to-registry push
// path (pushCNCFSingleManifest, pushCNCFMultiManifest). The two paths differ
// only in *where* a built blob ends up — a local content-addressed layout
// vs. streamed straight to a registry pusher — which is exactly what the
// cncfBlobSink abstraction below exists to hide.
// ---------------------------------------------------------------------------

// cncfBlobSink stores one marshaled CNCF blob (config or manifest JSON) and
// returns its descriptor.
type cncfBlobSink func(mediaType string, data []byte) (ocispec.Descriptor, error)

// layoutBlobSink is the cncfBlobSink for storing blobs in a local OCI layout
// directory.
func layoutBlobSink(layoutDir string) cncfBlobSink {
	return func(mediaType string, data []byte) (ocispec.Descriptor, error) {
		return writeBlob(layoutDir, mediaType, data)
	}
}

// buildCNCFManifest builds a conformant CNCF model-spec config blob and
// manifest referencing layers, storing each via sink, and returns the
// manifest's descriptor. filepathAnnotation sets the manifest-level
// org.cncf.model.filepath annotation for the single-weight-file case (GGUF);
// pass "" for the multi-layer safetensors case, which only sets
// ai.model.repo.
func buildCNCFManifest(sink cncfBlobSink, format, modelRepo, filepathAnnotation string, layers []ocispec.Descriptor) (ocispec.Descriptor, error) {
	var cfg cncfModelConfig
	cfg.Config.Format = format
	cfg.ModelFS.Type = "layers"
	for _, l := range layers {
		cfg.ModelFS.DiffIDs = append(cfg.ModelFS.DiffIDs, l.Digest.String())
	}
	cfgData, err := json.Marshal(cfg)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("marshal CNCF model config: %w", err)
	}
	configDesc, err := sink("application/vnd.cncf.model.config.v1+json", cfgData)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("store CNCF model config: %w", err)
	}

	annotations := map[string]string{"ai.model.repo": modelRepo}
	if filepathAnnotation != "" {
		annotations["org.cncf.model.filepath"] = filepathAnnotation
	}
	manifest := ocispec.Manifest{
		Versioned:    specs.Versioned{SchemaVersion: 2},
		MediaType:    ocispec.MediaTypeImageManifest,
		ArtifactType: "application/vnd.cncf.model.manifest.v1+json",
		Config:       configDesc,
		Layers:       layers,
		Annotations:  annotations,
	}
	manifestData, err := json.Marshal(manifest)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("marshal CNCF manifest: %w", err)
	}
	manifestDesc, err := sink(ocispec.MediaTypeImageManifest, manifestData)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("store CNCF manifest: %w", err)
	}
	return manifestDesc, nil
}

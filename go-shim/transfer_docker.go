//go:build !podman

// transfer_docker.go — `llmman transfer`'s docker/containerd-backed
// implementation: llmman's equivalent of `skopeo copy`.
//
// skopeo/containers-image's copy.Image() never fully materializes an image
// locally before pushing it: it already knows every blob's digest and size
// up front from the source's own OCI manifest, so it can open a reader on
// the source blob and a writer on the destination blob at the same time
// and stream one directly into the other (see the copy package's
// GetBlob/PutBlob pairing). This file reproduces that property for two
// cases:
//
//   - OCI registry → OCI registry (dockerTransferOCI): trivial — the
//     source manifest already gives every blob's digest/size, so it's a
//     straight Fetcher → Pusher stream per blob, exactly like skopeo.
//
//   - HuggingFace → OCI registry (dockerTransferHF): harder, because there
//     is no pre-existing manifest to read a digest from. But a HEAD
//     request against an LFS-tracked file's resolve URL exposes the real
//     content sha256 via the X-Linked-Etag header *before* any bytes are
//     downloaded (see hf.go's hfHeadMetadata) — which gives exactly the
//     "digest known ahead of time" property a registry push needs, the
//     same way an OCI manifest would. That's what makes streaming a
//     multi-gigabyte GGUF file straight from huggingface.co into a
//     registry possible without ever writing it to local disk.
//
// Anything else `llmman pull` understands (ms://, ngc://, s3://, gs://, a
// local path) falls back to transferViaStaging: pull into a throwaway
// local OCI layout, then push from it, exactly like `llmman transfer` did
// before this file existed. Reimplementing zero-disk streaming for every
// one of those source kinds isn't worth it yet — none of them are the
// large-model-file case this exists for.
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
	"time"

	"github.com/containerd/containerd/v2/core/remotes"
	digest "github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

func dockerTransfer(ctx context.Context, source, destination string) error {
	// A tagless destination (e.g. "docker.io/owner/repo") must default to
	// :latest here explicitly: unlike a local OCI layout's index.json
	// (which always has some ref-name annotation to look up),
	// resolver.Pusher parses the ref as given, and a repository object
	// left empty pushes the manifest addressable only by digest — no tag
	// is ever created, silently, so a plain `docker pull owner/repo`
	// afterwards would find nothing.
	destination = normalizeTag(destination)
	kind, normalized := classifySource(ctx, source)
	switch kind {
	case sourceOCI:
		return dockerTransferOCI(ctx, normalized, destination)
	case sourceHF:
		return dockerTransferHF(ctx, normalized, destination)
	default:
		return transferViaStaging(ctx, source, destination)
	}
}

// transferViaStaging is the fallback path for source kinds that don't have
// a streaming implementation (yet): pull into a throwaway local OCI
// layout, then push from it, mirroring what `llmman transfer` did before
// this file existed.
func transferViaStaging(ctx context.Context, source, destination string) error {
	tmp, err := os.MkdirTemp("", "llmman-transfer-")
	if err != nil {
		return fmt.Errorf("create staging directory: %w", err)
	}
	defer os.RemoveAll(tmp)

	if err := pullToLayout(ctx, source, tmp); err != nil {
		return err
	}
	return pushToRegistry(ctx, tmp, destination)
}

// ---------------------------------------------------------------------------
// OCI registry → OCI registry
// ---------------------------------------------------------------------------

func dockerTransferOCI(ctx context.Context, source, destination string) error {
	resolver := newResolver(ctx)
	name, manifestDesc, err := resolver.Resolve(ctx, source)
	if err != nil {
		return fmt.Errorf("resolve %s: %w", source, err)
	}
	fetcher, err := resolver.Fetcher(ctx, name)
	if err != nil {
		return fmt.Errorf("create fetcher: %w", err)
	}
	pusher, err := resolver.Pusher(ctx, destination)
	if err != nil {
		return fmt.Errorf("create pusher: %w", err)
	}

	rc, err := fetcher.Fetch(ctx, manifestDesc)
	if err != nil {
		return fmt.Errorf("fetch manifest: %w", err)
	}
	manifestData, err := io.ReadAll(rc)
	rc.Close()
	if err != nil {
		return fmt.Errorf("read manifest: %w", err)
	}

	var manifest ocispec.Manifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		// An image index (manifest list): push it as-is. Per-instance
		// selection (skopeo's --multi-arch) isn't implemented here.
		return pushBytes(ctx, pusher, manifestDesc, manifestData)
	}

	streamOne := func(desc ocispec.Descriptor) error {
		rc, err := fetcher.Fetch(ctx, desc)
		if err != nil {
			return fmt.Errorf("fetch %s: %w", desc.Digest, err)
		}
		defer rc.Close()
		if err := pushStream(ctx, pusher, desc, rc); err != nil {
			return fmt.Errorf("push %s: %w", desc.Digest, err)
		}
		fmt.Fprintf(os.Stderr, "Copied   %s\n", shortDigest(desc.Digest))
		return nil
	}

	for _, layer := range manifest.Layers {
		if err := streamOne(layer); err != nil {
			return err
		}
	}
	if err := streamOne(manifest.Config); err != nil {
		return err
	}
	return pushBytes(ctx, pusher, manifestDesc, manifestData)
}

// ---------------------------------------------------------------------------
// HuggingFace → OCI registry
// ---------------------------------------------------------------------------

func dockerTransferHF(ctx context.Context, ref, destination string) error {
	host, owner, repo, tag, err := parseHFRef(ref)
	if err != nil {
		return err
	}
	endpoint := hfEndpoint(host)
	token := hfToken()

	apiClient := &http.Client{Timeout: 120 * time.Second}
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

	commit, err := hfFetchCommit(ctx, apiClient, endpoint, owner, repo, token)
	if err != nil {
		return err
	}
	files, err := hfFetchFiles(ctx, apiClient, endpoint, owner, repo, commit, token)
	if err != nil {
		return err
	}

	resolver := newResolver(ctx)
	pusher, err := resolver.Pusher(ctx, destination)
	if err != nil {
		return fmt.Errorf("create pusher: %w", err)
	}

	// Try GGUF first; fall back to safetensors if the repo has none — same
	// selection logic pullHF uses.
	if chosen, err := selectGGUF(files, tag); err == nil {
		desc, err := streamHFFileToRegistry(
			ctx, dlClient, pusher, endpoint, owner, repo, commit, token,
			chosen, "application/vnd.cncf.model.weight.v1.raw",
		)
		if err != nil {
			return err
		}
		return pushCNCFSingleManifest(ctx, pusher, "gguf", owner+"/"+repo, chosen.Path, desc)
	}

	var toSend []hfFile
	for _, f := range files {
		if f.Type == "file" && shouldDownloadSafetensors(f.Path) {
			toSend = append(toSend, f)
		}
	}
	if len(toSend) == 0 {
		return fmt.Errorf("no model files found in repository %s/%s", owner, repo)
	}
	var layers []ocispec.Descriptor
	for _, f := range toSend {
		desc, err := streamHFFileToRegistry(
			ctx, dlClient, pusher, endpoint, owner, repo, commit, token,
			f, safetensorsMediaType(f.Path),
		)
		if err != nil {
			return fmt.Errorf("transfer %s: %w", f.Path, err)
		}
		desc.Annotations = map[string]string{"org.cncf.model.filepath": f.Path}
		layers = append(layers, desc)
	}
	return pushCNCFMultiManifest(ctx, pusher, owner+"/"+repo, layers)
}

// streamHFFileToRegistry copies one HuggingFace file directly into the
// registry pusher. When the file's real content digest can be learned
// ahead of time via a HEAD request (true for essentially every real
// LFS-tracked weight file — see hfHeadMetadata), the GET response body is
// piped straight into the push with no buffering at all: the file never
// touches local disk or is ever fully held in memory. Otherwise (small,
// non-LFS files such as config.json or a tokenizer file, where the ETag is
// a git blob sha1, not a sha256 of the content) it's buffered in memory —
// still zero disk I/O, and harmless given how small these files are.
func streamHFFileToRegistry(
	ctx context.Context,
	client *http.Client,
	pusher remotes.Pusher,
	endpoint, owner, repo, commit, token string,
	file hfFile,
	mediaType string,
) (ocispec.Descriptor, error) {
	url := endpoint + owner + "/" + repo + "/resolve/" + commit + "/" + file.Path
	// org.cncf.model.filepath on the *layer* descriptor itself (not just
	// the manifest) is what cmd::serve's layer_filepath/is_gguf_layer
	// actually look at to recognize a servable GGUF/safetensors layer —
	// see downloadAttempt in hf.go, which sets the same annotation for
	// `llmman pull`'s local-layout path. Omitting it here doesn't fail
	// the transfer itself (the push succeeds either way), but leaves the
	// pushed image unservable by `llmman run`/`llmman serve` afterwards.
	annotations := map[string]string{"org.cncf.model.filepath": filepath.Base(file.Path)}

	if dgst, size, ok, err := hfHeadMetadata(ctx, client, url, token); err == nil && ok {
		desc := ocispec.Descriptor{MediaType: mediaType, Digest: dgst, Size: size, Annotations: annotations}
		if err := streamHFGet(ctx, client, url, token, pusher, desc); err != nil {
			return ocispec.Descriptor{}, fmt.Errorf("stream %s: %w", file.Path, err)
		}
		fmt.Fprintf(os.Stderr, "Transferred %s\n", filepath.Base(file.Path))
		return desc, nil
	}

	data, err := hfGetBytes(ctx, client, url, token)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("download %s: %w", file.Path, err)
	}
	desc := ocispec.Descriptor{
		MediaType:   mediaType,
		Digest:      digest.FromBytes(data),
		Size:        int64(len(data)),
		Annotations: annotations,
	}
	if err := pushBytes(ctx, pusher, desc, data); err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("push %s: %w", file.Path, err)
	}
	fmt.Fprintf(os.Stderr, "Transferred %s\n", filepath.Base(file.Path))
	return desc, nil
}

func streamHFGet(ctx context.Context, client *http.Client, url, token string, pusher remotes.Pusher, desc ocispec.Descriptor) error {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 && resp.StatusCode != 206 {
		return fmt.Errorf("GET %s: HTTP %d", url, resp.StatusCode)
	}
	return pushStream(ctx, pusher, desc, resp.Body)
}

func hfGetBytes(ctx context.Context, client *http.Client, url, token string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("GET %s: HTTP %d", url, resp.StatusCode)
	}
	return io.ReadAll(resp.Body)
}

// ---------------------------------------------------------------------------
// CNCF ModelPack manifest construction — pushed directly, never written
// to local disk (mirrors storeHFAsOCI/storeSafetensorsAsOCI in hf.go,
// which do the local-layout equivalent for `llmman pull`).
// ---------------------------------------------------------------------------

func pushCNCFSingleManifest(ctx context.Context, pusher remotes.Pusher, format, modelRepo, filename string, weightDesc ocispec.Descriptor) error {
	var cfg cncfModelConfig
	cfg.Config.Format = format
	cfg.ModelFS.Type = "layers"
	cfg.ModelFS.DiffIDs = []string{weightDesc.Digest.String()}
	cfgData, err := json.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("marshal CNCF model config: %w", err)
	}
	configDesc := ocispec.Descriptor{
		MediaType: "application/vnd.cncf.model.config.v1+json",
		Digest:    digest.FromBytes(cfgData),
		Size:      int64(len(cfgData)),
	}
	if err := pushBytes(ctx, pusher, configDesc, cfgData); err != nil {
		return fmt.Errorf("push CNCF model config: %w", err)
	}

	manifest := ocispec.Manifest{
		Versioned:    specs.Versioned{SchemaVersion: 2},
		MediaType:    ocispec.MediaTypeImageManifest,
		ArtifactType: "application/vnd.cncf.model.manifest.v1+json",
		Config:       configDesc,
		Layers:       []ocispec.Descriptor{weightDesc},
		Annotations: map[string]string{
			"org.cncf.model.filepath": filepath.Base(filename),
			"ai.model.repo":           modelRepo,
		},
	}
	manifestData, err := json.Marshal(manifest)
	if err != nil {
		return fmt.Errorf("marshal OCI manifest: %w", err)
	}
	manifestDesc := ocispec.Descriptor{
		MediaType: ocispec.MediaTypeImageManifest,
		Digest:    digest.FromBytes(manifestData),
		Size:      int64(len(manifestData)),
	}
	return pushBytes(ctx, pusher, manifestDesc, manifestData)
}

func pushCNCFMultiManifest(ctx context.Context, pusher remotes.Pusher, modelRepo string, layers []ocispec.Descriptor) error {
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
	configDesc := ocispec.Descriptor{
		MediaType: "application/vnd.cncf.model.config.v1+json",
		Digest:    digest.FromBytes(cfgData),
		Size:      int64(len(cfgData)),
	}
	if err := pushBytes(ctx, pusher, configDesc, cfgData); err != nil {
		return fmt.Errorf("push CNCF config: %w", err)
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
	manifestDesc := ocispec.Descriptor{
		MediaType: ocispec.MediaTypeImageManifest,
		Digest:    digest.FromBytes(manifestData),
		Size:      int64(len(manifestData)),
	}
	return pushBytes(ctx, pusher, manifestDesc, manifestData)
}

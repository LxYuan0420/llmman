//go:build !podman

package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/containerd/containerd/v2/core/content"
	"github.com/containerd/containerd/v2/core/remotes"
	"github.com/containerd/containerd/v2/core/remotes/docker"
	dockerconfig "github.com/containerd/containerd/v2/core/remotes/docker/config"
	remoteerrors "github.com/containerd/containerd/v2/core/remotes/errors"
	"github.com/containerd/errdefs"
	dockercliconfig "github.com/docker/cli/cli/config"
	clitypes "github.com/docker/cli/cli/config/types"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/vbauerster/mpb/v8"
	"golang.org/x/sync/errgroup"
)

// ---------------------------------------------------------------------------
// Credential helpers
// ---------------------------------------------------------------------------

func dockerCredentials(host string) (string, string, error) {
	cfg := dockercliconfig.LoadDefaultConfigFile(io.Discard)
	store := cfg.GetCredentialsStore(host)
	creds, err := store.Get(host)
	if err != nil {
		return "", "", nil // not an error — just not found
	}
	if creds.IdentityToken != "" {
		return "", creds.IdentityToken, nil
	}
	return creds.Username, creds.Password, nil
}

func newResolver(ctx context.Context) remotes.Resolver {
	return docker.NewResolver(docker.ResolverOptions{
		Hosts: dockerconfig.ConfigureHosts(ctx, dockerconfig.HostOptions{
			Credentials: dockerCredentials,
		}),
		Client: &http.Client{Timeout: 120 * time.Second},
	})
}

// describeErr enriches a containerd registry error with the response body,
// when there is one — containerd's own ErrUnexpectedStatus.Error() deliberately
// omits it (only logged at debug level), which is exactly the detail needed to
// tell "repository doesn't exist", "insufficient scope", and similar registry-
// side rejections apart from each other instead of a bare, unexplained status
// code.
func describeErr(err error) error {
	var ue remoteerrors.ErrUnexpectedStatus
	if errors.As(err, &ue) && len(ue.Body) > 0 {
		return fmt.Errorf("%w: %s", err, strings.TrimSpace(string(ue.Body)))
	}
	return err
}

// ---------------------------------------------------------------------------
// ociProvider implements content.Provider backed by an OCI layout directory.
// ---------------------------------------------------------------------------

type ociProvider struct{ dir string }

func (p *ociProvider) ReaderAt(ctx context.Context, desc ocispec.Descriptor) (content.ReaderAt, error) {
	path := blobPath(p.dir, desc.Digest)
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("blob %s: %w", desc.Digest, err)
	}
	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	return &fileReaderAt{f: f, size: fi.Size()}, nil
}

type fileReaderAt struct {
	f    *os.File
	size int64
}

func (r *fileReaderAt) ReadAt(p []byte, off int64) (int, error) { return r.f.ReadAt(p, off) }
func (r *fileReaderAt) Close() error                            { return r.f.Close() }
func (r *fileReaderAt) Size() int64                             { return r.size }

// pushBlob pushes a single blob from the OCI layout to the registry pusher.
func pushBlob(ctx context.Context, pusher remotes.Pusher, provider *ociProvider, desc ocispec.Descriptor) error {
	cw, err := pusher.Push(ctx, desc)
	if err != nil {
		if errdefs.IsAlreadyExists(err) {
			return nil
		}
		return describeErr(err)
	}
	defer cw.Close()

	ra, err := provider.ReaderAt(ctx, desc)
	if err != nil {
		return err
	}
	defer ra.Close()

	return describeErr(content.Copy(ctx, cw, io.NewSectionReader(ra, 0, ra.Size()), desc.Size, desc.Digest))
}

// pushBytes pushes an in-memory blob (a manifest or a small config/metadata
// file) directly to the registry pusher — no local file involved.
func pushBytes(ctx context.Context, pusher remotes.Pusher, desc ocispec.Descriptor, data []byte) error {
	cw, err := pusher.Push(ctx, desc)
	if err != nil {
		if errdefs.IsAlreadyExists(err) {
			return nil
		}
		return describeErr(err)
	}
	defer cw.Close()
	return describeErr(content.Copy(ctx, cw, bytes.NewReader(data), desc.Size, desc.Digest))
}

// pushStream pushes a blob whose digest and size are already known (see
// hfHeadMetadata) directly from a live reader — typically an in-flight HTTP
// GET response body — straight into the registry pusher, with no local
// file or full in-memory buffer at any point. This is what makes
// `llmman transfer` behave like `skopeo copy` for large HuggingFace files:
// bytes flow source → destination without ever landing on disk in between.
func pushStream(ctx context.Context, pusher remotes.Pusher, desc ocispec.Descriptor, r io.Reader) error {
	cw, err := pusher.Push(ctx, desc)
	if err != nil {
		if errdefs.IsAlreadyExists(err) {
			return nil
		}
		return describeErr(err)
	}
	defer cw.Close()
	return describeErr(content.Copy(ctx, cw, r, desc.Size, desc.Digest))
}

// ---------------------------------------------------------------------------
// Exported CGO functions
// ---------------------------------------------------------------------------

// llmman_login stores credentials for a registry in the Docker credential store.
//
//export llmman_login
func llmman_login(cServer, cUsername, cPassword *C.char) *C.char {
	server := C.GoString(cServer)
	username := C.GoString(cUsername)
	password := C.GoString(cPassword)

	cfg := dockercliconfig.LoadDefaultConfigFile(io.Discard)
	store := cfg.GetCredentialsStore(server)

	if err := store.Store(clitypes.AuthConfig{
		ServerAddress: server,
		Username:      username,
		Password:      password,
	}); err != nil {
		return errResp(fmt.Errorf("store credentials: %w", err))
	}
	if err := cfg.Save(); err != nil {
		return errResp(fmt.Errorf("save config: %w", err))
	}
	return okResp("")
}

// llmman_logout removes credentials for a registry from the Docker credential store.
//
//export llmman_logout
func llmman_logout(cServer *C.char) *C.char {
	server := C.GoString(cServer)

	cfg := dockercliconfig.LoadDefaultConfigFile(io.Discard)
	store := cfg.GetCredentialsStore(server)
	if err := store.Erase(server); err != nil {
		return errResp(fmt.Errorf("erase credentials: %w", err))
	}
	if err := cfg.Save(); err != nil {
		return errResp(fmt.Errorf("save config: %w", err))
	}
	return okResp("")
}

// llmman_push pushes an image from a local OCI layout directory to a registry.
// layoutDir is the path to the OCI layout root; ref is the full registry reference.
//
//export llmman_push
func llmman_push(cLayoutDir, cRef *C.char) *C.char {
	if err := pushToRegistry(context.Background(), C.GoString(cLayoutDir), C.GoString(cRef)); err != nil {
		return errResp(err)
	}
	return okResp("")
}

// pushToRegistry is llmman_push's implementation, factored out so
// llmman_transfer's staging-directory fallback (see transferViaStaging in
// transfer.go) can reuse it without going through CGO.
func pushToRegistry(ctx context.Context, layoutDir, ref string) error {
	// Locate the manifest in the local index
	idx, err := readIndex(layoutDir)
	if err != nil {
		return fmt.Errorf("read OCI index: %w", err)
	}
	tag := tagFromRef(ref)
	manifestDesc, err := findManifestDesc(idx, tag)
	if err != nil {
		return err
	}

	// Read manifest
	manifestData, err := readBlob(layoutDir, manifestDesc.Digest)
	if err != nil {
		return fmt.Errorf("read manifest blob: %w", err)
	}
	var manifest ocispec.Manifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		return fmt.Errorf("parse manifest: %w", err)
	}

	resolver := newResolver(ctx)
	pusher, err := resolver.Pusher(ctx, ref)
	if err != nil {
		return fmt.Errorf("create pusher: %w", err)
	}
	provider := &ociProvider{dir: layoutDir}

	// Push layers
	for _, layer := range manifest.Layers {
		if err := pushBlob(ctx, pusher, provider, layer); err != nil {
			return fmt.Errorf("push layer %s: %w", layer.Digest, err)
		}
	}
	// Push config
	if err := pushBlob(ctx, pusher, provider, manifest.Config); err != nil {
		return fmt.Errorf("push config: %w", err)
	}
	// Push manifest
	if err := pushBlob(ctx, pusher, provider, manifestDesc); err != nil {
		return fmt.Errorf("push manifest: %w", err)
	}
	return nil
}

// llmman_pull pulls an image from a registry into a local OCI layout directory.
//
//export llmman_pull
func llmman_pull(cRef, cLayoutDir *C.char) *C.char {
	if err := pullToLayout(context.Background(), C.GoString(cRef), C.GoString(cLayoutDir)); err != nil {
		return errResp(err)
	}
	return okResp("")
}

// pullToLayout is llmman_pull's implementation, factored out so
// llmman_transfer's staging-directory fallback can reuse it.
func pullToLayout(ctx context.Context, ref, layoutDir string) error {
	// URI-scheme dispatch: hf://, ms://, ngc://, s3://, gs://, /absolute/path.
	// These bypass the OCI registry probe and HF host detection below.
	if handled, err := dispatchPull(ctx, ref, layoutDir); handled {
		return err
	}

	// Normalize: append :latest if reference has no tag or digest
	if strings.LastIndex(ref, ":") <= strings.LastIndex(ref, "/") {
		ref = ref + ":latest"
	}

	// Detect backend: probe the host to decide OCI registry vs HuggingFace-compatible.
	// Known OCI hosts skip the probe; known HF hosts go straight to HF.
	// Unknown hosts are probed via the OCI Distribution /v2/ endpoint.
	host := strings.SplitN(ref, "/", 2)[0]
	if !isKnownOCIHost(host) {
		probeClient := &http.Client{Timeout: 5 * time.Second}
		if isKnownHFHost(host) || !isOCIRegistry(ctx, probeClient, host) {
			return pullHF(ctx, ref, layoutDir)
		}
	}

	if err := ensureLayout(layoutDir); err != nil {
		return fmt.Errorf("init OCI layout: %w", err)
	}

	resolver := newResolver(ctx)
	// Deliberately not "resolve %s: %w" — containerd's own resolve errors
	// (e.g. errdefs.ErrNotFound) already embed ref themselves, and every
	// caller of llmman_pull (the Rust daemon's /api/pull handler) already
	// prefixes whatever error comes back with the reference it asked for.
	// Including ref here too just repeats it two or three times over.
	name, manifestDesc, err := resolver.Resolve(ctx, ref)
	if err != nil {
		return fmt.Errorf("resolve: %w", err)
	}
	fetcher, err := resolver.Fetcher(ctx, name)
	if err != nil {
		return fmt.Errorf("create fetcher: %w", err)
	}

	// Fetch and store manifest
	rc, err := fetcher.Fetch(ctx, manifestDesc)
	if err != nil {
		return fmt.Errorf("fetch manifest: %w", err)
	}
	manifestData, err := io.ReadAll(rc)
	rc.Close()
	if err != nil {
		return fmt.Errorf("read manifest: %w", err)
	}
	if _, err := writeBlob(layoutDir, manifestDesc.MediaType, manifestData); err != nil {
		return fmt.Errorf("write manifest blob: %w", err)
	}

	// Decode manifest to learn about layers and config
	var manifest ocispec.Manifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		// Could be an image index — store and return
		return updateIndex(layoutDir, ref, manifestDesc)
	}

	// Fetch config
	configRC, err := fetcher.Fetch(ctx, manifest.Config)
	if err != nil {
		return fmt.Errorf("fetch config: %w", err)
	}
	configData, readErr := io.ReadAll(configRC)
	configRC.Close()
	if readErr != nil {
		return fmt.Errorf("read config: %w", readErr)
	}
	if _, err := writeBlob(layoutDir, manifest.Config.MediaType, configData); err != nil {
		return fmt.Errorf("write config blob: %w", err)
	}

	// Fetch layers in parallel — up to 6 concurrent downloads, matching podman's
	// default maxParallelDownloads.  All bars share one mpb.Progress; OnComplete
	// decorators flip each bar to "Pulled   <digest>" when done so the final static
	// line is always correct regardless of render-tick timing.
	const maxParallel = 6
	prog := mpb.New(
		mpb.WithWidth(80),
		mpb.WithOutput(os.Stderr),
		mpb.WithRefreshRate(180*time.Millisecond),
	)
	sem := make(chan struct{}, maxParallel)
	g, gctx := errgroup.WithContext(ctx)
	var barMu sync.Mutex // serialise bar creation so order matches layer order
	for _, layer := range manifest.Layers {
		layer := layer // capture
		shortDigest := layer.Digest.Hex()
		if len(shortDigest) > 12 {
			shortDigest = shortDigest[:12]
		}
		if blobExists(layoutDir, layer) {
			fmt.Fprintf(prog, "Cached   %s\n", shortDigest)
			continue
		}
		// Create the bar before launching the goroutine so bars appear in
		// manifest order even when downloads finish out of order.
		barMu.Lock()
		bar := addLayerBar(prog, "Pulling  "+shortDigest, "Pulled   "+shortDigest, layer.Size)
		barMu.Unlock()
		sem <- struct{}{}
		g.Go(func() error {
			defer func() { <-sem }()
			layerRC, err := fetcher.Fetch(gctx, layer)
			if err != nil {
				bar.Abort(false)
				return fmt.Errorf("fetch layer %s: %w", layer.Digest, err)
			}
			// Resume from an existing partial download: seek the HTTP reader to
			// the already-downloaded offset (containerd's httpReadSeeker issues a
			// Range: bytes=N- request, or discards N bytes if the server doesn't
			// support range requests) and pre-fill the progress bar.
			partOffset := int64(0)
			partPath := blobPath(layoutDir, layer.Digest) + ".part"
			if fi, statErr := os.Stat(partPath); statErr == nil && fi.Size() > 0 {
				if seeker, ok := layerRC.(io.ReadSeeker); ok {
					if _, seekErr := seeker.Seek(fi.Size(), io.SeekStart); seekErr == nil {
						partOffset = fi.Size()
						bar.IncrInt64(partOffset)
					}
				}
			}
			proxyRC := bar.ProxyReader(layerRC)
			if proxyRC == nil { // bar already done (zero-size layer)
				proxyRC = io.NopCloser(layerRC)
			}
			_, writeErr := writeBlobStream(layoutDir, layer.MediaType, proxyRC, layer.Size, layer.Digest, partOffset)
			proxyRC.Close()
			if writeErr != nil {
				bar.Abort(false)
				return fmt.Errorf("write layer %s: %w", layer.Digest, writeErr)
			}
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		prog.Wait()
		return err
	}
	prog.Wait()

	return updateIndex(layoutDir, ref, manifestDesc)
}

// llmman_inspect fetches and returns the raw manifest JSON for a remote reference.
//
//export llmman_inspect
func llmman_inspect(cRef *C.char) *C.char {
	ref := C.GoString(cRef)
	ctx := context.Background()

	resolver := newResolver(ctx)
	name, manifestDesc, err := resolver.Resolve(ctx, ref)
	if err != nil {
		return errResp(fmt.Errorf("resolve: %w", err))
	}
	fetcher, err := resolver.Fetcher(ctx, name)
	if err != nil {
		return errResp(fmt.Errorf("create fetcher: %w", err))
	}
	rc, err := fetcher.Fetch(ctx, manifestDesc)
	if err != nil {
		return errResp(fmt.Errorf("fetch manifest: %w", err))
	}
	data, err := io.ReadAll(rc)
	rc.Close()
	if err != nil {
		return errResp(fmt.Errorf("read manifest: %w", err))
	}

	// Pretty-print
	var buf bytes.Buffer
	if err := json.Indent(&buf, data, "", "  "); err != nil {
		return okResp(string(data))
	}
	return okResp(buf.String())
}

// llmman_transfer copies an image directly from source to destination —
// llmman's equivalent of `skopeo copy` — without ever writing it to the
// persistent local store. See transfer.go for the three strategies this
// picks between (streamed OCI→OCI, streamed HuggingFace→OCI, and a
// staging-directory fallback for everything else) and why each exists.
//
//export llmman_transfer
func llmman_transfer(cSource, cDestination *C.char) *C.char {
	if err := dockerTransfer(context.Background(), C.GoString(cSource), C.GoString(cDestination)); err != nil {
		return errResp(err)
	}
	return okResp("")
}

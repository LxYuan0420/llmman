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

// ---------------------------------------------------------------------------
// OCI layout helpers
// ---------------------------------------------------------------------------

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
		return err
	}
	defer cw.Close()

	ra, err := provider.ReaderAt(ctx, desc)
	if err != nil {
		return err
	}
	defer ra.Close()

	return content.Copy(ctx, cw, io.NewSectionReader(ra, 0, ra.Size()), desc.Size, desc.Digest)
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
	layoutDir := C.GoString(cLayoutDir)
	ref := C.GoString(cRef)
	ctx := context.Background()

	// Locate the manifest in the local index
	idx, err := readIndex(layoutDir)
	if err != nil {
		return errResp(fmt.Errorf("read OCI index: %w", err))
	}
	tag := tagFromRef(ref)
	manifestDesc, err := findManifestDesc(idx, tag)
	if err != nil {
		return errResp(err)
	}

	// Read manifest
	manifestData, err := readBlob(layoutDir, manifestDesc.Digest)
	if err != nil {
		return errResp(fmt.Errorf("read manifest blob: %w", err))
	}
	var manifest ocispec.Manifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		return errResp(fmt.Errorf("parse manifest: %w", err))
	}

	resolver := newResolver(ctx)
	pusher, err := resolver.Pusher(ctx, ref)
	if err != nil {
		return errResp(fmt.Errorf("create pusher: %w", err))
	}
	provider := &ociProvider{dir: layoutDir}

	// Push layers
	for _, layer := range manifest.Layers {
		if err := pushBlob(ctx, pusher, provider, layer); err != nil {
			return errResp(fmt.Errorf("push layer %s: %w", layer.Digest, err))
		}
	}
	// Push config
	if err := pushBlob(ctx, pusher, provider, manifest.Config); err != nil {
		return errResp(fmt.Errorf("push config: %w", err))
	}
	// Push manifest
	if err := pushBlob(ctx, pusher, provider, manifestDesc); err != nil {
		return errResp(fmt.Errorf("push manifest: %w", err))
	}
	return okResp("")
}

// llmman_pull pulls an image from a registry into a local OCI layout directory.
//
//export llmman_pull
func llmman_pull(cRef, cLayoutDir *C.char) *C.char {
	ref := C.GoString(cRef)
	layoutDir := C.GoString(cLayoutDir)
	ctx := context.Background()

	// URI-scheme dispatch: hf://, ms://, ngc://, s3://, gs://, /absolute/path.
	// These bypass the OCI registry probe and HF host detection below.
	if handled, err := dispatchPull(ctx, ref, layoutDir); handled {
		if err != nil {
			return errResp(err)
		}
		return okResp("")
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
			if err := pullHF(ctx, ref, layoutDir); err != nil {
				return errResp(err)
			}
			return okResp("")
		}
	}

	if err := ensureLayout(layoutDir); err != nil {
		return errResp(fmt.Errorf("init OCI layout: %w", err))
	}

	resolver := newResolver(ctx)
	// Deliberately not "resolve %s: %w" — containerd's own resolve errors
	// (e.g. errdefs.ErrNotFound) already embed ref themselves, and every
	// caller of llmman_pull (the Rust daemon's /api/pull handler) already
	// prefixes whatever error comes back with the reference it asked for.
	// Including ref here too just repeats it two or three times over.
	name, manifestDesc, err := resolver.Resolve(ctx, ref)
	if err != nil {
		return errResp(fmt.Errorf("resolve: %w", err))
	}
	fetcher, err := resolver.Fetcher(ctx, name)
	if err != nil {
		return errResp(fmt.Errorf("create fetcher: %w", err))
	}

	// Fetch and store manifest
	rc, err := fetcher.Fetch(ctx, manifestDesc)
	if err != nil {
		return errResp(fmt.Errorf("fetch manifest: %w", err))
	}
	manifestData, err := io.ReadAll(rc)
	rc.Close()
	if err != nil {
		return errResp(fmt.Errorf("read manifest: %w", err))
	}
	if _, err := writeBlob(layoutDir, manifestDesc.MediaType, manifestData); err != nil {
		return errResp(fmt.Errorf("write manifest blob: %w", err))
	}

	// Decode manifest to learn about layers and config
	var manifest ocispec.Manifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		// Could be an image index — store and return
		if err2 := updateIndex(layoutDir, ref, manifestDesc); err2 != nil {
			return errResp(err2)
		}
		return okResp("")
	}

	// Fetch config
	configRC, err := fetcher.Fetch(ctx, manifest.Config)
	if err != nil {
		return errResp(fmt.Errorf("fetch config: %w", err))
	}
	configData, readErr := io.ReadAll(configRC)
	configRC.Close()
	if readErr != nil {
		return errResp(fmt.Errorf("read config: %w", readErr))
	}
	if _, err := writeBlob(layoutDir, manifest.Config.MediaType, configData); err != nil {
		return errResp(fmt.Errorf("write config blob: %w", err))
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
		return errResp(err)
	}
	prog.Wait()

	if err := updateIndex(layoutDir, ref, manifestDesc); err != nil {
		return errResp(err)
	}
	return okResp("")
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

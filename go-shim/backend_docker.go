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

// dockerCredentials looks up stored credentials for host. containerd's
// dockerAuthorizer calls this with the *actual connection host* it's
// talking to (see AddResponses in containerd/v2/core/remotes/docker/
// authorizer.go: `host := last.Request.URL.Host`) — which for Docker Hub
// is "registry-1.docker.io" (containerd/dockerconfig.ConfigureHosts
// rewrites "docker.io" to that for the connection itself, but the
// Credentials callback still sees the post-rewrite host). `llmman login`/
// `docker login` store credentials under "docker.io" (or the legacy
// "index.docker.io"/"https://index.docker.io/v1/" keys real `docker
// login` also writes), never under "registry-1.docker.io" — so without
// this normalization, every push/pull that reaches an authenticated Hub
// endpoint (bearer or basic) silently runs anonymously instead (a
// credential-store miss isn't an error here, by design, so nothing ever
// surfaced this beyond a confusing downstream "insufficient_scope" or
// 401 on push).
func dockerCredentials(host string) (string, string, error) {
	cfg := dockercliconfig.LoadDefaultConfigFile(io.Discard)
	for _, lookup := range dockerHubCredentialKeys(host) {
		store := cfg.GetCredentialsStore(lookup)
		creds, err := store.Get(lookup)
		if err != nil {
			continue // not found under this key — try the next one
		}
		if creds.Username == "" && creds.Password == "" && creds.IdentityToken == "" {
			continue
		}
		if creds.IdentityToken != "" {
			return "", creds.IdentityToken, nil
		}
		return creds.Username, creds.Password, nil
	}
	return "", "", nil // not an error — just not found under any key
}

// dockerHubCredentialKeys returns every credential-store key that could
// plausibly hold Docker Hub credentials for a given connection host,
// broadest/most-canonical first. For any non-Hub host this is just the
// host itself, unchanged.
func dockerHubCredentialKeys(host string) []string {
	switch host {
	case "registry-1.docker.io", "index.docker.io", "docker.io", "https://index.docker.io/v1/":
		return []string{"docker.io", "index.docker.io", "https://index.docker.io/v1/", "registry-1.docker.io"}
	default:
		return []string{host}
	}
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

// pushLazy is the one place that actually talks to pusher.Push. It checks
// whether the destination already has desc *before* calling open — open
// returns the content to upload plus a cleanup func (call it even on a nil
// reader/error, may be nil itself) — and open is only ever invoked once
// that check has confirmed an upload is actually needed. Returns whether
// the blob already existed, so callers can print their own "already
// present" line instead of ever creating a progress bar for it.
//
// Checking existence first, unconditionally, matters for two different
// reasons depending on the caller:
//   - It avoids ever opening a (potentially multi-gigabyte) source reader
//     for content that's just going to be thrown away unread — mattering
//     beyond bandwidth, since leaving a large HTTP response body unread
//     and then closing it can itself take as long as reading it would
//     have (the transport may drain it to keep the connection reusable),
//     which otherwise looks exactly like an unexplained hang.
//   - It means a progress bar is only ever created for a blob that's
//     actually going to be incremented. mpb's (*Bar).SetTotal is
//     documented as a no-op for any bar constructed with a definite
//     (>0) total — which every bar here is, since every desc.Size is
//     already known up front — so there is no supported way to
//     retroactively mark such a bar "already done" after creating it.
//     Doing so anyway (an earlier version of this code did) silently
//     leaves that bar incomplete forever, and mpb's pool.Wait() blocks
//     forever waiting for every bar it knows about to finish — hanging
//     the whole transfer with no error and no output to explain why.
func pushLazy(
	ctx context.Context,
	pusher remotes.Pusher,
	desc ocispec.Descriptor,
	open func() (r io.Reader, bar *mpb.Bar, cleanup func(), err error),
) (alreadyExists bool, err error) {
	cw, err := pusher.Push(ctx, desc)
	if err != nil {
		if errdefs.IsAlreadyExists(err) {
			return true, nil
		}
		return false, describeErr(err)
	}
	defer cw.Close()

	r, bar, cleanup, err := open()
	if cleanup != nil {
		defer cleanup()
	}
	if err != nil {
		return false, err
	}
	if copyErr := describeErr(content.Copy(ctx, cw, r, desc.Size, desc.Digest)); copyErr != nil {
		// A real failure partway through: the bar (if any) was already
		// incremented some amount short of its total, and never will be
		// any further — abort it explicitly so it doesn't likewise leave
		// pool.Wait() hanging on a bar that's now never going anywhere.
		if bar != nil {
			bar.Abort(false)
		}
		return false, copyErr
	}
	return false, nil
}

// withBar wraps r in newBar's progress bar (if newBar is non-nil), and
// returns that bar (so pushLazy can abort it on a copy failure) plus a
// cleanup func that closes both the bar's proxy reader and r itself (if r
// is an io.Closer) — for use as pushLazy's open callback.
func withBar(r io.Reader, newBar func() *mpb.Bar) (io.Reader, *mpb.Bar, func()) {
	var bar *mpb.Bar
	closers := []io.Closer{}
	if rc, ok := r.(io.Closer); ok {
		closers = append(closers, rc)
	}
	if newBar != nil {
		bar = newBar()
		if proxyRC := bar.ProxyReader(r); proxyRC != nil {
			r = proxyRC
			closers = append(closers, proxyRC)
		}
	}
	return r, bar, func() {
		for _, c := range closers {
			c.Close()
		}
	}
}

// pushBlob pushes a single blob from the OCI layout to the registry
// pusher, reporting progress via newBar — called, and its resulting bar
// wrapped around the read, only if the blob isn't already at the
// destination (see pushLazy). Pass nil for no progress reporting.
func pushBlob(ctx context.Context, pusher remotes.Pusher, provider *ociProvider, desc ocispec.Descriptor, newBar func() *mpb.Bar) (alreadyExists bool, err error) {
	return pushLazy(ctx, pusher, desc, func() (io.Reader, *mpb.Bar, func(), error) {
		ra, err := provider.ReaderAt(ctx, desc)
		if err != nil {
			return nil, nil, nil, err
		}
		r, bar, cleanup := withBar(io.NewSectionReader(ra, 0, ra.Size()), newBar)
		return r, bar, func() { cleanup(); ra.Close() }, nil
	})
}

// pushBytes pushes an in-memory blob (a manifest or a small config/metadata
// file) directly to the registry pusher — no local file involved.
func pushBytes(ctx context.Context, pusher remotes.Pusher, desc ocispec.Descriptor, data []byte) error {
	_, err := pushLazy(ctx, pusher, desc, func() (io.Reader, *mpb.Bar, func(), error) {
		return bytes.NewReader(data), nil, nil, nil
	})
	return err
}

// pushStreamLazy pushes a blob whose digest and size are already known
// (see hfHeadMetadata) from a source opened lazily by openSource — called,
// and its resulting reader wrapped in a progress bar via newBar, only if
// the blob isn't already at the destination (see pushLazy). This is what
// makes `llmman transfer` behave like `skopeo copy` for large HuggingFace
// files: bytes flow source → destination without ever landing on disk (or
// getting downloaded at all, if the destination turns out to already have
// them) in between. Pass a nil newBar for no progress reporting.
func pushStreamLazy(ctx context.Context, pusher remotes.Pusher, desc ocispec.Descriptor, newBar func() *mpb.Bar, openSource func() (io.ReadCloser, error)) (alreadyExists bool, err error) {
	return pushLazy(ctx, pusher, desc, func() (io.Reader, *mpb.Bar, func(), error) {
		rc, err := openSource()
		if err != nil {
			return nil, nil, nil, err
		}
		r, bar, cleanup := withBar(rc, newBar)
		return r, bar, cleanup, nil
	})
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
	// normalizeTag: a tagless ref pushes the manifest addressable only by
	// digest, with no tag ever created — silently, since containerd has
	// no opinion on what a missing tag should default to. See
	// transfer_docker.go's dockerTransfer for the same fix applied there.
	pusher, err := resolver.Pusher(ctx, normalizeTag(ref))
	if err != nil {
		return fmt.Errorf("create pusher: %w", err)
	}
	provider := &ociProvider{dir: layoutDir}

	// "Copying blob/config <digest>" progress bars, matching skopeo's own
	// copy.Image output exactly (see copy/progress_bars.go upstream).
	prog := mpb.New(mpb.WithWidth(40), mpb.WithOutput(os.Stderr), mpb.WithRefreshRate(180*time.Millisecond))
	pushWithBar := func(desc ocispec.Descriptor, kind string) error {
		short := shortDigest(desc.Digest)
		newBar := func() *mpb.Bar {
			return addLayerBar(prog, "Copying "+kind+" "+short, "Copied  "+kind+" "+short, desc.Size)
		}
		alreadyExists, err := pushBlob(ctx, pusher, provider, desc, newBar)
		if err != nil {
			return err
		}
		if alreadyExists {
			fmt.Fprintf(os.Stderr, "Copied  %s %s (already present)\n", kind, short)
		}
		return nil
	}

	// Push layers
	for _, layer := range manifest.Layers {
		if err := pushWithBar(layer, "blob"); err != nil {
			prog.Wait()
			return fmt.Errorf("push layer %s: %w", layer.Digest, err)
		}
	}
	// Push config
	if err := pushWithBar(manifest.Config, "config"); err != nil {
		prog.Wait()
		return fmt.Errorf("push config: %w", err)
	}
	prog.Wait()

	// Push manifest — no progress bar (a few hundred bytes of JSON),
	// mirroring skopeo's own plain "Writing manifest to image
	// destination" message instead of a bar for this step.
	if _, err := pushBlob(ctx, pusher, provider, manifestDesc, nil); err != nil {
		return fmt.Errorf("push manifest: %w", err)
	}
	fmt.Fprintln(os.Stderr, "Writing manifest to image destination")
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

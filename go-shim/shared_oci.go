// shared_oci.go – OCI layout helpers used by both the docker and podman backends.
// No build tag: compiled for all configurations.

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	digest "github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/vbauerster/mpb/v8"
	"github.com/vbauerster/mpb/v8/decor"
)

// tagFromRef extracts the tag portion of a registry reference.
//
//	"registry.example.com/repo:tag" → "tag"
//	"registry.example.com/repo"     → "latest"
func tagFromRef(ref string) string {
	if i := strings.LastIndex(ref, ":"); i > strings.LastIndex(ref, "/") {
		return ref[i+1:]
	}
	return "latest"
}

// blobPath returns the path for a blob in an OCI image layout directory.
func blobPath(layoutDir string, dgst digest.Digest) string {
	return filepath.Join(layoutDir, "blobs", dgst.Algorithm().String(), dgst.Hex())
}

// readBlob reads a blob from an OCI layout directory.
func readBlob(layoutDir string, dgst digest.Digest) ([]byte, error) {
	return os.ReadFile(blobPath(layoutDir, dgst))
}

// writeBlob atomically writes data to the OCI layout blobs directory.
func writeBlob(layoutDir string, mediaType string, data []byte) (ocispec.Descriptor, error) {
	dgst := digest.FromBytes(data)
	dir := filepath.Join(layoutDir, "blobs", dgst.Algorithm().String())
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return ocispec.Descriptor{}, err
	}
	dest := filepath.Join(dir, dgst.Hex())
	if fi, err := os.Stat(dest); err == nil && fi.Size() == int64(len(data)) {
		return ocispec.Descriptor{MediaType: mediaType, Digest: dgst, Size: int64(len(data))}, nil
	}
	tmp := dest + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return ocispec.Descriptor{}, err
	}
	if err := os.Rename(tmp, dest); err != nil {
		return ocispec.Descriptor{}, err
	}
	return ocispec.Descriptor{MediaType: mediaType, Digest: dgst, Size: int64(len(data))}, nil
}

// openForResume opens path for appending if a previously-partial download of
// resumeFrom bytes already exists there, re-hashing those bytes into the
// returned digester so the digest computed over subsequent writes still
// spans the whole file; otherwise (no existing partial, or re-hashing it
// fails) it creates path fresh with a zeroed digester. The returned offset is
// resumeFrom on a successful resume, or 0 if resume wasn't possible. Shared
// by writeBlobStream (OCI-to-OCI transfer) and downloadAttempt (HuggingFace
// pull), which both append to a deterministic ".part" file across retries.
func openForResume(path string, resumeFrom int64) (f *os.File, digester digest.Digester, offset int64, err error) {
	digester = digest.Canonical.Digester()

	if resumeFrom > 0 {
		if pf, openErr := os.Open(path); openErr == nil {
			_, hashErr := io.Copy(digester.Hash(), pf)
			pf.Close()
			if hashErr == nil {
				if af, appendErr := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644); appendErr == nil {
					f = af
					offset = resumeFrom
				}
			}
		}
		if f == nil {
			digester = digest.Canonical.Digester()
		}
	}
	if f == nil {
		if f, err = os.Create(path); err != nil {
			return nil, nil, 0, err
		}
	}
	return f, digester, offset, nil
}

// writeBlobStream writes a large stream to the OCI layout blobs directory with
// resume support via a deterministic .part file.
func writeBlobStream(layoutDir, mediaType string, r io.Reader, size int64, dgst digest.Digest, partOffset int64) (ocispec.Descriptor, error) {
	dir := filepath.Join(layoutDir, "blobs", dgst.Algorithm().String())
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return ocispec.Descriptor{}, err
	}
	dest := filepath.Join(dir, dgst.Hex())
	if fi, err := os.Stat(dest); err == nil && (size <= 0 || fi.Size() == size) {
		return ocispec.Descriptor{MediaType: mediaType, Digest: dgst, Size: fi.Size()}, nil
	}
	tmp := dest + ".part"

	f, digester, startOffset, err := openForResume(tmp, partOffset)
	if err != nil {
		return ocispec.Descriptor{}, err
	}

	written, err := io.Copy(io.MultiWriter(f, digester.Hash()), r)
	f.Close()
	if err != nil {
		os.Remove(tmp)
		return ocispec.Descriptor{}, err
	}
	total := startOffset + written
	if size > 0 && total != size {
		os.Remove(tmp)
		return ocispec.Descriptor{}, fmt.Errorf("size mismatch: expected %d got %d", size, total)
	}
	if got := digester.Digest(); got != dgst {
		os.Remove(tmp)
		return ocispec.Descriptor{}, fmt.Errorf("digest mismatch: expected %s got %s", dgst, got)
	}
	if err := os.Rename(tmp, dest); err != nil {
		os.Remove(tmp)
		return ocispec.Descriptor{}, err
	}
	return ocispec.Descriptor{MediaType: mediaType, Digest: dgst, Size: total}, nil
}

// blobExists reports whether a blob is already fully stored in the layout.
func blobExists(layoutDir string, desc ocispec.Descriptor) bool {
	fi, err := os.Stat(blobPath(layoutDir, desc.Digest))
	return err == nil && fi.Size() == desc.Size
}

// readIndex reads index.json from an OCI layout directory.
func readIndex(layoutDir string) (ocispec.Index, error) {
	data, err := os.ReadFile(filepath.Join(layoutDir, "index.json"))
	if err != nil {
		return ocispec.Index{}, err
	}
	var idx ocispec.Index
	return idx, json.Unmarshal(data, &idx)
}

// writeIndex writes index.json to an OCI layout directory.
func writeIndex(layoutDir string, idx ocispec.Index) error {
	data, err := json.MarshalIndent(idx, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(layoutDir, "index.json"), data, 0o644)
}

// ensureLayout initialises the OCI layout marker files if not present.
func ensureLayout(layoutDir string) error {
	if err := os.MkdirAll(layoutDir, 0o755); err != nil {
		return err
	}
	markerPath := filepath.Join(layoutDir, "oci-layout")
	if _, err := os.Stat(markerPath); os.IsNotExist(err) {
		if err := os.WriteFile(markerPath, []byte(`{"imageLayoutVersion":"1.0.0"}`), 0o644); err != nil {
			return err
		}
	}
	indexPath := filepath.Join(layoutDir, "index.json")
	if _, err := os.Stat(indexPath); os.IsNotExist(err) {
		idx := ocispec.Index{
			Versioned: specs.Versioned{SchemaVersion: 2},
			MediaType: ocispec.MediaTypeImageIndex,
		}
		return writeIndex(layoutDir, idx)
	}
	return nil
}

// findManifestDesc looks up the manifest descriptor for a ref name in the index.
func findManifestDesc(idx ocispec.Index, refName string) (ocispec.Descriptor, error) {
	for _, m := range idx.Manifests {
		if m.Annotations != nil && m.Annotations[ocispec.AnnotationRefName] == refName {
			return m, nil
		}
	}
	if len(idx.Manifests) == 1 {
		return idx.Manifests[0], nil
	}
	return ocispec.Descriptor{}, fmt.Errorf("no manifest found for %q", refName)
}

// ---------------------------------------------------------------------------
// Retry/stall-detection primitives shared by every download/transfer path:
// hf.go's downloadHFBlob (pull, HF → local disk, resumable via a .part
// file + Range request) and transfer_docker.go's streamHFFileToRegistry /
// dockerTransferOCI (transfer, streamed straight into a registry push,
// NOT resumable — see transfer_docker.go's own comment on why — so these
// retry a failed blob from scratch instead of resuming it).
// ---------------------------------------------------------------------------

const (
	dlMaxAttempts  = 3
	dlRetryBase    = 2 * time.Second // doubles each retry: 2s, 4s
	dlStallTimeout = 60 * time.Second
)

// stallReader cancels the context if no bytes arrive within timeout.
// Mirrors llama.cpp's implicit stall detection via cpp-httplib timeouts.
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
		sr.timer.Reset(dlStallTimeout) // bytes arrived, reset stall clock
	}
	return n, err
}

func (sr *stallReader) stop() { sr.timer.Stop() }

// stallReadCloser pairs a stallReader with the underlying response body's
// Close, so callers can pass it around as a plain io.ReadCloser and have
// the stall timer stopped automatically whenever the body is closed
// (success, error, or early abandonment all go through Close).
type stallReadCloser struct {
	*stallReader
	body io.Closer
}

func newStallReadCloser(rc io.ReadCloser, timeout time.Duration, cancel context.CancelFunc) *stallReadCloser {
	return &stallReadCloser{stallReader: newStallReader(rc, timeout, cancel), body: rc}
}

func (s *stallReadCloser) Close() error {
	s.stop()
	return s.body.Close()
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

// retryStream calls attempt up to dlMaxAttempts times with exponential
// backoff (2s, 4s, ...) between tries, stopping immediately (no further
// retries) once isPermanent reports the most recent error isn't worth
// retrying (e.g. a 404 — see isHTTP4xx). Every attempt is expected to
// restart its work entirely from scratch: unlike downloadHFBlob's local
// .part-file resume, there's no partial state to pick up from here (see
// the callers' own comments for why) — this only saves the operator from
// having to notice a transient failure and manually re-run the whole
// command, it doesn't avoid re-sending bytes a failed attempt already
// sent.
func retryStream(ctx context.Context, label string, isPermanent func(error) bool, attempt func() error) error {
	var lastErr error
	for i := 0; i < dlMaxAttempts; i++ {
		if i > 0 {
			delay := dlRetryBase * time.Duration(1<<uint(i-1)) // 2s, 4s
			fmt.Fprintf(os.Stderr, "\n[llmman] retrying %s (attempt %d/%d, wait %v)\n", label, i+1, dlMaxAttempts, delay)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(delay):
			}
		}
		err := attempt()
		if err == nil {
			return nil
		}
		lastErr = err
		if isPermanent != nil && isPermanent(err) {
			break
		}
		fmt.Fprintf(os.Stderr, "[llmman] %s error: %v\n", label, err)
	}
	return fmt.Errorf("%s failed after %d attempts: %w", label, dlMaxAttempts, lastErr)
}

// proxyOrNop wraps r in bar's progress-tracking proxy reader, falling back to
// a plain no-op-Close wrapper around r when the bar declines to proxy (e.g.
// a zero-total spinner bar). Every downloader in this package that reports
// progress via an mpb.Bar needs this same fallback.
func proxyOrNop(bar *mpb.Bar, r io.Reader) io.ReadCloser {
	if p := bar.ProxyReader(r); p != nil {
		return p
	}
	return io.NopCloser(r)
}

// newProgressPool creates an mpb.Progress bar pool with the output/refresh
// settings shared by every download/transfer path in llmman; only the bar
// width varies by call site (80 for pull, 40 for transfer).
func newProgressPool(width int) *mpb.Progress {
	return mpb.New(mpb.WithWidth(width), mpb.WithOutput(os.Stderr), mpb.WithRefreshRate(180*time.Millisecond))
}

// addLayerBar adds a progress bar into an existing mpb.Progress.
func addLayerBar(p *mpb.Progress, prefix, onComplete string, size int64) *mpb.Bar {
	bar := p.AddBar(size,
		mpb.BarFillerClearOnComplete(),
		mpb.PrependDecorators(
			decor.OnComplete(decor.Name(prefix), onComplete),
		),
		mpb.AppendDecorators(
			decor.OnComplete(decor.CountersKibiByte("% .1f / % .1f"), ""),
			decor.OnComplete(decor.Name("  "), ""),
			decor.OnComplete(decor.AverageSpeed(decor.SizeB1024(0), "% .1f"), ""),
		),
	)
	if size <= 0 {
		bar.SetTotal(0, true)
	}
	return bar
}

// updateIndex adds or replaces the manifest entry in index.json with an
// exclusive advisory lock to prevent concurrent corruption.
func updateIndex(layoutDir, ref string, manifestDesc ocispec.Descriptor) error {
	lock, err := lockIndex(layoutDir)
	if err != nil {
		return err
	}
	defer lock.release()

	idx, err := readIndex(layoutDir)
	if err != nil {
		idx = ocispec.Index{
			Versioned: specs.Versioned{SchemaVersion: 2},
			MediaType: ocispec.MediaTypeImageIndex,
		}
	}
	if manifestDesc.Annotations == nil {
		manifestDesc.Annotations = map[string]string{}
	}
	manifestDesc.Annotations[ocispec.AnnotationRefName] = ref

	replaced := false
	for i, m := range idx.Manifests {
		if m.Annotations != nil && m.Annotations[ocispec.AnnotationRefName] == ref {
			idx.Manifests[i] = manifestDesc
			replaced = true
			break
		}
	}
	if !replaced {
		idx.Manifests = append(idx.Manifests, manifestDesc)
	}
	return writeIndex(layoutDir, idx)
}

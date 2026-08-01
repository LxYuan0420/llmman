// shared_oci.go – OCI layout helpers used by both the docker and podman backends.
// No build tag: compiled for all configurations.

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

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
	digester := digest.Canonical.Digester()
	var f *os.File
	startOffset := int64(0)

	if partOffset > 0 {
		if pf, err := os.Open(tmp); err == nil {
			_, hashErr := io.Copy(digester.Hash(), pf)
			pf.Close()
			if hashErr == nil {
				f, err = os.OpenFile(tmp, os.O_APPEND|os.O_WRONLY, 0o644)
				if err == nil {
					startOffset = partOffset
				}
			}
		}
		if f == nil {
			digester = digest.Canonical.Digester()
		}
	}
	if f == nil {
		var err error
		if f, err = os.Create(tmp); err != nil {
			return ocispec.Descriptor{}, err
		}
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

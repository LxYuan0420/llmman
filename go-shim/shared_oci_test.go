package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"

	digest "github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// TestWriteIndexIsAtomic is a regression test: writeIndex used to
// os.WriteFile straight over index.json, so a reader racing a write (or a
// process killed mid-write) could observe a truncated/partial file. It now
// writes to a temp file and renames it into place — this checks that
// behaviour directly (content round-trips, no leftover .tmp) rather than
// relying on the previous version's absence of it.
func TestWriteIndexIsAtomic(t *testing.T) {
	dir := t.TempDir()
	idx := ocispec.Index{
		Versioned: specs.Versioned{SchemaVersion: 2},
		MediaType: ocispec.MediaTypeImageIndex,
		Manifests: []ocispec.Descriptor{
			{MediaType: ocispec.MediaTypeImageManifest, Digest: digest.FromString("one"), Size: 1},
		},
	}
	if err := writeIndex(dir, idx); err != nil {
		t.Fatalf("writeIndex: %v", err)
	}

	if _, err := os.Stat(filepath.Join(dir, "index.json.tmp")); !os.IsNotExist(err) {
		t.Fatalf("expected index.json.tmp to be gone after a successful write, stat error: %v", err)
	}

	got, err := readIndex(dir)
	if err != nil {
		t.Fatalf("readIndex: %v", err)
	}
	if len(got.Manifests) != 1 || got.Manifests[0].Digest != idx.Manifests[0].Digest {
		t.Fatalf("readIndex returned %+v, want the single manifest just written", got)
	}
}

// TestUpdateIndexIsSafeForConcurrentWriters is a regression test for the
// race updateIndex's lockIndex call exists to prevent: without it, two
// concurrent read-modify-write cycles on index.json can each read the same
// starting state and one's write clobbers the other's, silently losing a
// manifest entry. Runs many concurrent updateIndex calls, each adding a
// distinct ref, and checks every one of them survived.
func TestUpdateIndexIsSafeForConcurrentWriters(t *testing.T) {
	dir := t.TempDir()
	const n = 25

	var wg sync.WaitGroup
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ref := fmt.Sprintf("ref-%d", i)
			desc := ocispec.Descriptor{
				MediaType: ocispec.MediaTypeImageManifest,
				Digest:    digest.FromString(ref),
				Size:      int64(i),
			}
			if err := updateIndex(dir, ref, desc); err != nil {
				errs <- fmt.Errorf("updateIndex(%s): %w", ref, err)
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}

	idx, err := readIndex(dir)
	if err != nil {
		t.Fatalf("readIndex: %v", err)
	}
	if len(idx.Manifests) != n {
		t.Fatalf("got %d manifests after %d concurrent updateIndex calls, want %d (lost a concurrent write)",
			len(idx.Manifests), n, n)
	}
	seen := make(map[string]bool, n)
	for _, m := range idx.Manifests {
		seen[m.Annotations[ocispec.AnnotationRefName]] = true
	}
	for i := 0; i < n; i++ {
		ref := fmt.Sprintf("ref-%d", i)
		if !seen[ref] {
			t.Errorf("manifest for %s missing from final index", ref)
		}
	}
}

package main

import (
	"encoding/json"
	"testing"

	modelspec "github.com/modelpack/model-spec/specs-go/v1"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

func TestSelectMMProjPrefersF16(t *testing.T) {
	files := []hfFile{
		{Path: "model-Q4_K_M.gguf", Type: "file"},
		{Path: "mmproj-BF16.gguf", Type: "file"},
		{Path: "mmproj-F16.gguf", Type: "file"},
		{Path: "mmproj-F32.gguf", Type: "file"},
	}
	f, ok := selectMMProj(files)
	if !ok {
		t.Fatal("expected a multimodal projector to be found")
	}
	if f.Path != "mmproj-F16.gguf" {
		t.Errorf("expected mmproj-F16.gguf to be preferred, got %q", f.Path)
	}
}

func TestSelectMMProjFallsBackWhenF16Absent(t *testing.T) {
	files := []hfFile{
		{Path: "model-Q4_K_M.gguf", Type: "file"},
		{Path: "mmproj-BF16.gguf", Type: "file"},
	}
	f, ok := selectMMProj(files)
	if !ok {
		t.Fatal("expected a multimodal projector to be found")
	}
	if f.Path != "mmproj-BF16.gguf" {
		t.Errorf("expected mmproj-BF16.gguf, got %q", f.Path)
	}
}

func TestSelectMMProjAbsent(t *testing.T) {
	files := []hfFile{
		{Path: "model-Q4_K_M.gguf", Type: "file"},
		{Path: "README.md", Type: "file"},
	}
	if _, ok := selectMMProj(files); ok {
		t.Error("expected no multimodal projector to be found")
	}
}

func TestSelectMMProjIgnoresDirectories(t *testing.T) {
	files := []hfFile{
		{Path: "mmproj-F16.gguf", Type: "directory"},
	}
	if _, ok := selectMMProj(files); ok {
		t.Error("a directory entry named like an mmproj file should not be selected")
	}
}

func TestSelectLicenseFile(t *testing.T) {
	cases := []struct {
		name  string
		files []hfFile
		want  string
		found bool
	}{
		{
			name:  "plain LICENSE",
			files: []hfFile{{Path: "LICENSE", Type: "file"}, {Path: "README.md", Type: "file"}},
			want:  "LICENSE",
			found: true,
		},
		{
			name:  "case insensitive",
			files: []hfFile{{Path: "license", Type: "file"}},
			want:  "license",
			found: true,
		},
		{
			name:  "LICENSE.md",
			files: []hfFile{{Path: "LICENSE.md", Type: "file"}},
			want:  "LICENSE.md",
			found: true,
		},
		{
			name:  "none present",
			files: []hfFile{{Path: "README.md", Type: "file"}},
			found: false,
		},
		{
			name:  "directory named LICENSE is not a file",
			files: []hfFile{{Path: "LICENSE", Type: "directory"}},
			found: false,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f, ok := selectLicenseFile(c.files)
			if ok != c.found {
				t.Fatalf("ok = %v, want %v", ok, c.found)
			}
			if ok && f.Path != c.want {
				t.Errorf("got %q, want %q", f.Path, c.want)
			}
		})
	}
}

func TestNormalizeSPDXLicense(t *testing.T) {
	cases := map[string]string{
		"apache-2.0": "Apache-2.0",
		"MIT":        "MIT",
		"mit":        "MIT",
		"other":      "",
		"unknown":    "",
		"":           "",
		"bespoke-license-slug": "bespoke-license-slug", // unknown slug passes through unchanged
	}
	for in, want := range cases {
		if got := normalizeSPDXLicense(in); got != want {
			t.Errorf("normalizeSPDXLicense(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestHFModelInfoLicense(t *testing.T) {
	t.Run("from cardData", func(t *testing.T) {
		info := hfModelInfo{}
		info.CardData.License = "apache-2.0"
		got, ok := info.license()
		if !ok || got != "Apache-2.0" {
			t.Errorf("got (%q, %v), want (%q, true)", got, ok, "Apache-2.0")
		}
	})

	t.Run("falls back to license tag", func(t *testing.T) {
		info := hfModelInfo{Tags: []string{"gguf", "license:mit"}}
		got, ok := info.license()
		if !ok || got != "MIT" {
			t.Errorf("got (%q, %v), want (%q, true)", got, ok, "MIT")
		}
	})

	t.Run("cardData takes precedence over tags", func(t *testing.T) {
		info := hfModelInfo{Tags: []string{"license:mit"}}
		info.CardData.License = "apache-2.0"
		got, ok := info.license()
		if !ok || got != "Apache-2.0" {
			t.Errorf("got (%q, %v), want (%q, true)", got, ok, "Apache-2.0")
		}
	})

	t.Run("no usable license", func(t *testing.T) {
		info := hfModelInfo{Tags: []string{"gguf"}}
		if _, ok := info.license(); ok {
			t.Error("expected no usable license")
		}
	})

	t.Run("other/unknown slugs are not usable", func(t *testing.T) {
		info := hfModelInfo{}
		info.CardData.License = "other"
		if _, ok := info.license(); ok {
			t.Error(`"other" should not be reported as a usable license`)
		}
	})
}

func TestHFModelInfoCommit(t *testing.T) {
	if got := (hfModelInfo{SHA: "abc123"}).commit(); got != "abc123" {
		t.Errorf("commit() = %q, want %q", got, "abc123")
	}
	if got := (hfModelInfo{}).commit(); got != "main" {
		t.Errorf("commit() with no SHA = %q, want %q", got, "main")
	}
}

// TestBuildCNCFManifestPopulatesMetadata exercises buildCNCFManifest
// end-to-end against a local OCI layout (layoutBlobSink), with no
// network involved, to verify the actual JSON shape written for
// descriptor.licenses and config.capabilities matches what model-spec's
// schema expects — see
// https://github.com/modelpack/model-spec/blob/main/docs/config.md.
func TestBuildCNCFManifestPopulatesMetadata(t *testing.T) {
	dir := t.TempDir()
	if err := ensureLayout(dir); err != nil {
		t.Fatalf("ensureLayout: %v", err)
	}

	weightDesc, err := writeBlob(dir, modelspec.MediaTypeModelWeightRaw, []byte("fake gguf weight"))
	if err != nil {
		t.Fatalf("writeBlob weight: %v", err)
	}
	weightDesc.Annotations = map[string]string{modelspec.AnnotationFilepath: "model-Q4_K_M.gguf"}

	mmprojDesc, err := writeBlob(dir, modelspec.MediaTypeModelWeightRaw, []byte("fake mmproj weight"))
	if err != nil {
		t.Fatalf("writeBlob mmproj: %v", err)
	}
	mmprojDesc.Annotations = map[string]string{modelspec.AnnotationFilepath: "mmproj-F16.gguf"}

	meta := modelMeta{
		Format:   "gguf",
		Licenses: []string{"Apache-2.0"},
		Vision:   true,
	}
	layers := []ocispec.Descriptor{weightDesc, mmprojDesc}
	manifestDesc, err := buildCNCFManifest(layoutBlobSink(dir), meta, "unsloth/example-GGUF", "", layers)
	if err != nil {
		t.Fatalf("buildCNCFManifest: %v", err)
	}

	// Read the manifest and config blobs straight back out of the layout
	// and verify the exact JSON shape a real consumer would see.
	manifestData, err := readBlob(dir, manifestDesc.Digest)
	if err != nil {
		t.Fatalf("readBlob manifest: %v", err)
	}
	var manifest ocispec.Manifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		t.Fatalf("unmarshal manifest: %v", err)
	}
	if manifest.ArtifactType != modelspec.ArtifactTypeModelManifest {
		t.Errorf("artifactType = %q, want %q", manifest.ArtifactType, modelspec.ArtifactTypeModelManifest)
	}
	if len(manifest.Layers) != 2 {
		t.Fatalf("expected 2 layers, got %d", len(manifest.Layers))
	}

	configData, err := readBlob(dir, manifest.Config.Digest)
	if err != nil {
		t.Fatalf("readBlob config: %v", err)
	}
	var model modelspec.Model
	if err := json.Unmarshal(configData, &model); err != nil {
		t.Fatalf("unmarshal config: %v", err)
	}
	if len(model.Descriptor.Licenses) != 1 || model.Descriptor.Licenses[0] != "Apache-2.0" {
		t.Errorf("descriptor.licenses = %v, want [Apache-2.0]", model.Descriptor.Licenses)
	}
	if model.Config.Format != "gguf" {
		t.Errorf("config.format = %q, want gguf", model.Config.Format)
	}
	if model.Config.Capabilities == nil {
		t.Fatal("expected config.capabilities to be set for a vision model")
	}
	wantIn := []modelspec.Modality{modelspec.TextModality, modelspec.ImageModality}
	if len(model.Config.Capabilities.InputTypes) != 2 ||
		model.Config.Capabilities.InputTypes[0] != wantIn[0] ||
		model.Config.Capabilities.InputTypes[1] != wantIn[1] {
		t.Errorf("capabilities.inputTypes = %v, want %v", model.Config.Capabilities.InputTypes, wantIn)
	}
	if len(model.ModelFS.DiffIDs) != 2 {
		t.Errorf("modelfs.diffIds has %d entries, want 2 (one per layer)", len(model.ModelFS.DiffIDs))
	}
}

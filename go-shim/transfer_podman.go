//go:build podman

// transfer_podman.go — `llmman transfer`'s containers/image-backed
// implementation.
//
// For an OCI registry source, this is a one-line reuse of exactly what
// skopeo itself does: hand `copy.Image` a docker://source and a
// docker://destination reference directly, with no local OCI layout in
// between at all. containers/image's copy.Image already streams each blob
// straight from source to destination (it reads the source manifest first,
// so every blob's digest/size is known up front, then opens a GetBlob
// reader and a PutBlob writer for each one — see copy/copy.go upstream);
// there's nothing llmman needs to add on top for that case.
//
// containers/image has no HuggingFace (or ms:///ngc:///s3:///gs:///local
// path) source transport, though, so those fall back to staging through a
// throwaway local OCI layout — pull, then push from it, mirroring what
// `llmman transfer` did before streaming support existed, and what
// transfer_docker.go's docker/containerd backend still does for those same
// source kinds (see its own doc comment for why implementing zero-disk
// streaming for each of them individually isn't worth it yet).
package main

import (
	"context"
	"fmt"
	"os"

	"go.podman.io/image/v5/copy"
	"go.podman.io/image/v5/transports/alltransports"
)

func podmanTransfer(ctx context.Context, source, destination string) error {
	// See transfer_docker.go's dockerTransfer for why a tagless
	// destination must default to :latest explicitly here.
	destination = normalizeTag(destination)
	kind, normalized := classifySource(ctx, source)
	if kind == sourceOCI {
		return podmanTransferOCI(ctx, normalized, destination)
	}
	return transferViaStaging(ctx, source, destination)
}

// podmanTransferOCI streams directly between two registries via
// containers/image's copy.Image — no local OCI layout involved.
func podmanTransferOCI(ctx context.Context, source, destination string) error {
	srcStr := "docker://" + source
	srcRef, err := alltransports.ParseImageName(srcStr)
	if err != nil {
		return fmt.Errorf("parse src ref %q: %w", srcStr, err)
	}
	dstStr := "docker://" + destination
	dstRef, err := alltransports.ParseImageName(dstStr)
	if err != nil {
		return fmt.Errorf("parse dst ref %q: %w", dstStr, err)
	}

	pctx, err := insecurePolicy()
	if err != nil {
		return fmt.Errorf("policy context: %w", err)
	}
	defer pctx.Destroy()

	_, err = copy.Image(ctx, pctx, dstRef, srcRef, &copy.Options{
		ReportWriter: os.Stderr,
	})
	if err != nil {
		return fmt.Errorf("copy image: %w", err)
	}
	return nil
}

// transferViaStaging is the fallback path for source kinds containers/image
// has no transport for (HuggingFace, ms://, ngc://, s3://, gs://, a local
// path): pull into a throwaway local OCI layout, then push from it.
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

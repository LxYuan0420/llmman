//go:build podman

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
	"time"

	"github.com/vbauerster/mpb/v8"
	"github.com/vbauerster/mpb/v8/decor"
	commonauth "go.podman.io/common/pkg/auth"
	"go.podman.io/image/v5/copy"
	"go.podman.io/image/v5/signature"
	"go.podman.io/image/v5/transports/alltransports"
	"go.podman.io/image/v5/types"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func insecurePolicy() (*signature.PolicyContext, error) {
	policy := &signature.Policy{
		Default: signature.PolicyRequirements{
			signature.NewPRInsecureAcceptAnything(),
		},
	}
	return signature.NewPolicyContext(policy)
}

// ---------------------------------------------------------------------------
// Exported CGO functions
// ---------------------------------------------------------------------------

// llmman_login stores credentials for a registry using the containers/common auth library.
//
//export llmman_login
func llmman_login(cServer, cUsername, cPassword *C.char) *C.char {
	server := C.GoString(cServer)
	username := C.GoString(cUsername)
	password := C.GoString(cPassword)

	sys := &types.SystemContext{}
	opts := &commonauth.LoginOptions{
		Username: username,
		Password: password,
	}
	if err := commonauth.Login(context.Background(), sys, opts, []string{server}); err != nil {
		return errResp(fmt.Errorf("login: %w", err))
	}
	return okResp("")
}

// llmman_logout removes credentials for a registry.
//
//export llmman_logout
func llmman_logout(cServer *C.char) *C.char {
	server := C.GoString(cServer)

	sys := &types.SystemContext{}
	opts := &commonauth.LogoutOptions{All: false}
	if err := commonauth.Logout(sys, opts, []string{server}); err != nil {
		return errResp(fmt.Errorf("logout: %w", err))
	}
	return okResp("")
}

// llmman_push pushes an image from a local OCI layout to a registry.
//
//export llmman_push
func llmman_push(cLayoutDir, cRef *C.char) *C.char {
	if err := pushToRegistry(context.Background(), C.GoString(cLayoutDir), C.GoString(cRef)); err != nil {
		return errResp(err)
	}
	return okResp("")
}

// pushToRegistry is llmman_push's implementation, factored out so
// llmman_transfer's staging-directory fallback (see transfer_podman.go)
// can reuse it without going through CGO.
func pushToRegistry(ctx context.Context, layoutDir, ref string) error {
	tag := tagFromRef(ref)

	// Source: OCI layout directory
	srcStr := fmt.Sprintf("oci:%s:%s", layoutDir, tag)
	srcRef, err := alltransports.ParseImageName(srcStr)
	if err != nil {
		return fmt.Errorf("parse src ref %q: %w", srcStr, err)
	}

	// Destination: Docker registry
	dstStr := "docker://" + ref
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
	if handled, err := dispatchPull(ctx, ref, layoutDir); handled {
		return err
	}

	// Normalize: append :latest if reference has no tag or digest
	if strings.LastIndex(ref, ":") <= strings.LastIndex(ref, "/") {
		ref = ref + ":latest"
	}

	// HuggingFace and similar hosts cannot be pulled via the OCI registry
	// protocol (their paths contain uppercase letters which containers/image
	// rejects).  Delegate to the shared HF pull path instead.
	host := strings.SplitN(ref, "/", 2)[0]
	if isKnownHFHost(host) || (!isKnownOCIHost(host) && !isOCIRegistry(ctx, &http.Client{Timeout: 5 * time.Second}, host)) {
		if err := ensureLayout(layoutDir); err != nil {
			return fmt.Errorf("init OCI layout: %w", err)
		}
		return pullHF(ctx, ref, layoutDir)
	}

	tag := tagFromRef(ref)

	// Source: Docker registry
	srcStr := "docker://" + ref
	srcRef, err := alltransports.ParseImageName(srcStr)
	if err != nil {
		return fmt.Errorf("parse src ref %q: %w", srcStr, err)
	}

	// Ensure the OCI layout directory exists
	if err := os.MkdirAll(layoutDir, 0o755); err != nil {
		return fmt.Errorf("create layout dir: %w", err)
	}

	// Destination: OCI layout directory
	dstStr := fmt.Sprintf("oci:%s:%s", layoutDir, tag)
	dstRef, err := alltransports.ParseImageName(dstStr)
	if err != nil {
		return fmt.Errorf("parse dst ref %q: %w", dstStr, err)
	}

	pctx, err := insecurePolicy()
	if err != nil {
		return fmt.Errorf("policy context: %w", err)
	}
	defer pctx.Destroy()

	prog := mpb.New(mpb.WithOutput(os.Stderr))
	ch := make(chan types.ProgressProperties)
	bars := make(map[string]*mpb.Bar)
	progDone := make(chan struct{})
	go func() {
		defer close(progDone)
		for p := range ch {
			key := p.Artifact.Digest.String()
			switch p.Event {
			case types.ProgressEventNewArtifact:
				total := p.Artifact.Size
				if total < 0 {
					total = 0
				}
				short := p.Artifact.Digest.Hex()
				if len(short) > 12 {
					short = short[:12]
				}
				bar := prog.AddBar(total,
					mpb.BarFillerClearOnComplete(),
					mpb.PrependDecorators(
						decor.OnComplete(decor.Name("Pulling  "+short), "Pulled   "+short),
					),
					mpb.AppendDecorators(
						decor.OnComplete(decor.CountersKibiByte("% .1f / % .1f"), ""),
						decor.OnComplete(decor.Name("  "), ""),
						decor.OnComplete(decor.AverageSpeed(decor.SizeB1024(0), "% .1f"), ""),
					),
				)
				if total == 0 {
					bar.SetTotal(0, true)
				}
				bars[key] = bar
			case types.ProgressEventRead:
				if bar, ok := bars[key]; ok {
					bar.IncrInt64(int64(p.OffsetUpdate))
				}
			case types.ProgressEventDone:
				if bar, ok := bars[key]; ok {
					// SetTotal with triggerComplete forces current=total regardless of
					// timing, then fires done() — the OnComplete decorators take over.
					bar.SetTotal(int64(p.Offset), true)
					delete(bars, key)
				}
			case types.ProgressEventSkipped:
				if bar, ok := bars[key]; ok {
					bar.Abort(true)
					delete(bars, key)
					fmt.Fprintf(prog, "Cached   %s\n", p.Artifact.Digest.Hex()[:12])
				}
			}
		}
	}()

	_, err = copy.Image(ctx, pctx, dstRef, srcRef, &copy.Options{
		Progress:             ch,
		ProgressInterval:     200 * time.Millisecond,
		MaxParallelDownloads: 6,
	})
	close(ch)
	<-progDone
	prog.Wait()

	if err != nil {
		return fmt.Errorf("copy image: %w", err)
	}
	return nil
}

// llmman_inspect fetches and returns the raw manifest JSON for a remote reference.
//
//export llmman_inspect
func llmman_inspect(cRef *C.char) *C.char {
	ref := C.GoString(cRef)

	srcStr := "docker://" + ref
	srcRef, err := alltransports.ParseImageName(srcStr)
	if err != nil {
		return errResp(fmt.Errorf("parse ref %q: %w", srcStr, err))
	}

	sys := &types.SystemContext{}
	img, err := srcRef.NewImage(context.Background(), sys)
	if err != nil {
		return errResp(fmt.Errorf("open image: %w", err))
	}
	defer img.Close()

	manifestData, _, err := img.Manifest(context.Background())
	if err != nil {
		return errResp(fmt.Errorf("fetch manifest: %w", err))
	}

	var buf bytes.Buffer
	if err := json.Indent(&buf, manifestData, "", "  "); err != nil {
		return okResp(string(manifestData))
	}
	return okResp(buf.String())
}

// llmman_transfer copies an image directly from source to destination —
// llmman's equivalent of `skopeo copy` — without ever writing it to the
// persistent local store. See transfer_podman.go for what this picks
// between (a direct docker://→docker:// copy.Image, which — like skopeo
// itself — streams every blob straight through since containers/image
// already knows each one's digest from the source manifest; or a
// staging-directory fallback for HuggingFace and other non-OCI sources,
// which containers/image has no source transport for).
//
//export llmman_transfer
func llmman_transfer(cSource, cDestination *C.char) *C.char {
	if err := podmanTransfer(context.Background(), C.GoString(cSource), C.GoString(cDestination)); err != nil {
		return errResp(err)
	}
	return okResp("")
}

// Ensure io is used (imported via shared helpers but referenced here for the build)
var _ = io.Discard

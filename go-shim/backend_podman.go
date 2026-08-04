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
	"os"
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
	progressReset("retrieving manifest")
	if _, err := pushToRegistry(context.Background(), C.GoString(cLayoutDir), C.GoString(cRef)); err != nil {
		return errResp(err)
	}
	return okResp("")
}

// pushToRegistry is llmman_push's implementation, factored out so
// llmman_transfer's staging-directory fallback (see transfer_podman.go)
// can reuse it without going through CGO.
func pushToRegistry(ctx context.Context, layoutDir, ref string) (changed bool, err error) {
	tag := tagFromRef(ref)

	// Source: OCI layout directory
	srcStr := fmt.Sprintf("oci:%s:%s", layoutDir, tag)
	srcRef, err := alltransports.ParseImageName(srcStr)
	if err != nil {
		return false, fmt.Errorf("parse src ref %q: %w", srcStr, err)
	}

	// Destination: Docker registry. containers/image's docker transport
	// already defaults a tagless ref to :latest internally, but
	// normalizing here too keeps this consistent with the docker/
	// containerd backend (whose resolver has no such default — see
	// backend_docker.go's pushToRegistry) and the local index.json tag
	// lookup above, which effectively assumes the same thing.
	dstStr := "docker://" + normalizeTag(ref)
	dstRef, err := alltransports.ParseImageName(dstStr)
	if err != nil {
		return false, fmt.Errorf("parse dst ref %q: %w", dstStr, err)
	}

	pctx, err := insecurePolicy()
	if err != nil {
		return false, fmt.Errorf("policy context: %w", err)
	}
	defer pctx.Destroy()

	progressSetStatus("pushing")
	if err := copyImageWithProgress(ctx, pctx, dstRef, srcRef, "Pushing", "Pushed", &copy.Options{}, &changed); err != nil {
		return false, fmt.Errorf("copy image: %w", err)
	}
	return changed, nil
}

// llmman_pull pulls an image from a registry into a local OCI layout directory.
//
//export llmman_pull
func llmman_pull(cRef, cLayoutDir *C.char) *C.char {
	progressReset("pulling manifest")
	if err := pullToLayout(context.Background(), C.GoString(cRef), C.GoString(cLayoutDir)); err != nil {
		return errResp(err)
	}
	return okResp("")
}

// pullToLayout is llmman_pull's implementation, factored out so
// llmman_transfer's staging-directory fallback can reuse it.
func pullToLayout(ctx context.Context, ref, layoutDir string) error {
	ref, isOCI, handled, err := classifyPullRef(ctx, ref, layoutDir)
	if handled {
		return err
	}
	// HuggingFace and similar hosts cannot be pulled via the OCI registry
	// protocol (their paths contain uppercase letters which containers/image
	// rejects).  Delegate to the shared HF pull path instead.
	if !isOCI {
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

	progressSetStatus("pulling")
	if err := copyImageWithProgress(ctx, pctx, dstRef, srcRef, "Pulling", "Pulled", &copy.Options{
		MaxParallelDownloads: 6,
	}, nil); err != nil {
		return fmt.Errorf("copy image: %w", err)
	}
	return nil
}

// copyImageWithProgress runs copy.Image with an mpb bar per artifact (for
// direct/foreground FFI callers, e.g. `llmman transfer`'s podman backend —
// though transfer_podman.go's own copy.Image calls don't currently go
// through this, only pull/push do) and folds the same byte counts into the
// shared progressState snapshot (see progress_state.go) that lets
// cmd::serve poll them out of the daemon process — two consumers of the
// same underlying containers/image progress channel. present/pastTense
// label each artifact's bar (e.g. "Pulling"/"Pulled", "Pushing"/"Pushed").
//
// If changed is non-nil, it's set to true whenever at least one artifact
// actually completes a copy (types.ProgressEventDone) rather than turning
// out to already exist at the destination (types.ProgressEventSkipped,
// which never leads to a Done for that same artifact) — letting a caller
// like pushToRegistry/podmanTransferOCI tell whether anything was really
// pushed, e.g. to report "already up to date" for a no-op re-transfer.
func copyImageWithProgress(ctx context.Context, pctx *signature.PolicyContext, dst, src types.ImageReference, present, pastTense string, opts *copy.Options, changed *bool) error {
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
				progressAddTotal(total)
				short := p.Artifact.Digest.Hex()
				if len(short) > 12 {
					short = short[:12]
				}
				bar := prog.AddBar(total,
					mpb.BarFillerClearOnComplete(),
					mpb.PrependDecorators(
						decor.OnComplete(decor.Name(present+"  "+short), pastTense+"   "+short),
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
				progressAddCompleted(int64(p.OffsetUpdate))
				if bar, ok := bars[key]; ok {
					bar.IncrInt64(int64(p.OffsetUpdate))
				}
			case types.ProgressEventDone:
				if changed != nil {
					*changed = true
				}
				if bar, ok := bars[key]; ok {
					// SetTotal with triggerComplete forces current=total regardless of
					// timing, then fires done() — the OnComplete decorators take over.
					bar.SetTotal(int64(p.Offset), true)
					delete(bars, key)
				}
			case types.ProgressEventSkipped:
				// This artifact turned out to already exist at the
				// destination — no bytes will ever flow for it via
				// ProgressEventRead, so undo the provisional total
				// ProgressEventNewArtifact already added above.
				progressAddTotal(-p.Artifact.Size)
				if bar, ok := bars[key]; ok {
					bar.Abort(true)
					delete(bars, key)
					fmt.Fprintf(prog, "Cached   %s\n", p.Artifact.Digest.Hex()[:12])
				}
			}
		}
	}()

	opts.Progress = ch
	opts.ProgressInterval = 200 * time.Millisecond

	_, err := copy.Image(ctx, pctx, dst, src, opts)
	close(ch)
	<-progDone
	prog.Wait()
	return err
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

// llmman_transfer transfers an image directly from source to destination,
// without ever writing it to the persistent local store. See
// transfer_podman.go for what this picks between (a direct
// docker://→docker:// copy.Image, which streams every blob straight
// through since containers/image already knows each one's digest from
// the source manifest; or a staging-directory fallback for HuggingFace
// and other non-OCI sources, which containers/image has no source
// transport for).
//
//export llmman_transfer
func llmman_transfer(cSource, cDestination *C.char) *C.char {
	changed, err := podmanTransfer(context.Background(), C.GoString(cSource), C.GoString(cDestination))
	if err != nil {
		return errResp(err)
	}
	// See backend_docker.go's llmman_transfer for why data carries this.
	if changed {
		return okResp(transferStatusChanged)
	}
	return okResp(transferStatusUnchanged)
}

// Ensure io is used (imported via shared helpers but referenced here for the build)
var _ = io.Discard

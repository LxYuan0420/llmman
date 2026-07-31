// transfer_common.go — backend-agnostic helpers shared by both
// transfer_docker.go (!podman) and transfer_podman.go (podman).
package main

import (
	"context"
	"net/http"
	"strings"
	"time"

	digest "github.com/opencontainers/go-digest"
)

type transferSourceKind int

const (
	sourceOCI transferSourceKind = iota
	sourceHF
	sourceOther
)

// classifySource mirrors pullToLayout's own OCI-vs-HuggingFace routing
// (known-host shortcuts, then a live /v2/ probe for anything else), so a
// source that `llmman pull` would treat as HuggingFace is also treated as
// HuggingFace here, and likewise for OCI registries. Returns the ref
// normalized the same way (tag defaulted to :latest, hf:// scheme
// resolved to a host-qualified hf.co/... form).
func classifySource(ctx context.Context, ref string) (transferSourceKind, string) {
	for _, scheme := range []string{"ms://", "modelscope://", "ngc://", "s3://", "gs://"} {
		if strings.HasPrefix(ref, scheme) {
			return sourceOther, ref
		}
	}
	if strings.HasPrefix(ref, "/") {
		return sourceOther, ref
	}
	if r, ok := cutAnyPrefix(ref, "hf://", "huggingface://"); ok {
		if strings.Count(r, "/") < 2 {
			r = "hf.co/" + r
		}
		return sourceHF, normalizeTag(r)
	}

	normalized := normalizeTag(ref)
	host := strings.SplitN(normalized, "/", 2)[0]
	if isKnownHFHost(host) {
		return sourceHF, normalized
	}
	if isKnownOCIHost(host) {
		return sourceOCI, normalized
	}
	probeClient := &http.Client{Timeout: 5 * time.Second}
	if isOCIRegistry(ctx, probeClient, host) {
		return sourceOCI, normalized
	}
	return sourceHF, normalized
}

func cutAnyPrefix(s string, prefixes ...string) (string, bool) {
	for _, p := range prefixes {
		if r, ok := strings.CutPrefix(s, p); ok {
			return r, true
		}
	}
	return s, false
}

func normalizeTag(ref string) string {
	if strings.LastIndex(ref, ":") <= strings.LastIndex(ref, "/") {
		return ref + ":latest"
	}
	return ref
}

func shortDigest(d digest.Digest) string {
	h := d.Hex()
	if len(h) > 12 {
		return h[:12]
	}
	return h
}

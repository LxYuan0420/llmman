// progress_state.go – a byte-level progress snapshot for each pull/push
// currently in flight, polled by the Rust daemon (cmd::serve) via
// llmman_progress.
//
// `llmman transfer`'s mpb bars (shared_oci.go's newProgressPool/addLayerBar)
// already show up on an interactive terminal for free, because their FFI
// call runs in the foreground `llmman transfer` process itself, stderr and
// all. `llmman pull`/`llmman push` don't get that for free: the actual
// llmman_pull/llmman_push FFI call happens inside the long-running `llmman
// serve` daemon (see daemon::ensure_server), whose stdio is redirected to a
// log file, not whatever terminal ran `llmman pull`. This snapshot is the
// bridge: cmd::serve polls it every ~200ms (matching mpb's own default
// refresh rate) while a pull/push task is in flight and relays total/
// completed byte counts over its existing NDJSON stream, for the CLI to
// render its own bar from — same underlying numbers as the mpb bars
// already being drawn (uselessly) into the daemon's log, just delivered a
// second way.
//
// One entry per model reference: the daemon serializes concurrent
// pulls/pushes of the *same* model reference (see Rust's per-model lock
// registry in serve.rs), but two different models now pull/push fully in
// parallel, so a single global snapshot (as this used to be) would let
// their numbers interleave and corrupt each other exactly the way a
// single global lock was once needed to prevent. Keying by reference
// instead means each pull/push's own NDJSON stream only ever sees its own
// numbers, however many others are running alongside it.
package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"encoding/json"
	"sync"
)

type progressEntry struct {
	status    string
	total     int64
	completed int64
}

var progressState struct {
	mu      sync.Mutex
	entries map[string]*progressEntry
}

// entryLocked returns key's entry, creating it if absent. Callers must
// hold progressState.mu. An empty key means "don't track" (used by
// llmman_transfer's call paths, which share the same helpers as
// pull/push — see addLayerBar/proxyOrNop — but are polled a different
// way, per this file's own doc comment): every function below is a no-op
// for key == "" rather than accumulating entries nobody will ever poll
// or clean up.
func entryLocked(key string) *progressEntry {
	if progressState.entries == nil {
		progressState.entries = make(map[string]*progressEntry)
	}
	e, ok := progressState.entries[key]
	if !ok {
		e = &progressEntry{}
		progressState.entries[key] = e
	}
	return e
}

// progressReset clears any leftover total/completed from a previous
// pull/push of the same key and sets the initial status text — called
// once, right at the top of llmman_pull/llmman_push, before any bar/
// download work begins.
func progressReset(key, status string) {
	if key == "" {
		return
	}
	progressState.mu.Lock()
	defer progressState.mu.Unlock()
	if progressState.entries == nil {
		progressState.entries = make(map[string]*progressEntry)
	}
	progressState.entries[key] = &progressEntry{status: status}
}

// progressSetStatus updates only the status text (e.g. "pulling manifest"
// -> "pulling"), leaving the running totals untouched.
func progressSetStatus(key, status string) {
	if key == "" {
		return
	}
	progressState.mu.Lock()
	defer progressState.mu.Unlock()
	entryLocked(key).status = status
}

// progressAddTotal adjusts key's running total by delta bytes — positive
// when a new layer/blob's size becomes known and it's about to be
// downloaded/uploaded, negative to undo that once a blob turns out to
// already exist at the destination and no bytes will actually move for it
// (see backend_podman.go's ProgressEventSkipped handling).
func progressAddTotal(key string, delta int64) {
	if key == "" || delta == 0 {
		return
	}
	progressState.mu.Lock()
	defer progressState.mu.Unlock()
	entryLocked(key).total += delta
}

// progressAddCompleted adds delta bytes to key's running completed count.
// Called from the same places that already increment an mpb bar (see
// proxyOrNop in shared_oci.go), so the two stay in lockstep.
func progressAddCompleted(key string, delta int64) {
	if key == "" || delta <= 0 {
		return
	}
	progressState.mu.Lock()
	defer progressState.mu.Unlock()
	entryLocked(key).completed += delta
}

// progressDone removes key's entry once its pull/push has finished
// (successfully or not) — called via defer from llmman_pull/llmman_push,
// so a long-running daemon doesn't accumulate one stale entry per
// distinct model ref it has ever pulled/pushed. Safe to call for a key
// that was never tracked (empty key, or a poll raced past completion).
func progressDone(key string) {
	if key == "" {
		return
	}
	progressState.mu.Lock()
	defer progressState.mu.Unlock()
	delete(progressState.entries, key)
}

// progressSnapshot is the JSON shape returned (as the `data` field of the
// usual response envelope) by llmman_progress.
type progressSnapshot struct {
	Status    string `json:"status"`
	Total     int64  `json:"total"`
	Completed int64  `json:"completed"`
}

// llmman_progress returns key's current pull/push byte-level progress
// snapshot — polled by cmd::serve roughly every 200ms while a pull/push
// task for that same key is in flight. See progressState's own doc
// comment for why this exists. A key with no tracked entry (not yet
// started, or already finished and cleaned up via progressDone) returns a
// zero-value snapshot rather than an error — cmd::serve's poll loop
// treats that as "nothing to report yet" and falls back to plain status
// text.
//
//export llmman_progress
func llmman_progress(cKey *C.char) *C.char {
	key := C.GoString(cKey)
	progressState.mu.Lock()
	var snap progressSnapshot
	if e, ok := progressState.entries[key]; ok {
		snap = progressSnapshot{Status: e.status, Total: e.total, Completed: e.completed}
	}
	progressState.mu.Unlock()
	data, _ := json.Marshal(snap)
	return okResp(string(data))
}

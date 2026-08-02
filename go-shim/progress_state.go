// progress_state.go – a process-wide byte-level progress snapshot for
// whichever pull/push is currently in flight, polled by the Rust daemon
// (cmd::serve) via llmman_progress.
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
// Only one pull/push is tracked at a time: llmman_pull/llmman_push each
// reset this on entry (see their own doc comments), so two requests
// in flight at once would interleave their numbers. That matches every
// other piece of shared daemon state (e.g. the OCI store's own advisory
// locks) — llmman's daemon is designed around one interactive client at a
// time, not concurrent multi-tenant transfers.
package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"encoding/json"
	"sync"
)

var progressState struct {
	mu        sync.Mutex
	status    string
	total     int64
	completed int64
}

// progressReset clears any leftover total/completed from a previous
// pull/push and sets the initial status text — called once, right at the
// top of llmman_pull/llmman_push, before any bar/download work begins.
func progressReset(status string) {
	progressState.mu.Lock()
	defer progressState.mu.Unlock()
	progressState.status = status
	progressState.total = 0
	progressState.completed = 0
}

// progressSetStatus updates only the status text (e.g. "pulling manifest"
// -> "pulling"), leaving the running totals untouched.
func progressSetStatus(status string) {
	progressState.mu.Lock()
	defer progressState.mu.Unlock()
	progressState.status = status
}

// progressAddTotal adjusts the running total by delta bytes — positive
// when a new layer/blob's size becomes known and it's about to be
// downloaded/uploaded, negative to undo that once a blob turns out to
// already exist at the destination and no bytes will actually move for it
// (see backend_podman.go's ProgressEventSkipped handling).
func progressAddTotal(delta int64) {
	if delta == 0 {
		return
	}
	progressState.mu.Lock()
	defer progressState.mu.Unlock()
	progressState.total += delta
}

// progressAddCompleted adds delta bytes to the running completed count.
// Called from the same places that already increment an mpb bar (see
// proxyOrNop in shared_oci.go), so the two stay in lockstep.
func progressAddCompleted(delta int64) {
	if delta <= 0 {
		return
	}
	progressState.mu.Lock()
	defer progressState.mu.Unlock()
	progressState.completed += delta
}

// progressSnapshot is the JSON shape returned (as the `data` field of the
// usual response envelope) by llmman_progress.
type progressSnapshot struct {
	Status    string `json:"status"`
	Total     int64  `json:"total"`
	Completed int64  `json:"completed"`
}

// llmman_progress returns the current pull/push's byte-level progress
// snapshot — polled by cmd::serve roughly every 200ms while a pull/push
// task is in flight. See progressState's own doc comment for why this
// exists.
//
//export llmman_progress
func llmman_progress() *C.char {
	progressState.mu.Lock()
	snap := progressSnapshot{
		Status:    progressState.status,
		Total:     progressState.total,
		Completed: progressState.completed,
	}
	progressState.mu.Unlock()
	data, _ := json.Marshal(snap)
	return okResp(string(data))
}

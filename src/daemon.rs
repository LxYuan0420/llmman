//! Shared client-side helpers for talking to a local `llmman serve`
//! instance over its Ollama-protocol HTTP API (127.0.0.1:17434).
//!
//! Used by any CLI subcommand that acts as a client of that API rather than
//! calling the FFI/model-management logic directly — currently `pull`,
//! `push`, and `launch` — so bare model-name resolution (see
//! `shortnames::resolve_ollama_api`), the local model store, and any
//! already-loaded models are always the daemon's, never duplicated
//! per-invocation.

use std::io::BufRead;
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::Context;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;

/// The fixed loopback origin `llmman serve` always binds to (see
/// cmd::serve's own doc comment on why this isn't configurable).
pub const SERVER: &str = "http://127.0.0.1:17434";

/// Quick synchronous reachability check — none of this module's callers
/// run inside an async runtime, so a plain TCP connect attempt is enough
/// (no need to actually round-trip an HTTP request just to check liveness).
pub fn server_alive() -> bool {
    std::net::TcpStream::connect("127.0.0.1:17434").is_ok()
}

/// Starts `llmman serve` as a background process if one isn't already
/// running, and waits (up to 60s) for it to start accepting connections.
/// The process is intentionally never stopped by this command — once
/// started it keeps running indefinitely, independent of this invocation,
/// so later commands (from this CLI or a concurrent one) reuse it instead
/// of starting a redundant copy.
///
/// stdin/stdout/stderr are all detached from this process (redirected to a
/// log file, or /dev/null if the log file can't be opened): the daemon
/// outlives this command, so anything that inherited its stdout/stderr
/// (a parent shell, a script capturing this command's output, `$(...)`,
/// etc.) would otherwise block forever waiting for those pipes to close,
/// since the daemon never exits to close them itself. The child is also
/// put in its own process group so terminal signals (e.g. Ctrl-C) sent to
/// this command's foreground process group don't reach it either.
///
/// `preload_model`, if non-empty, is passed through as `llmman serve`'s
/// optional positional argument so the daemon starts loading it
/// immediately (see cmd::serve::ServeArgs::model) instead of waiting for
/// the first request that references it. Pass "" when there's nothing to
/// preload — `run`/`pull`/`push` all pass "" so the daemon they spawn
/// stays a plain, model-agnostic `llmman serve` (only `launch` still
/// preloads, since its whole point is warming up one model for the
/// integration it's about to hand off to).
pub fn ensure_server(preload_model: &str) -> anyhow::Result<()> {
    if server_alive() {
        return Ok(());
    }
    let exe = std::env::current_exe().context("could not resolve own executable")?;

    let log_path = crate::default_store(None)
        .ok()
        .and_then(|store| store.parent().map(|p| p.join("serve.log")));

    let mut cmd = Command::new(&exe);
    cmd.arg("serve");
    if !preload_model.is_empty() {
        cmd.arg(preload_model);
    }
    cmd.stdin(Stdio::null());
    // Silently redirect the daemon's stdio to its log file (or /dev/null if
    // that file can't be opened) — no "starting serve" status line here:
    // every caller of ensure_server (run/pull/push/launch) wants to look
    // like a plain client of an already-running server, not announce that
    // it happened to be the one that started it this time. Anyone who
    // needs to know still can — see log_path above / `llmman serve.log`.
    match log_path
        .as_ref()
        .and_then(|p| std::fs::OpenOptions::new().create(true).append(true).open(p).ok())
        .and_then(|f| f.try_clone().ok().map(|f2| (f, f2)))
    {
        Some((out, err)) => {
            cmd.stdout(out);
            cmd.stderr(err);
        }
        None => {
            cmd.stdout(Stdio::null());
            cmd.stderr(Stdio::null());
        }
    }
    detach(&mut cmd);
    cmd.spawn().context("spawn llmman serve")?;

    for _ in 0..120 {
        std::thread::sleep(Duration::from_millis(500));
        if server_alive() {
            return Ok(());
        }
    }
    anyhow::bail!("llmman serve did not start within 60s")
}

/// Puts the about-to-be-spawned child in its own process group (Unix) or
/// process group (Windows), so signals delivered to this process's
/// foreground process group (e.g. Ctrl-C in an interactive shell) don't
/// also terminate the daemon.
#[cfg(unix)]
fn detach(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;
    cmd.process_group(0);
}

#[cfg(windows)]
fn detach(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
}

/// A single line of Ollama's streamed NDJSON progress protocol (see
/// api.ProgressResponse) — status text plus an optional error, and
/// (unlike real Ollama's per-layer digest/total/completed) our own
/// aggregate total/completed byte counts across the whole pull/push, once
/// cmd::serve's stream_ffi_progress has one to report — see that
/// function's own doc comment for where these come from.
#[derive(Deserialize)]
struct ProgressLine {
    status: Option<String>,
    error: Option<String>,
    #[serde(default)]
    total: Option<u64>,
    #[serde(default)]
    completed: Option<u64>,
}

/// The indicatif template shared by `llmman pull`/`llmman push`'s
/// byte-level bar — deliberately similar in shape to `llmman transfer`'s
/// own mpb bars (go-shim/shared_oci.go's addLayerBar) so both commands'
/// output looks like the same family of progress bar.
fn progress_bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{msg:<20} [{bar:32.cyan/blue}] {bytes:>10}/{total_bytes:<10} {bytes_per_sec:>12}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=> ")
}

/// POSTs `{"model": reference}` to `path` (e.g. "/api/pull" or "/api/push")
/// on the local daemon and renders each streamed line to stderr as it
/// arrives: a real byte-level progress bar for any line carrying a nonzero
/// `total` (see ProgressLine), or a plain status line otherwise — matching
/// how `llmman transfer`'s own mpb bars render foreground FFI progress.
/// Returns an error if the stream reports one, or if it ends without ever
/// reporting "success".
pub fn stream_progress(path: &str, reference: &str) -> anyhow::Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(None) // model transfers can take much longer than any sane fixed timeout
        .build()
        .context("build http client")?;
    let resp = client
        .post(format!("{SERVER}{path}"))
        .json(&serde_json::json!({"model": reference}))
        .send()
        .with_context(|| format!("request {path} for {reference}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        anyhow::bail!("{path} {reference}: server returned {status}: {body}");
    }

    let mut saw_success = false;
    let mut last_status = String::new();
    let mut bar: Option<ProgressBar> = None;
    for line in std::io::BufReader::new(resp).lines() {
        let line = line.context("read response stream")?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(msg) = serde_json::from_str::<ProgressLine>(line) else {
            continue; // tolerate stray non-JSON keepalive output
        };
        if let Some(err) = msg.error.filter(|e| !e.is_empty()) {
            if let Some(b) = bar.take() {
                b.abandon(); // leave whatever was drawn in place instead of clearing it
            }
            // Only prefix with `reference` if the error doesn't already
            // mention it — many pull failures (e.g. containerd's "not
            // found") already embed the exact reference themselves, and
            // piling this prefix on unconditionally produced the same
            // reference two or three times over in one error line.
            if err.contains(reference) {
                anyhow::bail!("{err}");
            }
            anyhow::bail!("{reference}: {err}");
        }

        if let Some(total) = msg.total.filter(|&t| t > 0) {
            // A byte-level progress line: render/update the bar instead of
            // printing a new line for every update.
            let pb = bar.get_or_insert_with(|| {
                let pb = ProgressBar::new(total);
                pb.set_style(progress_bar_style());
                pb
            });
            pb.set_length(total);
            pb.set_position(msg.completed.unwrap_or(0).min(total));
            if let Some(status) = &msg.status {
                pb.set_message(status.clone());
            }
            continue;
        }
        // No byte counts on this line: finish/clear any bar in progress
        // before falling back to plain status text, so the two don't
        // interleave on the same terminal lines.
        if let Some(b) = bar.take() {
            b.finish_and_clear();
        }
        if let Some(status) = msg.status {
            if !status.is_empty() && status != last_status {
                println!("{status}");
                last_status = status;
            }
            saw_success = last_status == "success";
        }
    }
    if let Some(b) = bar.take() {
        b.finish_and_clear();
    }
    if !saw_success {
        anyhow::bail!("{reference}: stream ended without a success status");
    }
    Ok(())
}

/// POSTs `{"model": reference}` to `/api/show` and reports whether the
/// daemon's local store already has it — a read-only existence check with
/// no download/pull side effects.
fn model_exists(reference: &str) -> anyhow::Result<bool> {
    let resp = reqwest::blocking::Client::new()
        .post(format!("{SERVER}/api/show"))
        .json(&serde_json::json!({"model": reference}))
        .send()
        .with_context(|| format!("request /api/show for {reference}"))?;
    Ok(resp.status().is_success())
}

/// Ensures `reference` is present in the daemon's local store, pulling it
/// (and streaming progress the same way `llmman pull` does) if it isn't.
///
/// Mirrors ollama's `RunHandler`, which calls `client.Show` before ever
/// entering the interactive/one-shot prompt loop and only falls back to
/// `PullHandler` on a miss — so a bad reference (typo'd tag, malformed
/// `hf.co/...` name, etc.) is reported and aborts the command immediately,
/// instead of only surfacing once the first message is submitted to
/// `/api/chat` (by which point the interactive `> ` prompt has already
/// been shown and read from).
pub fn ensure_model_pulled(reference: &str) -> anyhow::Result<()> {
    if model_exists(reference).unwrap_or(false) {
        return Ok(());
    }
    stream_progress("/api/pull", reference)
}

/// A plain `GET {SERVER}{path}` returning the parsed JSON body — for
/// callers (currently just `ps`) that don't need `stream_progress`'s
/// newline-delimited-JSON streaming, just a single request/response.
pub fn get_json<T: serde::de::DeserializeOwned>(path: &str) -> anyhow::Result<T> {
    let resp = reqwest::blocking::get(format!("{SERVER}{path}"))
        .with_context(|| format!("request {path}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        anyhow::bail!("{path}: server returned {status}: {body}");
    }
    resp.json().with_context(|| format!("parse response from {path}"))
}

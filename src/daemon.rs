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
/// preload (e.g. pull/push, which only need the daemon up, not any
/// particular model warm).
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
    match log_path
        .as_ref()
        .and_then(|p| std::fs::OpenOptions::new().create(true).append(true).open(p).ok())
        .and_then(|f| f.try_clone().ok().map(|f2| (f, f2)))
    {
        Some((out, err)) => {
            cmd.stdout(out);
            cmd.stderr(err);
            if let Some(p) = &log_path {
                eprintln!("[llmman] starting serve (logs: {})...", p.display());
            }
        }
        None => {
            cmd.stdout(Stdio::null());
            cmd.stderr(Stdio::null());
            eprintln!("[llmman] starting serve...");
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
/// api.ProgressResponse) — status text plus an optional error. Every other
/// field (digest/total/completed) is omitted server-side; see
/// cmd::serve::stream_ffi_progress.
#[derive(Deserialize)]
struct ProgressLine {
    status: Option<String>,
    error: Option<String>,
}

/// POSTs `{"model": reference}` to `path` (e.g. "/api/pull" or "/api/push")
/// on the local daemon and prints each streamed status line to stdout as
/// it arrives. Returns an error if the stream reports one, or if it ends
/// without ever reporting "success".
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
            anyhow::bail!("{reference}: {err}");
        }
        if let Some(status) = msg.status {
            if !status.is_empty() && status != last_status {
                println!("{status}");
                last_status = status;
            }
            saw_success = last_status == "success";
        }
    }
    if !saw_success {
        anyhow::bail!("{reference}: stream ended without a success status");
    }
    Ok(())
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

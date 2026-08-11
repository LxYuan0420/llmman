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
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::Context;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// The fixed loopback origin `llmman serve` always binds to (see
/// cmd::serve's own doc comment on why this isn't configurable).
pub const SERVER: &str = "http://127.0.0.1:17434";

/// `SERVER`'s port, broken out on its own for `pid_listening_on` below
/// (which needs a bare `u16`, not a URL) — keep the two in sync.
const PORT: u16 = 17434;

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
        if server_is_current_build() {
            return Ok(());
        }
        // The already-running daemon was started from a now-superseded
        // build of this same binary (see build_fingerprint's doc comment
        // for how that's detected) — e.g. `cargo build`/a package upgrade
        // replaced the file on disk after it was spawned. Left alone, every
        // command would keep silently talking to already-fixed-elsewhere
        // code indefinitely, with no way for a user to know why a bug they
        // just fixed (or a model added to a curated namespace, etc.) still
        // doesn't seem to take effect. Shut it down and fall through to
        // spawn a fresh one below.
        eprintln!("[llmman] restarting llmman serve (running an outdated build)");
        shut_down_stale_server();
    }
    let exe = std::env::current_exe().context("could not resolve own executable")?;

    let log_path = crate::default_store(None)
        .ok()
        .and_then(|store| store.parent().map(|p| p.join("serve.log")));

    // create_dir_all the log's parent (e.g. ~/.local/share/llmman) before
    // ever trying to open it below: `OpenOptions::create(true)` only
    // creates the *file*, never missing intermediate directories, so on a
    // genuinely fresh machine (nothing has ever pulled/served a model
    // yet — confirmed missing on a real macOS CI runner) the `.open()`
    // below would otherwise fail every single time, permanently losing
    // the daemon's entire stdout/stderr to the `None` branch's
    // `Stdio::null()` fallback instead of just this one first call.
    if let Some(p) = log_path.as_ref().and_then(|p| p.parent()) {
        let _ = std::fs::create_dir_all(p);
    }

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

/// A fingerprint of the currently *running* llmman binary's on-disk bytes,
/// computed once and cached for the lifetime of this process.
///
/// This exists to catch a "stale daemon" class of bug: `ensure_server`
/// only ever checks whether *something* is listening on `SERVER` before
/// deciding to reuse it, never whether that something was built from the
/// same source as the client currently asking. If the on-disk binary is
/// replaced (a fresh `cargo build`, a package upgrade, ...) while an
/// `llmman serve` spawned from the old one is still running, every later
/// command silently keeps talking to that already-superseded process
/// forever — e.g. a bare-name resolution bug fixed in a newer build would
/// keep failing exactly as before, with no indication why, until someone
/// thinks to manually kill the old daemon.
///
/// Reads via `/proc/self/exe` rather than `std::env::current_exe()`
/// matters here: on Linux, `current_exe()` re-resolves the *path*, which
/// after a replacement points at the new file — useless for detecting
/// that the currently-executing code differs from what's on disk now. A
/// process's own `/proc/self/exe` is a magic symlink the kernel always
/// resolves to the exact inode it's actually executing, even after that
/// inode has been unlinked from its original path by a newer build, so
/// hashing it always reflects the bytes actually running. Platforms
/// without `/proc` (non-Linux) fall back to `current_exe()`, which can't
/// detect this case but is no worse than not checking at all.
pub fn build_fingerprint() -> &'static str {
    static FP: OnceLock<String> = OnceLock::new();
    FP.get_or_init(|| {
        let bytes = std::fs::read("/proc/self/exe")
            .or_else(|_| std::env::current_exe().and_then(std::fs::read))
            .unwrap_or_default();
        hex::encode(Sha256::digest(&bytes))
    })
}

#[derive(Deserialize, Default)]
struct VersionResponse {
    #[serde(default)]
    build: String,
}

/// Reports whether the already-running daemon (see `server_alive`) was
/// built from the exact same binary bytes as this process — see
/// `build_fingerprint`'s doc comment for why that can differ. Any failure
/// to ask it (network error, an older daemon predating the `/api/version`
/// `build` field entirely, ...) is treated as "not current": safest
/// default is to restart rather than risk silently running stale code, and
/// an older daemon lacking the field is itself exactly the staleness this
/// exists to catch.
fn server_is_current_build() -> bool {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .ok()
        .and_then(|c| c.get(format!("{SERVER}/api/version")).send().ok())
        .and_then(|r| r.json::<VersionResponse>().ok())
        .is_some_and(|v| !v.build.is_empty() && v.build == build_fingerprint())
}

/// Asks an already-running daemon to exit (best-effort — a connection
/// error/reset while it's shutting down is expected, not a failure) and
/// waits up to ~5s for it to stop accepting connections. Falls back to
/// finding and killing the listening process directly (see
/// `pid_listening_on`) if that doesn't work — notably the case for a
/// daemon old enough to predate the `/api/shutdown` route entirely (i.e.
/// exactly the kind of staleness this whole mechanism exists to replace),
/// which just 404s and keeps running. Gives up (leaving the stale daemon
/// in place — `ensure_server`'s subsequent spawn will then simply fail to
/// bind the port, surfacing as a normal "did not start" error rather than
/// silently doing nothing) after a further ~10s.
fn shut_down_stale_server() {
    if let Ok(client) = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
    {
        let _ = client.post(format!("{SERVER}/api/shutdown")).send();
    }
    for _ in 0..20 {
        if !server_alive() {
            return;
        }
        std::thread::sleep(Duration::from_millis(250));
    }

    force_kill_listening_process();
    for _ in 0..40 {
        if !server_alive() {
            return;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// Sends SIGTERM (then, if it's still alive half a second later, SIGKILL)
/// to whatever process `pid_listening_on(PORT)` finds — the graceful
/// `/api/shutdown` fallback for daemons too old to have that route. A
/// no-op if the port's owning process can't be identified, or on
/// non-Linux platforms (see `pid_listening_on`).
#[cfg(target_os = "linux")]
fn force_kill_listening_process() {
    let Some(pid) = pid_listening_on(PORT) else {
        return;
    };
    unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
    std::thread::sleep(Duration::from_millis(500));
    if server_alive() {
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
    }
}

#[cfg(not(target_os = "linux"))]
fn force_kill_listening_process() {}

/// Finds the PID of the process with a listening socket bound to
/// `127.0.0.1:<port>`, by cross-referencing `/proc/net/tcp` (socket inode
/// for the listening endpoint) against every process's `/proc/<pid>/fd/*`
/// entries (which symlinks matching sockets do reference by that same
/// inode). There's no portable syscall for "which PID owns this port", and
/// shelling out to `lsof`/`fuser` isn't guaranteed to be installed, so this
/// reads the same `/proc` files those tools do themselves.
#[cfg(target_os = "linux")]
fn pid_listening_on(port: u16) -> Option<u32> {
    const TCP_LISTEN: &str = "0A";
    let hex_port = format!("{port:04X}");

    let table = std::fs::read_to_string("/proc/net/tcp").ok()?;
    let inode = table.lines().skip(1).find_map(|line| {
        let f: Vec<&str> = line.split_whitespace().collect();
        // Columns: sl local_address rem_address st ... inode
        if f.len() < 10 || f[3] != TCP_LISTEN {
            return None;
        }
        let (_, local_port) = f[1].split_once(':')?;
        if local_port.eq_ignore_ascii_case(&hex_port) {
            Some(f[9].to_string())
        } else {
            None
        }
    })?;
    let target = format!("socket:[{inode}]");

    for proc_entry in std::fs::read_dir("/proc").ok()?.flatten() {
        let Some(pid) = proc_entry
            .file_name()
            .to_str()
            .and_then(|s| s.parse::<u32>().ok())
        else {
            continue;
        };
        let Ok(fds) = std::fs::read_dir(proc_entry.path().join("fd")) else {
            continue;
        };
        for fd in fds.flatten() {
            if std::fs::read_link(fd.path()).is_ok_and(|l| l.to_string_lossy() == target) {
                return Some(pid);
            }
        }
    }
    None
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

// ---------------------------------------------------------------------------
// Windows-only: stop this process's own stdio handles from leaking into
// whatever it spawns
// ---------------------------------------------------------------------------

/// Root-caused a real Windows-only E2E hang: `cargo test`'s own harness
/// (tests/launch_e2e.rs's `spawn_with_timeout`) spawns `llmman run` with
/// piped stdout/stderr to capture its output live, and that child's own
/// `try_wait()`-detected exit was always reported correctly and promptly
/// — the hang was in the *next* step, `stdout_thread`/`stderr_thread`'s
/// `.join()`, which blocks on each reader thread's `read()` returning
/// `Ok(0)` (EOF). A pipe only reaches EOF once *every* handle to its
/// write end is closed — not just the one held by the process the reader
/// thinks it's reading from.
///
/// On Windows, any `CreateProcess` call that redirects a child's stdio via
/// a handle (a `File`, or another pipe — exactly what `ensure_server`
/// below does, redirecting the daemon's stdout/stderr to its log file)
/// requires `bInheritHandles = TRUE`, and that flag does not selectively
/// inherit only the handles you asked to redirect — it inherits *every*
/// handle in the parent's own handle table that's currently marked
/// inheritable. `llmman run`'s own stdout/stderr handles — received from
/// the test harness above specifically so they could be piped and
/// captured, which requires them to be inheritable in the first place —
/// are exactly such handles. So when `llmman run` calls `ensure_server`
/// and spawns the detached daemon below, that daemon process — which is
/// deliberately left running indefinitely — ends up with its own
/// duplicate, still-open handle to `llmman run`'s original stdout/stderr
/// pipes, even though its own stdio was separately, correctly redirected
/// to the log file. The pipe's write end therefore never fully closes,
/// the test harness's reader thread blocks on `read()` forever waiting
/// for an EOF that can now never come, and the *entire* E2E job times out
/// 45 minutes later with no error of its own — indistinguishable from
/// there being no bug at all up to that point, since `llmman run` itself
/// already exited successfully.
///
/// The fix: explicitly clear the inheritable flag on this process's own
/// three standard handles, once, as early as possible (see `main.rs`) —
/// before this process (or anything it calls into) ever spawns another
/// child with `bInheritHandles = TRUE` for any reason. This does not
/// affect this process's own use of those handles at all (only whether a
/// *future child* of this process can inherit them), and is safe to call
/// unconditionally even when nothing was ever piped in the first place
/// (`GetStdHandle` returning a real console handle, or even
/// `INVALID_HANDLE_VALUE`/null when there's no console at all, are all
/// handled as no-ops below).
#[cfg(windows)]
mod win_handles {
    use std::ffi::c_void;

    #[allow(non_snake_case)]
    extern "system" {
        fn GetStdHandle(nStdHandle: i32) -> *mut c_void;
        fn SetHandleInformation(hObject: *mut c_void, dwMask: u32, dwFlags: u32) -> i32;
    }

    const STD_INPUT_HANDLE: i32 = -10;
    const STD_OUTPUT_HANDLE: i32 = -11;
    const STD_ERROR_HANDLE: i32 = -12;
    const HANDLE_FLAG_INHERIT: u32 = 0x0000_0001;

    pub fn disable_std_handle_inheritance() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            for which in [STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE] {
                // SAFETY: GetStdHandle/SetHandleInformation are ordinary
                // kernel32 calls; `handle` is only ever passed to the one
                // API that accepts exactly this value (including the
                // documented NULL/INVALID_HANDLE_VALUE "no such stream"
                // cases, which SetHandleInformation simply fails on
                // harmlessly — the return value is deliberately ignored).
                unsafe {
                    let handle = GetStdHandle(which);
                    if !handle.is_null() && handle != usize::MAX as *mut c_void {
                        let _ = SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0);
                    }
                }
            }
        });
    }
}

/// See `win_handles`' own module doc comment. Must be called once, as
/// early as possible in `main()` — a no-op on every platform but Windows.
pub fn disable_std_handle_inheritance() {
    #[cfg(windows)]
    win_handles::disable_std_handle_inheritance();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_fingerprint_is_stable_and_nonempty() {
        // Same process, called twice: must be the exact same string both
        // times (OnceLock-cached) and never empty (this test process's own
        // binary is always readable, unlike some hypothetical daemon under
        // test).
        let a = build_fingerprint();
        let b = build_fingerprint();
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn pid_listening_on_finds_a_real_listener_and_rejects_a_free_port() {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let port = listener.local_addr().unwrap().port();
        assert_eq!(pid_listening_on(port), Some(std::process::id()));

        // An unbound port near the ephemeral range almost certainly has no
        // listener at all right after the one above is dropped.
        drop(listener);
        assert_eq!(pid_listening_on(port), None);
    }
}

//! End-to-end tests for `llmman launch <integration> --model qwen3.5:0.8b`.
//!
//! These exercise the real launch path: a real, auto-started `llmman
//! serve`, a real pulled model (`docker.io/ai/qwen3.5:0.8b`, resolved
//! from the bare short name the same way `llmman launch`/`pull` always
//! resolve one — see `shortnames::resolve_ollama_api`), a real
//! `llama-server` backing it, and the real third-party CLI under test
//! (`claude`, `opencode`, `codex`) — not mocks. That's the only way this
//! actually verifies anything: every one of the three bugs this file's
//! tests were written to catch (see below) only ever showed up against
//! the real binaries, never in isolation.
//!
//! Each test prints a message and skips (rather than failing) when a
//! prerequisite isn't installed in the current environment, since none of
//! that is under this crate's control:
//!
//!   - `llama-server` on PATH — required by every one of these (llmman's
//!     daemon can't serve any model without it).
//!   - the specific integration binary under test (`claude`, `opencode`,
//!     or `codex`) on PATH.
//!
//! Network access to pull `docker.io/ai/qwen3.5:0.8b` (~740MB, on the
//! first run only — later runs reuse whatever's already in the daemon's
//! store) is assumed available and NOT treated as skippable: a pull
//! failure is a real failure here, not an environment-setup gap.
//!
//! `llmman serve` is a process-wide singleton bound to a fixed loopback
//! port (127.0.0.1:17434 — see `daemon::SERVER`), so these tests can't
//! isolate their own daemon instance from each other or from one already
//! running on the machine: `llmman launch` (via `daemon::ensure_server`)
//! just reuses whatever's already listening there, preloaded with
//! whichever model first asked for it. What each test *does* isolate is
//! `HOME` (and so each integration's own config directory — `~/.claude`,
//! `~/.codex`, `~/.config/opencode`), by pointing its child process at a
//! fresh temp directory. `SERIAL` below keeps the three tests from
//! running concurrently regardless, both to avoid racing to spawn that
//! one shared daemon and to keep three real model launches from
//! competing for the same GPU/CPU at once.
//!
//! Not run as part of `cargo build` or CI (`.github/workflows/ci.yml`
//! never invokes `cargo test`) — run explicitly:
//!
//!   cargo test --release --test launch_e2e -- --nocapture --test-threads=1
//!
//! # Regressions this file guards against
//!
//! All three were found by actually running these exact commands against
//! a real model, not by inspection:
//!
//!   - `claude`: real Claude Code sessions inject a second `role:"system"`
//!     message later in the conversation (e.g. an available-agents/skills
//!     reminder) in addition to its leading system prompt. Qwen3.5's chat
//!     template raises a hard Jinja error ("System message must be at the
//!     beginning") the moment that happens, so every real multi-turn
//!     request 500'd and Claude Code retried in a loop until giving up —
//!     fixed in `cmd::serve::handle_anthropic_messages` by folding every
//!     system-role turn into one leading message.
//!   - `codex`: the config `write_codex_config` wrote (a `[profiles.llmman]`
//!     table in `config.toml`) is a format current codex (0.134+) refuses
//!     to load at all — fixed by writing the sibling
//!     `~/.codex/llmman.config.toml` overlay codex now expects instead.
//!   - `codex`: real `codex exec` always includes Responses-API tool
//!     entries llama-server's `/v1/responses` rejects outright
//!     (`"'type' of tool must be 'function'"` for a `"namespace"`-typed
//!     sub-agent tool bundle, and for the bare `{"type":"web_search"}`
//!     entry) — fixed in `cmd::serve::filter_non_function_tools`. Real
//!     `codex exec` also always carries a `developer`-role item alongside
//!     its top-level `instructions`, which llama-server's own Responses
//!     conversion turns into a second, misplaced `system`-role chat
//!     message (a confirmed, unresolved upstream llama.cpp gap — see
//!     ggml-org/llama.cpp#20733/#23423) — fixed in
//!     `cmd::serve::consolidate_responses_instructions`.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Mutex, Once};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The exact short name every `llmman launch ... --model` invocation in
/// this suite uses — resolves (via `resolve_ollama_api`, the same path
/// real CLI usage takes) to `docker.io/ai/qwen3.5:0.8b`, a ~740MB Q4_K_M
/// quantization small enough to pull and run within a normal test
/// timeout.
const MODEL: &str = "qwen3.5:0.8b";

/// A short, literal prompt every integration is asked to answer. Kept
/// deliberately simple: the small quantized model's answer quality isn't
/// what's under test here, the launch/env/config plumbing is.
const PROMPT: &str = "Reply with exactly the single word: pong";

/// How long a single command (an `llmman launch` invocation, or
/// `warm_model`'s `llmman run`) may run before these tests give up and
/// fail it as hung, rather than waiting forever. This has to comfortably
/// cover a cold pull of the ~740MB model plus llama-server startup, which
/// only happens once per fresh daemon (in `warm_model`, not in every
/// individual test) — but that one pull is a real download over
/// whatever network the daemon's machine has, so this is generous rather
/// than tight: a CI run has already hit the low end of plausible pull
/// times at 300s.
const TIMEOUT: Duration = Duration::from_secs(600);

/// Serializes the tests in this file: they'd otherwise run in parallel
/// threads (the default `cargo test` behavior) and race to spawn
/// `llmman serve` for the one shared daemon slot every `llmman launch`
/// invocation targets (see the module doc comment).
static SERIAL: Mutex<()> = Mutex::new(());

/// Guards `warm_model` so its real (slow, first-time) work happens only
/// once per test binary run, no matter how many of the three tests below
/// actually reach it.
static WARM: Once = Once::new();

fn llmman_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_llmman"))
}

/// True if `bin` resolves on `PATH` — enough for test-skip purposes
/// without depending on llmman's own (private) `launch::find_on_path`.
/// Mirrors that function's own Windows handling (see its doc comment):
/// a bare name on Windows is checked against `.exe`/`.cmd`/`.bat`, not
/// just the bare file itself, since every integration under test here is
/// installed via `npm install -g` — which on Windows never produces a
/// bare, extension-less file — and this needs to agree with what
/// `find_on_path` can actually locate. A test skipping here despite the
/// CLI being on `PATH` (or, worse, not skipping and then hitting
/// `find_on_path`'s own "not installed" error inside `llmman launch`)
/// would both be this check silently drifting from that one.
fn on_path(bin: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    if cfg!(windows) {
        const EXTS: &[&str] = &["exe", "cmd", "bat"];
        std::env::split_paths(&path)
            .any(|dir| EXTS.iter().any(|ext| dir.join(format!("{bin}.{ext}")).is_file()))
    } else {
        std::env::split_paths(&path).any(|dir| dir.join(bin).is_file())
    }
}

/// A fresh, unique temp `HOME` for one test's child process — isolates
/// each integration's own config directory (`~/.claude`, `~/.codex`,
/// `~/.config/opencode`) both from the real developer's and from the
/// other tests in this file.
fn fresh_home(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("llmman-e2e-{label}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("create temp HOME");
    dir
}

/// Runs `cmd`, waiting up to `timeout` and killing (then panicking with
/// `description` in the message) it if it hasn't exited by then.
/// stdout/stderr are drained on background threads while polling for
/// exit, rather than read only after the child finishes, so a chatty
/// child can't deadlock on a full pipe buffer before the timeout ever
/// gets a chance to fire.
fn spawn_with_timeout(mut cmd: Command, timeout: Duration, description: &str) -> std::process::Output {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().unwrap_or_else(|e| panic!("spawn {description}: {e}"));
    let mut stdout_pipe = child.stdout.take().expect("child stdout");
    let mut stderr_pipe = child.stderr.take().expect("child stderr");
    let stdout_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout_pipe.read_to_end(&mut buf);
        buf
    });
    let stderr_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr_pipe.read_to_end(&mut buf);
        buf
    });

    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll child status") {
            break status;
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            // Join the reader threads (rather than just discarding them)
            // before panicking: killing the child closes its end of both
            // pipes, so read_to_end on each should return promptly — and
            // whatever the process printed before it got stuck is exactly
            // what's needed to tell "genuinely still downloading/loading
            // a large model" apart from "stuck in a real hang" from the
            // outside, instead of a bare timeout message that can't
            // distinguish either.
            let stdout = stdout_thread.join().map(|b| String::from_utf8_lossy(&b).into_owned())
                .unwrap_or_else(|_| "<stdout reader thread panicked>".into());
            let stderr = stderr_thread.join().map(|b| String::from_utf8_lossy(&b).into_owned())
                .unwrap_or_else(|_| "<stderr reader thread panicked>".into());
            panic!(
                "{description} did not finish within {timeout:?} — likely a hang\n\
                 --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
            );
        }
        std::thread::sleep(Duration::from_millis(200));
    };

    std::process::Output {
        status,
        stdout: stdout_thread.join().expect("join stdout reader thread"),
        stderr: stderr_thread.join().expect("join stderr reader thread"),
    }
}

/// Forces `MODEL` to be pulled and fully loaded — via `llmman run`, our
/// own trusted client, never a third-party CLI — before any of this
/// file's tests hand off to one. Real third-party AI-SDK-based clients
/// (opencode's included) have been observed to retry a failed connection
/// indefinitely rather than giving up on it, so if one of them happens to
/// be the first to reach the daemon while `MODEL` is still on its way in
/// (a cold pull of ~740MB, plus llama-server startup), any transient
/// connection hiccup during that window risks wedging that client forever
/// — a failure this suite can only detect (via each test's own `TIMEOUT`),
/// never recover from. Paying that cold-start cost here first, against
/// code this suite controls, removes the window entirely: by the time any
/// test execs a real integration, `MODEL` has already answered a real
/// prompt successfully at least once.
fn warm_model() {
    WARM.call_once(|| {
        let mut cmd = Command::new(llmman_bin());
        cmd.arg("run").arg(MODEL).arg(PROMPT);
        let output = spawn_with_timeout(cmd, TIMEOUT, "llmman run (model warm-up)");
        assert!(
            output.status.success(),
            "llmman run {MODEL} {PROMPT:?} (model warm-up) failed (status: {:?})\n\
             --- stdout ---\n{}\n--- stderr ---\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    });
}

/// Runs `llmman launch <integration> --model qwen3.5:0.8b -- <extra_args>`
/// with `home` as its `HOME`. See `warm_model` and `spawn_with_timeout`.
fn run_launch(home: &Path, integration: &str, extra_args: &[&str]) -> std::process::Output {
    warm_model();

    let mut cmd = Command::new(llmman_bin());
    cmd.arg("launch").arg(integration).arg("--model").arg(MODEL);
    if !extra_args.is_empty() {
        cmd.arg("--").args(extra_args);
    }
    cmd.env("HOME", home)
        .env("USERPROFILE", home)
        .env("XDG_CONFIG_HOME", home.join(".config"))
        .env("XDG_DATA_HOME", home.join(".local/share"));

    spawn_with_timeout(
        cmd,
        TIMEOUT,
        &format!("`llmman launch {integration} --model {MODEL} -- {extra_args:?}`"),
    )
}

/// Shared assertion for all three integrations: the launch must succeed,
/// and the model's real answer must show up somewhere in stdout.
fn assert_launch_succeeded(integration: &str, extra_args: &[&str], output: &std::process::Output) {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "`llmman launch {integration} --model {MODEL} -- {extra_args:?}` failed \
         (status: {:?})\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
        output.status
    );
    assert!(
        stdout.to_lowercase().contains("pong"),
        "expected {integration}'s reply to contain \"pong\"\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
}

#[test]
fn launch_claude_with_model() {
    let _guard = SERIAL.lock().unwrap();
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("claude") {
        eprintln!("skipping: claude not on PATH — https://code.claude.com/docs/en/quickstart");
        return;
    }

    let home = fresh_home("claude");
    // `-p`/`--print`: Claude Code's non-interactive one-shot mode — the
    // scriptable equivalent of typing a message into the interactive TUI
    // `llmman launch claude --model qwen3.5:0.8b` would otherwise open.
    let extra_args = ["-p", PROMPT];
    let output = run_launch(&home, "claude", &extra_args);
    assert_launch_succeeded("claude", &extra_args, &output);
}

#[test]
fn launch_opencode_with_model() {
    let _guard = SERIAL.lock().unwrap();
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("opencode") {
        eprintln!("skipping: opencode not on PATH — https://opencode.ai");
        return;
    }

    let home = fresh_home("opencode");
    // `run <message>`: opencode's non-interactive one-shot mode.
    // --print-logs --log-level DEBUG: opencode's provider (configured via
    // OPENCODE_CONFIG_CONTENT's "npm" field — see launch::opencode_config)
    // is installed on demand into ~/.config/opencode/node_modules the
    // first time a fresh HOME uses it, which showed up as a slow/hanging
    // step in one environment during development; keep this on so a CI
    // failure's logs show exactly what opencode was doing right up to a
    // timeout, instead of just the banner and silence.
    let extra_args = ["run", PROMPT, "--print-logs", "--log-level", "DEBUG"];
    let output = run_launch(&home, "opencode", &extra_args);
    assert_launch_succeeded("opencode", &extra_args, &output);
}

#[test]
fn launch_codex_with_model() {
    let _guard = SERIAL.lock().unwrap();
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("codex") {
        eprintln!("skipping: codex not on PATH — npm install -g @openai/codex");
        return;
    }

    let home = fresh_home("codex");
    // `exec <prompt>`: codex's non-interactive one-shot mode.
    let extra_args = ["exec", PROMPT];
    let output = run_launch(&home, "codex", &extra_args);
    assert_launch_succeeded("codex", &extra_args, &output);
}

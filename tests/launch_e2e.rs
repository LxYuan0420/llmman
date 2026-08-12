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
use std::sync::{Arc, Mutex, Once};
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

/// Locks [`SERIAL`], recovering from poisoning instead of panicking with
/// an opaque `PoisonError` (via the default `.lock().unwrap()`) — this
/// mutex only ever guards *ordering*, never any data an earlier panicking
/// test could have left inconsistent (its payload is `()`), so there's
/// nothing to actually protect against here. Without this, one test
/// legitimately failing (including via `launch_and_assert`'s own retries
/// being exhausted) poisons the lock for the remaining two, which then
/// fail with a meaningless `PoisonError` instead of ever getting to run
/// and report their own real result — exactly what happened live in CI:
/// a `launch_claude_with_model` failure took down `launch_codex_with_model`
/// and `launch_opencode_with_model` too, hiding whether either of *those*
/// would have actually passed on their own.
fn lock_serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

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

/// Returns the last `n` bytes of `buf` as a lossy string, prefixed with an
/// ellipsis marker when truncated — used by spawn_with_timeout's
/// heartbeat to show recent output without the printed heartbeat itself
/// growing unboundedly as a slow child produces more and more of it.
fn tail_str(buf: &[u8], n: usize) -> String {
    if buf.len() > n {
        format!("...<{} bytes total>...{}", buf.len(), String::from_utf8_lossy(&buf[buf.len() - n..]))
    } else {
        String::from_utf8_lossy(buf).into_owned()
    }
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
    // Kept permanently (not removed once the investigation that added it
    // concluded — see this repo's own git history): a real Windows/macOS
    // hang in this file once showed only a bare "still running" heartbeat
    // with no insight into what the child itself was doing at that point
    // — its own stdout/stderr used to only become visible once it exited
    // (or was killed at the timeout), by which point a forceful GH
    // Actions job cancellation (the outer timeout-minutes, not this
    // function's own) can lose the last several minutes of log output
    // entirely. Reading into shared buffers the heartbeat can peek at
    // live (rather than only handing them back once each reader thread's
    // `read_to_end` finally returns) means a hung/slow child's progress
    // up to the very last heartbeat before that happens stays visible
    // even if everything after it is lost — useful for whatever the next
    // one of these turns out to be, not just the one this was built for.
    eprintln!("[spawn_with_timeout] pid={} spawned: {description}", child.id());
    let mut stdout_pipe = child.stdout.take().expect("child stdout");
    let mut stderr_pipe = child.stderr.take().expect("child stderr");
    let stdout_buf = Arc::new(Mutex::new(Vec::new()));
    let stderr_buf = Arc::new(Mutex::new(Vec::new()));
    let stdout_thread = {
        let buf = Arc::clone(&stdout_buf);
        std::thread::spawn(move || {
            let mut chunk = [0u8; 4096];
            loop {
                match stdout_pipe.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => buf.lock().unwrap().extend_from_slice(&chunk[..n]),
                }
            }
        })
    };
    let stderr_thread = {
        let buf = Arc::clone(&stderr_buf);
        std::thread::spawn(move || {
            let mut chunk = [0u8; 4096];
            loop {
                match stderr_pipe.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => buf.lock().unwrap().extend_from_slice(&chunk[..n]),
                }
            }
        })
    };

    let start = Instant::now();
    let mut last_heartbeat = start;
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll child status") {
            eprintln!(
                "[spawn_with_timeout] pid={} exited after {:?}: {description}",
                child.id(),
                start.elapsed()
            );
            break status;
        }
        if last_heartbeat.elapsed() > Duration::from_secs(30) {
            last_heartbeat = Instant::now();
            eprintln!(
                "[spawn_with_timeout] pid={} still running after {:?}: {description}\n  stdout tail: {:?}\n  stderr tail: {:?}",
                child.id(),
                start.elapsed(),
                tail_str(&stdout_buf.lock().unwrap(), 300),
                tail_str(&stderr_buf.lock().unwrap(), 300),
            );
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            // Join the reader threads (rather than just discarding them)
            // before panicking: killing the child closes its end of both
            // pipes, so each thread's blocking read should return promptly
            // — and whatever the process printed before it got stuck is
            // exactly what's needed to tell "genuinely still downloading/
            // loading a large model" apart from "stuck in a real hang"
            // from the outside, instead of a bare timeout message that
            // can't distinguish either.
            let _ = stdout_thread.join();
            let _ = stderr_thread.join();
            let stdout = String::from_utf8_lossy(&stdout_buf.lock().unwrap()).into_owned();
            let stderr = String::from_utf8_lossy(&stderr_buf.lock().unwrap()).into_owned();
            panic!(
                "{description} did not finish within {timeout:?} — likely a hang\n\
                 --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
            );
        }
        std::thread::sleep(Duration::from_millis(200));
    };

    stdout_thread.join().expect("join stdout reader thread");
    stderr_thread.join().expect("join stderr reader thread");
    std::process::Output {
        status,
        stdout: Arc::try_unwrap(stdout_buf).expect("stdout_buf uniquely owned after join").into_inner().unwrap(),
        stderr: Arc::try_unwrap(stderr_buf).expect("stderr_buf uniquely owned after join").into_inner().unwrap(),
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
///
/// `--think false`: qwen3.5's thinking mode has been observed (directly,
/// against a real windows-2025/macos-15 run — see this repo's own git
/// history) to occasionally degenerate into repeating the same handful of
/// reasoning sentences indefinitely instead of ever reaching an answer,
/// hanging this warm-up for the full `TIMEOUT` and poisoning `WARM` for
/// every other test in the same run. That failure mode lives entirely
/// inside the "Thinking:" block this flag skips outright.
///
/// `--num-predict 64`: disabling thinking alone turned out not to be
/// enough — a *second*, real windows-2025 run (this one with `--think
/// false` already in effect) still hung the full `TIMEOUT`, this time
/// repeating a single token in the actual answer instead of inside a
/// thinking block. Nothing about *why* a small quantized model's sampling
/// might degenerate is reliably preventable from here; a hard ceiling on
/// how many tokens it's even allowed to generate is. 64 is generous for
/// this file's own PROMPT (a literal one-word answer) while still capping
/// a worst-case degenerate run at a few seconds, not `TIMEOUT`'s full 600.
///
/// Neither flag is used for the three real per-integration launches
/// below: they exercise real third-party clients exactly as a real user
/// would run them, unable to pass either flag those clients don't
/// themselves expose.
fn warm_model() {
    WARM.call_once(|| {
        eprintln!("[warm_model] starting");
        let mut cmd = Command::new(llmman_bin());
        cmd.arg("run").arg(MODEL).arg("--think").arg("false").arg("--num-predict").arg("64").arg(PROMPT);
        let output = spawn_with_timeout(cmd, TIMEOUT, "llmman run (model warm-up)");
        eprintln!("[warm_model] done, status={:?}", output.status);
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
    eprintln!("[run_launch] {integration}: calling warm_model()");
    warm_model();
    eprintln!("[run_launch] {integration}: warm_model() returned, spawning launch");

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

/// How many times [`launch_and_assert`] will (re-)run a whole
/// `llmman launch <integration> ...` invocation, against a fresh `HOME`
/// each time, before giving up — see that function's own doc comment for
/// why this exists at all.
const MAX_ATTEMPTS: u32 = 3;

/// Runs `llmman launch <integration> --model qwen3.5:0.8b -- <extra_args>`
/// (via [`run_launch`], against a fresh temp `HOME` each attempt) and
/// asserts it succeeded and that the model's real answer shows up in
/// stdout — retrying up to [`MAX_ATTEMPTS`] times, but *only* when the
/// launch itself exited successfully and merely didn't happen to answer
/// with "pong" this particular time.
///
/// That retry exists because real batched inference through llama-server
/// (`n_slots` continuous batching serving whatever concurrent requests
/// each of these real, un-mocked third-party CLIs' own startup traffic —
/// title generation, memory/skill checks, etc. — happens to send
/// alongside the actual prompt) is not bit-for-bit deterministic run to
/// run even for identical input text: a small quantized model's sampling
/// can tip a different way depending on exactly how those concurrent
/// requests happen to batch together. Directly observed in practice
/// (not hypothetical): real `qwen3.5:0.8b` runs through a real `claude
/// --model ... -p ...` occasionally answer with a nonsensical safety
/// refusal, or attempt a spurious tool call, instead of the one literal
/// word asked for — see this file's module doc comment's own catalogue of
/// similar small-model degeneracy already fought here (`warm_model`'s
/// `--think false --num-predict 64`). A *real* regression (a non-zero
/// exit status: an actual crash, a rejected request, a 500) is never
/// retried — only "the process exited 0 but the live model's one-shot
/// answer this time didn't include the word" is, so this can't mask the
/// actual API-compat bugs this suite exists to catch (see that same doc
/// comment).
fn launch_and_assert(integration: &str, extra_args: &[&str]) {
    let mut last_output = None;
    for attempt in 1..=MAX_ATTEMPTS {
        let home = fresh_home(integration);
        let output = run_launch(&home, integration, extra_args);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "`llmman launch {integration} --model {MODEL} -- {extra_args:?}` failed \
             (status: {:?})\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
            output.status
        );
        if stdout.to_lowercase().contains("pong") {
            return;
        }
        eprintln!(
            "[test] {integration}: attempt {attempt}/{MAX_ATTEMPTS} succeeded but the reply \
             didn't contain \"pong\" (small-model sampling variance — see launch_and_assert's \
             own doc comment); {}",
            if attempt < MAX_ATTEMPTS { "retrying with a fresh HOME" } else { "giving up" }
        );
        last_output = Some(output);
    }
    let output = last_output.expect("loop runs at least once, so this is always set");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    panic!(
        "expected {integration}'s reply to contain \"pong\" after {MAX_ATTEMPTS} attempts\n\
         --- stdout (last attempt) ---\n{stdout}\n--- stderr (last attempt) ---\n{stderr}"
    );
}

#[test]
fn launch_claude_with_model() {
    eprintln!("[test] launch_claude_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_claude_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("claude") {
        eprintln!("skipping: claude not on PATH — https://code.claude.com/docs/en/quickstart");
        return;
    }

    // `-p`/`--print`: Claude Code's non-interactive one-shot mode — the
    // scriptable equivalent of typing a message into the interactive TUI
    // `llmman launch claude --model qwen3.5:0.8b` would otherwise open.
    launch_and_assert("claude", &["-p", PROMPT]);
}

#[test]
fn launch_opencode_with_model() {
    eprintln!("[test] launch_opencode_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_opencode_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("opencode") {
        eprintln!("skipping: opencode not on PATH — https://opencode.ai");
        return;
    }

    // `run <message>`: opencode's non-interactive one-shot mode.
    // --print-logs --log-level DEBUG: opencode's provider (configured via
    // OPENCODE_CONFIG_CONTENT's "npm" field — see launch::opencode_config)
    // is installed on demand into ~/.config/opencode/node_modules the
    // first time a fresh HOME uses it, which showed up as a slow/hanging
    // step in one environment during development; keep this on so a CI
    // failure's logs show exactly what opencode was doing right up to a
    // timeout, instead of just the banner and silence.
    launch_and_assert("opencode", &["run", PROMPT, "--print-logs", "--log-level", "DEBUG"]);
}

#[test]
fn launch_codex_with_model() {
    eprintln!("[test] launch_codex_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_codex_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("codex") {
        eprintln!("skipping: codex not on PATH — npm install -g @openai/codex");
        return;
    }

    // `exec <prompt>`: codex's non-interactive one-shot mode.
    launch_and_assert("codex", &["exec", PROMPT]);
}


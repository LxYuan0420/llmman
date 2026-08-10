//! `llmman run` — interactive chat or one-shot prompt.
//!
//! Interactive mode uses a raw-mode readline ported directly from ollama's
//! readline package (readline/readline.go, readline/term.go).  The key
//! mechanism for paste detection mirrors ollama exactly:
//!
//!   // ollama (Go)
//!   if i.Terminal.reader.Buffered() > 0 { draining = true }
//!
//!   // llmman (Rust)
//!   if !reader.buffer().is_empty() { draining = true; }
//!
//! When the user pastes, the terminal sends all characters to the PTY buffer
//! at once.  BufReader fills its internal buffer in one syscall.  After
//! read()ing one byte, buffer() is non-empty ↔ we are draining a paste.
//! While draining, a '\n' (CharCtrlJ) submits the line like Enter does
//! (same as ollama).  When not draining, '\n' is Ctrl-J multiline.

use std::io::{self, IsTerminal, Write};

use anyhow::Context;
use clap::Args;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use crate::daemon::SERVER;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Args, Debug)]
pub struct RunArgs {
    #[arg(value_name = "MODEL")]
    pub model: String,
    /// Forwarded as Ollama's own top-level `think` field on every request
    /// this sends (see cmd::serve's think_to_chat_template_kwargs) —
    /// `--think false` disables a reasoning model's thinking block
    /// entirely, `--think true` forces it on. Omitted (leaving the
    /// model's own template default in effect) if not passed at all.
    #[arg(long)]
    pub think: Option<bool>,
    /// Forwarded as `options.num_predict` (Ollama's own name for
    /// llama-server's `max_tokens` — see opt_u32 in cmd::serve) on every
    /// request this sends: a hard ceiling on how many tokens a single
    /// reply may generate, regardless of *why* it might otherwise run
    /// away (a real, no-stopping-condition-hit degenerate loop, observed
    /// directly with qwen3.5:0.8b even with `--think false` and a
    /// repeat_penalty already in effect — see this repo's own git history
    /// — is not reliably preventable any other way). Omitted (no ceiling
    /// at all, matching Ollama's own num_predict default of -1) if not
    /// passed.
    #[arg(long)]
    pub num_predict: Option<u32>,
    #[arg(value_name = "PROMPT", trailing_var_arg = true, allow_hyphen_values = true)]
    pub prompt: Vec<String>,
}

/// Per-request knobs `chat_submit`'s every caller in this file threads
/// through unchanged from `RunArgs` — see `RunArgs::think`/`num_predict`.
#[derive(Debug, Clone, Copy, Default)]
struct ChatOptions {
    think: Option<bool>,
    num_predict: Option<u32>,
}

pub fn run(args: &RunArgs) -> anyhow::Result<()> {
    // resolve_ollama_api, not resolve: `llmman run` is an /api/chat client
    // (see chat_submit/run_interactive_tty below), so a bare name must
    // resolve the same way it would if requested directly over the Ollama
    // API — otherwise a name resolved here, then handed to ensure_server as
    // a --model preload and to every /api/chat request this sends, is no
    // longer "bare" by the time ensure_model resolves it server-side (it
    // already has a "/" and a "."), so the docker.io/ai/ default never
    // fires and this silently falls back to hf.co/<name> instead.
    let model = crate::shortnames::resolve_ollama_api(&args.model);
    let prompt = args.prompt.join(" ");

    // Starts `llmman serve` detached, left running indefinitely, if one
    // isn't already reachable — the same shared helper pull/push/launch
    // use (see daemon::ensure_server's doc comment for why stdio is
    // redirected there: without it, this command would hang forever
    // waiting for the (never-exiting) daemon's inherited stdout/stderr
    // pipes to close). No preload model is passed: the resulting daemon
    // is a plain `llmman serve` with no model argument, so it's shared
    // cleanly across every future `run`/`pull`/`push`/`launch` in this
    // session rather than looking like it's dedicated to whatever model
    // happened to start it first. ensure_model_pulled below still makes
    // sure the model is on disk before the first /api/chat request.
    crate::daemon::ensure_server("")?;

    // Fail fast on a bad/unresolvable reference — mirrors ollama's
    // RunHandler, which resolves (Show, falling back to Pull) the model
    // before ever showing its interactive prompt. Without this, an error
    // like an invalid `hf.co/...` reference wouldn't surface until the
    // first message was submitted to /api/chat, well after the `> `
    // prompt had already been shown and read from.
    crate::daemon::ensure_model_pulled(&model)?;

    let interactive = prompt.is_empty() && io::stdin().is_terminal();

    if interactive {
        run_interactive_tty(&model, ChatOptions { think: args.think, num_predict: args.num_predict })
    } else {
        let p = if prompt.is_empty() {
            let mut s = String::new();
            io::stdin().read_line(&mut s)?;
            s.trim().to_string()
        } else {
            prompt
        };
        if !p.is_empty() {
            // A single-turn "conversation" over /api/chat — the same
            // endpoint, and the same chat_submit helper, interactive mode
            // uses below. Ollama's own CLI uses /api/generate for one-shot
            // prompts and /api/chat for its interactive REPL, but keeping
            // every mode in this file on one endpoint means there's a
            // single wire-format implementation to maintain here, and it
            // works identically against llmman or a real Ollama install
            // either way (both expose /api/chat).
            let client = chat_client()?;
            chat_submit(&client, &model, &mut Vec::new(), p, ChatOptions { think: args.think, num_predict: args.num_predict })?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Msg {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
}

/// Builds the `reqwest::blocking::Client` every chat-submitting caller in
/// this file shares. `reqwest::blocking::Client::new()` carries a 30s
/// default request timeout (unlike the async `Client`, which has none) —
/// fine for quick calls, but loading a model into `llama-server`/vllm for
/// the first time (or the daemon's own up-to-600s wait_for_ready health
/// poll, see cmd::serve::wait_for_ready) routinely takes longer than
/// that, so `Client::new()` here would abort an otherwise-succeeding
/// request with a misleading "operation timed out" long before the model
/// actually finished loading. Mirrors daemon::stream_progress's own
/// `.timeout(None)` for the same reason on the pull/push side.
fn chat_client() -> anyhow::Result<Client> {
    Client::builder().timeout(None).build().context("build http client")
}

#[derive(Serialize)]
struct ChatReq<'a> {
    model: &'a str,
    messages: &'a [Msg],
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<ChatReqOptions>,
}

#[derive(Serialize)]
struct ChatReqOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
}

#[derive(Deserialize)]
struct ChatChunk {
    #[serde(default)]
    message: Option<Msg>,
    #[serde(default)]
    done: bool,
}

// ---------------------------------------------------------------------------
// Interactive — TTY path
// ---------------------------------------------------------------------------

fn run_interactive_tty(model: &str, opts: ChatOptions) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        run_interactive_unix(model, opts)
    }
    #[cfg(not(unix))]
    {
        // Windows fallback: basic cooked-mode loop
        run_interactive_cooked(model, opts)
    }
}

// ---------------------------------------------------------------------------
// Interactive — Unix raw-mode readline (ported from ollama readline package)
// ---------------------------------------------------------------------------

#[cfg(unix)]
fn run_interactive_unix(model: &str, opts: ChatOptions) -> anyhow::Result<()> {
    use unix_readline::Readline;

    let client = chat_client()?;
    let mut messages: Vec<Msg> = Vec::new();
    let mut rl = Readline::new()?;
    let mut multiline: Option<String> = None; // Some while inside """
    // paste_sb accumulates lines while rl.pasting — mirrors ollama's `sb` +
    // `case scanner.Pasting: fmt.Fprintln(&sb, line); continue`
    let mut paste_sb = String::new();

    loop {
        let prompt = if multiline.is_some() {
            ". "
        } else if !paste_sb.is_empty() {
            ". " // AltPrompt shown while pasting, mirrors ollama
        } else {
            "> "
        };

        let line = match rl.readline(prompt) {
            Ok(Some(l)) => l,
            Ok(None) => break,
            Err(unix_readline::ReadlineError::Interrupted) => {
                multiline = None;
                paste_sb.clear();
                continue;
            }
        };

        // ── Bracketed paste accumulation ────────────────────────────────────
        // Mirrors `case scanner.Pasting: fmt.Fprintln(&sb, line); continue`
        // rl.pasting is true while between \x1b[200~ and \x1b[201~.
        // While pasting, ACCUMULATE into paste_sb WITHOUT submitting.
        // When pasting ends, the final line falls through to normal handling
        // with paste_sb prepended — same as ollama's `default: sb.WriteString`.
        if rl.pasting {
            paste_sb.push_str(&line);
            paste_sb.push('\n');
            continue;
        }

        // Not pasting: prepend any accumulated paste content to this line.
        // (ollama: `default: sb.WriteString(line)` then submit if sb.Len()>0)
        let line = if !paste_sb.is_empty() {
            let mut full = std::mem::take(&mut paste_sb);
            full.push_str(&line);
            full
        } else {
            line
        };

        // ── """ multiline mode ───────────────────────────────────────────────
        if let Some(ref mut buf) = multiline {
            if let Some(content) = line.strip_suffix("\"\"\"") {
                buf.push_str(content);
                let full = std::mem::take(buf).trim_end_matches('\n').to_string();
                multiline = None;
                if !full.trim().is_empty() {
                    chat_submit(&client, model, &mut messages, full, opts)?;
                }
            } else {
                buf.push_str(&line);
                buf.push('\n');
            }
            continue;
        }

        // ── Slash commands ───────────────────────────────────────────────────
        match line.trim() {
            "" => continue,
            "/bye" | "/exit" => break,
            "/clear" => {
                messages.clear();
                eprintln!("Conversation cleared.");
                continue;
            }
            s if s.starts_with('/') => {
                eprintln!("Commands: /bye  /clear  \"\"\" (multiline)");
                continue;
            }
            _ => {}
        }

        // ── Triple-quote multiline opener ────────────────────────────────────
        if line.trim_start().starts_with("\"\"\"") {
            let inner = line.trim_start().trim_start_matches("\"\"\"");
            if let Some(closed) = inner.strip_suffix("\"\"\"") {
                let content = closed.to_string();
                if !content.trim().is_empty() {
                    chat_submit(&client, model, &mut messages, content, opts)?;
                }
            } else {
                multiline = Some(inner.to_string() + "\n");
            }
            continue;
        }

        if !line.trim().is_empty() {
            chat_submit(&client, model, &mut messages, line, opts)?;
        }
    }

    Ok(())
}

/// Send one chat turn using the blocking reqwest client and stream the
/// response. Platform-agnostic (used by one-shot mode, the Unix
/// raw-mode REPL, and the Windows/non-TTY cooked-mode fallback below).
fn chat_submit(
    client: &reqwest::blocking::Client,
    model: &str,
    messages: &mut Vec<Msg>,
    content: String,
    opts: ChatOptions,
) -> anyhow::Result<()> {
    messages.push(Msg { role: "user".into(), content, thinking: None });

    let resp = client
        .post(&format!("{SERVER}/api/chat"))
        .json(&ChatReq {
            model,
            messages,
            stream: true,
            think: opts.think,
            options: opts.num_predict.map(|n| ChatReqOptions { num_predict: Some(n) }),
        })
        .send()
        .context("connect to llmman serve")?;

    if !resp.status().is_success() {
        let e = resp.text().unwrap_or_default();
        anyhow::bail!("{e}");
    }

    // Stream NDJSON lines from the response body.
    // reqwest::blocking::Response implements Read, so BufReader gives us lines
    // as they arrive — each line appears when the next token is generated.
    use std::io::BufRead;
    let mut full = String::new();
    let mut thinking_open = false;
    for line in std::io::BufReader::new(resp).lines() {
        let line = line?;
        if line.is_empty() { continue; }
        let Ok(chunk) = serde_json::from_str::<ChatChunk>(&line) else { continue };
        if let Some(ref msg) = chunk.message {
            if let Some(ref t) = msg.thinking {
                if !t.is_empty() {
                    if !thinking_open {
                        eprint!("Thinking: ");
                        thinking_open = true;
                    }
                    eprint!("{t}");
                }
            }
            if !msg.content.is_empty() && thinking_open {
                eprintln!();
                thinking_open = false;
            }
            if !msg.content.is_empty() {
                print!("{}", msg.content);
                io::stdout().flush().ok();
                full.push_str(&msg.content);
            }
        }
        if chunk.done { break; }
    }
    println!("\n");
    messages.push(Msg { role: "assistant".into(), content: full, thinking: None });
    Ok(())
}

// ---------------------------------------------------------------------------
// Unix raw-mode readline — direct port of ollama readline/readline.go
// ---------------------------------------------------------------------------

#[cfg(unix)]
mod unix_readline {
    use std::io::{BufRead, BufReader, Read, Stdin, Write};
    use std::os::unix::io::AsRawFd;

    // Character codes — identical to ollama readline/types.go
    const CHAR_INTERRUPT: u8 = 3;  // Ctrl-C
    const CHAR_EOF: u8 = 4;        // Ctrl-D
    const CHAR_CTRL_J: u8 = 10;    // \n  line feed / pasted newline
    const CHAR_ENTER: u8 = 13;     // \r  keyboard Enter
    const CHAR_ESC: u8 = 27;
    const CHAR_ESCAPE_EX: u8 = 91; // '[' — second byte of ESC[
    const CHAR_BACKSPACE: u8 = 127;

    pub enum ReadlineError {
        Interrupted,
    }

    // CharBracketedPaste = 50 ('2') — third byte of ESC[ sequence;
    // reading 3 more bytes gives "00~" (paste start) or "01~" (paste end).
    // Mirrors ollama readline/types.go: CharBracketedPaste/Start/End.
    const CHAR_BRACKETED_PASTE: u8 = 50;   // '2'
    const PASTE_START: &[u8; 3] = b"00~";
    const PASTE_END:   &[u8; 3] = b"01~";

    pub struct Readline {
        reader: BufReader<Stdin>,
        orig: libc::termios,
        fd: std::os::unix::io::RawFd,
        pub pasting: bool, // true while inside \x1b[200~...\x1b[201~
    }

    impl Readline {
        /// Enable raw mode + bracketed paste (mirrors ollama SetRawMode + StartBracketedPaste).
        pub fn new() -> anyhow::Result<Self> {
            let stdin = std::io::stdin();
            let fd = stdin.as_raw_fd();

            let orig = unsafe {
                let mut t: libc::termios = std::mem::zeroed();
                if libc::tcgetattr(fd, &mut t) < 0 {
                    anyhow::bail!("tcgetattr failed");
                }
                t
            };

            let mut raw = orig;
            unsafe {
                raw.c_iflag &= !(libc::IGNBRK | libc::BRKINT | libc::PARMRK
                    | libc::ISTRIP | libc::INLCR | libc::IGNCR
                    | libc::ICRNL  | libc::IXON);
                raw.c_lflag &= !(libc::ECHO | libc::ECHONL | libc::ICANON
                    | libc::ISIG | libc::IEXTEN);
                raw.c_cflag &= !(libc::CSIZE | libc::PARENB);
                raw.c_cflag |= libc::CS8;
                raw.c_cc[libc::VMIN as usize]  = 1;
                raw.c_cc[libc::VTIME as usize] = 0;
                if libc::tcsetattr(fd, libc::TCSANOW, &raw) < 0 {
                    anyhow::bail!("tcsetattr failed");
                }
            }

            // Enable bracketed paste mode — mirrors `fmt.Print(readline.StartBracketedPaste)`
            print!("\x1b[?2004h");
            std::io::stdout().flush().ok();

            Ok(Self { reader: BufReader::new(stdin), orig, fd, pasting: false })
        }

        /// Read one logical line from the terminal.
        ///
        /// Paste detection mirrors ollama readline/readline.go exactly:
        ///   - After each read, check reader.buffer() (≡ reader.Buffered() in Go)
        ///   - If non-empty → draining (we are consuming a paste)
        ///   - CharCtrlJ (\n) while draining → submit (same as Enter)
        ///   - CharCtrlJ while NOT draining → Ctrl-J multiline continuation
        ///   - CharEnter (\r) → always submit
        pub fn readline(&mut self, prompt: &str) -> Result<Option<String>, ReadlineError> {
            print!("{prompt}");
            std::io::stdout().flush().ok();

            let mut buf: Vec<u8> = Vec::new();
            let mut pasted_lines: Vec<String> = Vec::new();
            let mut draining = false;
            let mut stop_draining = false;
            let mut esc = false;
            let mut esc_ex = false;

            loop {
                // Apply deferred state from previous iteration (ollama lines 130-134)
                if stop_draining {
                    draining = false;
                    stop_draining = false;
                }

                // Read exactly one byte
                let mut b = [0u8; 1];
                match self.reader.read_exact(&mut b) {
                    Ok(_) => {}
                    Err(_) => return Ok(None),
                }
                let r = b[0];

                // Paste detection: mirrors `if i.Terminal.reader.Buffered() > 0`
                if !self.reader.buffer().is_empty() {
                    draining = true;
                } else if draining {
                    stop_draining = true;
                }

                // ESC sequence handling — mirrors ollama readline.go escex block.
                // Key addition: CharBracketedPaste ('2') reads 3 more bytes to
                // detect "00~" (paste start) or "01~" (paste end).
                if esc_ex {
                    esc_ex = false;
                    match r {
                        CHAR_BRACKETED_PASTE => {
                            // Read 3 more bytes: "00~" or "01~"
                            let mut code = [0u8; 3];
                            if self.reader.read_exact(&mut code).is_ok() {
                                if &code == PASTE_START {
                                    self.pasting = true;
                                } else if &code == PASTE_END {
                                    self.pasting = false;
                                }
                                // Update draining after reading extra bytes
                                if !self.reader.buffer().is_empty() {
                                    draining = true;
                                }
                            }
                        }
                        // Consume the '~' for delete/other 2-byte sequences
                        51 | 53 | 54 => {
                            let mut tilde = [0u8; 1];
                            let _ = self.reader.read_exact(&mut tilde);
                        }
                        _ => {} // arrow keys etc. — just skip
                    }
                    continue;
                } else if esc {
                    esc = false;
                    if r == CHAR_ESCAPE_EX { esc_ex = true; }
                    continue;
                }

                match r {
                    CHAR_INTERRUPT => {
                        pasted_lines.clear();
                        buf.clear();
                        println!();
                        return Err(ReadlineError::Interrupted);
                    }
                    CHAR_EOF => {
                        if buf.is_empty() && pasted_lines.is_empty() {
                            println!();
                            return Ok(None);
                        }
                    }
                    CHAR_ESC => { esc = true; }
                    CHAR_BACKSPACE => {
                        if !buf.is_empty() {
                            // Remove last complete UTF-8 codepoint
                            loop {
                                match buf.pop() {
                                    None => break,
                                    Some(b) if (b & 0xC0) != 0x80 => break, // lead byte
                                    Some(_) => {} // continuation byte, keep going
                                }
                            }
                            print!("\x08 \x08");
                            std::io::stdout().flush().ok();
                        } else if !pasted_lines.is_empty() {
                            let prev = pasted_lines.pop().unwrap();
                            print!("\r\x1b[K\x1b[A\r\x1b[K{prompt}{prev}");
                            std::io::stdout().flush().ok();
                            buf = prev.into_bytes();
                        }
                    }
                    CHAR_CTRL_J => {
                        // \n: pasted newline (draining) or Ctrl-J multiline (not draining)
                        // Mirrors ollama case CharCtrlJ
                        if !draining {
                            // Not draining → multiline continuation (Ctrl-J typed)
                            pasted_lines.push(String::from_utf8_lossy(&buf).to_string());
                            buf.clear();
                            println!();
                            print!(". ");
                            std::io::stdout().flush().ok();
                        } else {
                            // Draining → submit (pasted \n acts like Enter)
                            return Ok(Some(Self::assemble(&mut buf, &mut pasted_lines)));
                        }
                    }
                    CHAR_ENTER => {
                        // \r: keyboard Enter → always submit
                        return Ok(Some(Self::assemble(&mut buf, &mut pasted_lines)));
                    }
                    c => {
                        // Printable ASCII, tab, or UTF-8 bytes
                        if c >= 32 || c == 9 || c >= 0x80 {
                            buf.push(c);
                            let _ = std::io::stdout().write_all(&[c]);
                            std::io::stdout().flush().ok();
                        }
                    }
                }
            }
        }

        fn assemble(buf: &mut Vec<u8>, pasted_lines: &mut Vec<String>) -> String {
            let last = String::from_utf8_lossy(buf).to_string();
            buf.clear();
            println!();
            if pasted_lines.is_empty() {
                last
            } else {
                let prefix = pasted_lines.join("\n");
                pasted_lines.clear();
                format!("{prefix}\n{last}")
            }
        }
    }

    impl Drop for Readline {
        fn drop(&mut self) {
            // Disable bracketed paste, restore terminal — mirrors ollama's defer
            print!("\x1b[?2004l");
            std::io::stdout().flush().ok();
            unsafe { libc::tcsetattr(self.fd, libc::TCSANOW, &self.orig); }
        }
    }
}

// ---------------------------------------------------------------------------
// Windows / non-TTY fallback (cooked mode)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn run_interactive_cooked(model: &str, opts: ChatOptions) -> anyhow::Result<()> {
    let client = chat_client()?;
    let mut messages: Vec<Msg> = Vec::new();
    use std::io::BufRead;
    let stdin = std::io::stdin();
    let mut reader = std::io::BufReader::new(stdin.lock());

    loop {
        print!("> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 { break; }
        let line = line.trim_end_matches('\n').trim_end_matches('\r').to_string();
        match line.trim() {
            "" => continue,
            "/bye" | "/exit" => break,
            "/clear" => { messages.clear(); continue; }
            _ => {}
        }
        if !line.trim().is_empty() {
            chat_submit(&client, model, &mut messages, line, opts)?;
        }
    }
    Ok(())
}

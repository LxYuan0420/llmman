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
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncBufReadExt;
use tokio::time::{sleep, Duration, Instant};

const SERVER: &str = "http://127.0.0.1:17434";

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Args, Debug)]
pub struct RunArgs {
    #[arg(value_name = "MODEL")]
    pub model: String,
    #[arg(value_name = "PROMPT", trailing_var_arg = true, allow_hyphen_values = true)]
    pub prompt: Vec<String>,
}

pub fn run(args: &RunArgs) -> anyhow::Result<()> {
    // resolve_ollama_api, not resolve: `llmman run` is an /api/chat client
    // (see run_oneshot/run_interactive_tty below), so a bare name must
    // resolve the same way it would if requested directly over the Ollama
    // API — otherwise a name resolved here, then handed to ensure_server as
    // a --model preload and to every /api/chat request this sends, is no
    // longer "bare" by the time ensure_model resolves it server-side (it
    // already has a "/" and a "."), so the docker.io/ai/ default never
    // fires and this silently falls back to hf.co/<name> instead.
    let model = crate::shortnames::resolve_ollama_api(&args.model);
    let prompt = args.prompt.join(" ");

    // Ensure serve is running before anything else.
    let rt = tokio::runtime::Runtime::new()?;
    let async_client = Client::new();
    rt.block_on(ensure_server(&async_client, &model))?;

    let interactive = prompt.is_empty() && io::stdin().is_terminal();

    if interactive {
        run_interactive_tty(&model)
    } else {
        let p = if prompt.is_empty() {
            let mut s = String::new();
            io::stdin().read_line(&mut s)?;
            s.trim().to_string()
        } else {
            prompt
        };
        if !p.is_empty() {
            rt.block_on(run_oneshot(&async_client, &model, &p))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Server lifecycle
// ---------------------------------------------------------------------------

async fn server_alive(client: &Client) -> bool {
    client
        .get(SERVER)
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .is_ok()
}

async fn ensure_server(client: &Client, model: &str) -> anyhow::Result<()> {
    if server_alive(client).await {
        return Ok(());
    }
    let exe = std::env::current_exe().context("could not resolve own executable")?;
    eprintln!("[llmman] starting serve...");
    tokio::process::Command::new(&exe)
        .arg("serve")
        .arg(model)
        .kill_on_drop(false)
        .spawn()
        .context("spawn llmman serve")?;
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if Instant::now() > deadline {
            anyhow::bail!("llmman serve did not start within 60 s");
        }
        if server_alive(client).await {
            return Ok(());
        }
        sleep(Duration::from_millis(300)).await;
    }
}

// ---------------------------------------------------------------------------
// Wire types (shared between async one-shot and sync interactive)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Msg {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
}

#[derive(Serialize)]
struct ChatReq<'a> {
    model: &'a str,
    messages: &'a [Msg],
    stream: bool,
}

#[derive(Deserialize)]
struct ChatChunk {
    #[serde(default)]
    message: Option<Msg>,
    #[serde(default)]
    done: bool,
}

#[derive(Serialize)]
struct GenReq<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
}

#[derive(Deserialize)]
struct GenChunk {
    #[serde(default)]
    response: String,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    done: bool,
}

// ---------------------------------------------------------------------------
// One-shot (async streaming)
// ---------------------------------------------------------------------------

async fn run_oneshot(client: &Client, model: &str, prompt: &str) -> anyhow::Result<()> {
    let resp = client
        .post(&format!("{SERVER}/api/generate"))
        .json(&GenReq { model, prompt, stream: true })
        .send()
        .await
        .context("connect to llmman serve")?;
    if !resp.status().is_success() {
        anyhow::bail!("{}", resp.text().await.unwrap_or_default());
    }
    let mut thinking_open = false;
    stream_lines(resp, |line| {
        let Ok(chunk) = serde_json::from_str::<GenChunk>(line) else { return };
        if let Some(ref t) = chunk.thinking {
            if !t.is_empty() {
                if !thinking_open {
                    eprint!("Thinking: ");
                    thinking_open = true;
                }
                eprint!("{t}");
            }
        }
        if !chunk.response.is_empty() && thinking_open {
            eprintln!();
            thinking_open = false;
        }
        if !chunk.response.is_empty() {
            print!("{}", chunk.response);
            io::stdout().flush().ok();
        }
    })
    .await?;
    println!("\n");
    Ok(())
}

async fn stream_lines(
    resp: reqwest::Response,
    mut f: impl FnMut(&str),
) -> anyhow::Result<()> {
    use tokio_util::io::StreamReader;
    let body = resp
        .bytes_stream()
        .map(|r| r.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)));
    let reader = StreamReader::new(body);
    let mut lines = tokio::io::BufReader::new(reader).lines();
    while let Some(line) = lines.next_line().await? {
        f(&line);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Interactive — TTY path
// ---------------------------------------------------------------------------

fn run_interactive_tty(model: &str) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        run_interactive_unix(model)
    }
    #[cfg(not(unix))]
    {
        // Windows fallback: basic cooked-mode loop
        run_interactive_cooked(model)
    }
}

// ---------------------------------------------------------------------------
// Interactive — Unix raw-mode readline (ported from ollama readline package)
// ---------------------------------------------------------------------------

#[cfg(unix)]
fn run_interactive_unix(model: &str) -> anyhow::Result<()> {
    use unix_readline::Readline;

    let client = reqwest::blocking::Client::new();
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
                    chat_submit(&client, model, &mut messages, full)?;
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
                    chat_submit(&client, model, &mut messages, content)?;
                }
            } else {
                multiline = Some(inner.to_string() + "\n");
            }
            continue;
        }

        if !line.trim().is_empty() {
            chat_submit(&client, model, &mut messages, line)?;
        }
    }

    Ok(())
}

/// Send one chat turn using the blocking reqwest client and stream the response.
#[cfg(unix)]
fn chat_submit(
    client: &reqwest::blocking::Client,
    model: &str,
    messages: &mut Vec<Msg>,
    content: String,
) -> anyhow::Result<()> {
    messages.push(Msg { role: "user".into(), content, thinking: None });

    let resp = client
        .post(&format!("{SERVER}/api/chat"))
        .json(&ChatReq { model, messages, stream: true })
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
fn run_interactive_cooked(model: &str) -> anyhow::Result<()> {
    let client = reqwest::blocking::Client::new();
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
            #[cfg(unix)]
            chat_submit(&client, model, &mut messages, line)?;
            #[cfg(not(unix))]
            chat_submit_win(&client, model, &mut messages, line)?;
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn chat_submit_win(
    client: &reqwest::blocking::Client,
    model: &str,
    messages: &mut Vec<Msg>,
    content: String,
) -> anyhow::Result<()> {
    messages.push(Msg { role: "user".into(), content, thinking: None });
    let resp = client
        .post(&format!("{SERVER}/api/chat"))
        .json(&ChatReq { model, messages, stream: true })
        .send()?;
    use std::io::BufRead;
    let mut full = String::new();
    for line in std::io::BufReader::new(resp).lines() {
        let line = line?;
        if line.is_empty() { continue; }
        let Ok(chunk) = serde_json::from_str::<ChatChunk>(&line) else { continue };
        if let Some(ref msg) = chunk.message {
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

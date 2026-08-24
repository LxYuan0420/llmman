//! `llmman serve` – HTTP server exposing Ollama, OpenAI (including the
//! Responses API), and Anthropic-compatible APIs backed by `llama-server`
//! sub-processes from llama.cpp.

use std::collections::{HashMap, VecDeque};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex as StdMutex};

use anyhow::{anyhow, Context};
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use clap::Args;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration, Instant};

use crate::default_store;
use crate::modelpack::{resolve_model, ModelPath};
use crate::storage::OciStore;
use crate::webui;

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Model to pre-load immediately on startup (e.g. hf.co/unsloth/Qwen3.5-0.8B-GGUF:latest)
    #[arg(value_name = "MODEL")]
    pub model: Option<String>,

    /// Local store directory (overrides the default). Every client of this
    /// daemon — the CLI's `pull`/`push`/`run`/`list`/etc. and any Ollama-API
    /// HTTP client — shares whichever store the daemon was started with;
    /// there is no per-client override.
    #[arg(long, value_name = "DIR")]
    pub store: Option<PathBuf>,

    /// Run llama-server in a container (docker or podman) instead of as a
    /// local process — Linux only. Auto-selects the matching
    /// ghcr.io/ggml-org/llama.cpp:server-<backend> image for whatever GPU
    /// acceleration the host has (see crate::container); no local
    /// llama-server binary is required on PATH when this is set.
    #[arg(long, value_name = "docker|podman")]
    pub ociman: Option<crate::container::ContainerManager>,

    /// Pin the llama.cpp release used for this server, instead of always
    /// taking whatever is currently latest. With --ociman, this pins the
    /// ghcr.io/ggml-org/llama.cpp container image tag (e.g. `b9994`
    /// instead of the floating `server`/`server-cuda`/... tags — pick one
    /// that's actually published for every backend variant you might run;
    /// see docs/docker.md in ggml-org/llama.cpp). Without --ociman, this
    /// pins which GitHub release of llama.cpp's own prebuilt
    /// `llama-server` `llmman serve` downloads and caches (see
    /// crate::llama_release) — set this to force that managed download
    /// even when some other `llama-server` is already on PATH, which is
    /// otherwise preferred untouched.
    #[arg(long, value_name = "TAG")]
    pub llama_cpp_version: Option<String>,

    /// Proactively pull the ghcr.io/ggml-org/llama.cpp image `--ociman`
    /// would run, as its own explicit foreground step, then exit — this
    /// process does not go on to bind the listener or serve — with the
    /// pull's own progress (a real `docker pull`/`podman pull` progress
    /// bar) inherited directly to this process's stdout/stderr — only
    /// meaningful together with --ociman, ignored otherwise.
    ///
    /// `--ociman`'s underlying `docker run`/`podman run` pulls an image
    /// that isn't already cached on its own, but silently: `serve` is
    /// normally started detached (see daemon.rs), its stdio redirected to
    /// a log file, so a caller waiting on the first request that actually
    /// needs the container (the first real prompt) sees nothing happen
    /// for however long a multi-hundred-MB-to-GB image pull takes —
    /// indistinguishable from a hang. Run `llmman serve --ociman ...
    /// --pull-oci` first, in the foreground, to do that pull visibly and
    /// finish as soon as it completes; then start the real, detached
    /// `llmman serve --ociman ...` (without `--pull-oci`) separately.
    #[arg(long, requires = "ociman")]
    pub pull_oci: bool,

    /// Proactively download and cache the local `llama-server` binary
    /// `llmman serve` would otherwise fetch on first use (see
    /// crate::llama_release), as its own explicit foreground step, then
    /// exit — the non-container equivalent of --pull-oci: same rationale,
    /// same "run this first, in the foreground, then start the real
    /// `llmman serve` separately" pattern, just for the local-binary path
    /// instead of --ociman's container path. Backend selection (CPU,
    /// CUDA, ROCm, Vulkan, Metal) uses the same host detection
    /// (crate::hostgpu) as a normal `llmman serve` would, mirroring
    /// llama.cpp's own installer's CUDA > ROCm > Vulkan > CPU probing
    /// order. Not meaningful together with --ociman (that path never
    /// resolves a local binary at all).
    #[arg(long, conflicts_with_all = ["ociman", "pull_oci"])]
    pub pull_bin: bool,

    /// Request this many tokens of context (`--ctx-size`/`-c`) for every
    /// `llama-server` this daemon spawns, instead of leaving it unset and
    /// letting llama-server fall back to each model's own trained context
    /// length (`n_ctx_train` from its GGUF metadata). This is a ceiling,
    /// not a guarantee: llama-server caps the requested value back down
    /// to a model's own n_ctx_train whenever it's smaller, logging "the
    /// slot context (N) exceeds the training context of the model (M) -
    /// capping" and loading at M instead — the same clamp-and-warn
    /// behavior Ollama's own server (llm/server.go in ollama/ollama)
    /// independently implements for the same reason: serving positions
    /// beyond a model's trained length is unverified territory for a
    /// model's RoPE-based position embeddings and risks incoherent or
    /// NaN output, so neither this flag nor anything else in llmman
    /// tries to defeat that safety net (e.g. via llama-server's own
    /// `--override-kv`, which can rewrite the GGUF metadata llama-server
    /// checks against — deliberately not done here). This is a single
    /// value applied to every model this daemon loads (there is no
    /// per-model override): a model whose own trained context is larger
    /// than this gets capped down to this value; a model whose own
    /// trained context is smaller gets capped down to its own instead,
    /// per the paragraph above.
    #[arg(long, value_name = "N")]
    pub ctx_size: Option<u32>,
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState(Arc<Inner>);

struct Inner {
    manager: Mutex<ModelManager>,
    // None when --ociman is set: llama-server then runs in a container, so
    // no local binary is resolved (or required on PATH) at all. Behind a
    // mutex because the path resolved at startup can be deleted while this
    // daemon keeps running (an upgrade/uninstall of whatever install
    // provided it) — see local_llama_server_bin, which re-resolves and
    // stores a replacement in that case.
    llama_server_bin: StdMutex<Option<PathBuf>>,
    // This daemon's own executable path, canonicalized at startup (while
    // it still exists on disk). Reported by /api/version so clients — the
    // CLI's daemon::ensure_server, sbx — can detect a daemon left running
    // after the install that provided its binary was deleted, instead of
    // blindly reusing it.
    exe: Option<PathBuf>,
    ociman: Option<crate::container::ContainerManager>,
    llama_cpp_version: Option<String>,
    // See ServeArgs::ctx_size's doc comment — forwarded verbatim to every
    // spawn_llama_server/container::spawn call, local or containerized.
    ctx_size: Option<u32>,
    store_path: PathBuf,
    cache_path: PathBuf,
    client: Client,
}

struct ModelManager {
    running: HashMap<String, RunningModel>,
}

/// Everything `handle_ps` (and, transitively, `llmman ps`) needs to know
/// about a running model — see cmd::ps for the CLI side of this.
struct RunningModel {
    process: ModelProcess,
    port: u16,
    /// Full manifest digest (e.g. "sha256:abcd...") from the OCI store,
    /// captured at load time (see resolve_model's caller in ensure_model).
    digest: String,
    /// GGUF file size in bytes; 0 for a safetensors dir (vllm) — walking a
    /// multi-file safetensors directory isn't worth the cost just for
    /// `ps` output today.
    size: u64,
    started_at: String,
}

/// Which engine is actually serving requests for a [`RunningModel`] — surfaced
/// in `llmman ps`'s PROCESSOR column since, unlike Ollama's embedded
/// inference engine, llmman shells out to one of several different ones and
/// none of them report GPU/CPU memory split back to llmman, so there's no
/// equivalent of Ollama's "100% GPU"/"N%/N% CPU/GPU" figure to show here —
/// only which engine, and (for containers) which engine manager, is running.
impl RunningModel {
    fn processor(&self) -> String {
        match &self.process {
            ModelProcess::Local(Engine::LlamaServer, _, _) => "llama-server (local)".into(),
            ModelProcess::Local(Engine::Vllm, _, _) => "vllm (local)".into(),
            ModelProcess::Container(ociman, _) => {
                format!("llama-server (container/{})", ociman.binary())
            }
        }
    }

    fn pid(&self) -> Option<u32> {
        match &self.process {
            ModelProcess::Local(_, child, _) => child.id(),
            ModelProcess::Container(_, child) => child.id(),
        }
    }
}

/// Which local engine a [`ModelProcess::Local`] is running — see
/// [`RunningModel::processor`].
#[derive(Clone, Copy, Debug)]
enum Engine {
    LlamaServer,
    Vllm,
}

/// A running inference backend: either a local `llama-server`/`vllm`
/// process (killed via `Child::kill_on_drop`, except `Engine::Vllm` —
/// see this Drop impl) or an attached `docker run`/`podman run` process,
/// gracefully stopped via SIGTERM on drop since `kill_on_drop`'s SIGKILL
/// can't be forwarded to (and so doesn't stop) the container.
enum ModelProcess {
    // `Option<u32>` is the pid captured right after spawn, not
    // `child.id()` at drop time: `is_alive`'s `try_wait` reaps the child
    // once it exits, after which `child.id()` returns `None` — losing the
    // only pid needed to SIGKILL an `Engine::Vllm` group in Drop below.
    Local(Engine, tokio::process::Child, Option<u32>),
    Container(crate::container::ContainerManager, tokio::process::Child),
}

impl Drop for ModelProcess {
    fn drop(&mut self) {
        match self {
            ModelProcess::Container(_, child) => {
                if let Some(pid) = child.id() {
                    crate::container::stop(pid);
                }
            }
            // vllm forks its own API-server/engine-core workers, which
            // don't share a process tree `kill_on_drop`'s single-pid kill
            // can reach — SIGKILLing just the top pid (e.g. on a
            // cancelled load) orphans them, still holding GPU memory
            // indefinitely. spawn_vllm_server puts this child in its own
            // process group so the whole group can be killed here.
            #[cfg(unix)]
            ModelProcess::Local(Engine::Vllm, _, pid) => {
                if let Some(pid) = pid {
                    let result = unsafe { libc::kill(-(*pid as libc::pid_t), libc::SIGKILL) };
                    if result != 0 {
                        let err = std::io::Error::last_os_error();
                        eprintln!(
                            "[llmman] warning: SIGKILL to vllm process group {pid} failed: {err}"
                        );
                    }
                }
            }
            #[cfg(not(unix))]
            ModelProcess::Local(Engine::Vllm, _, _) => {}
            ModelProcess::Local(Engine::LlamaServer, _, _) => {}
        }
    }
}

impl ModelProcess {
    /// True if the underlying child process hasn't exited on its own since
    /// this model was marked running. Nothing else ever tells `mgr.running`
    /// about a process exiting unexpectedly (the only place that removes an
    /// entry today is the explicit Ollama unload signal in
    /// `handle_ollama_generate`) — a crash, an OOM kill, or anything else
    /// that takes `llama-server`/vllm down on its own would otherwise keep
    /// handing out that now-dead port forever, indistinguishable from a
    /// real live one until whichever caller's request to it fails with a
    /// bare connection error. `try_wait` is non-blocking either way: `Ok(None)`
    /// (still running) is the overwhelmingly common case this needs to stay
    /// cheap for.
    fn is_alive(&mut self) -> bool {
        let child = match self {
            ModelProcess::Local(_, child, _) => child,
            ModelProcess::Container(_, child) => child,
        };
        matches!(child.try_wait(), Ok(None))
    }
}

// ---------------------------------------------------------------------------
// Ollama API types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaChatRequest {
    model: String,
    #[serde(default)]
    messages: Vec<OllamaMessage>,
    #[serde(default = "bool_true")]
    stream: bool,
    options: Option<serde_json::Value>,
    /// Ollama's own top-level `think` field ("for thinking models, should
    /// the model think before responding? Can be a boolean or a thinking
    /// level"). See `think_to_chat_template_kwargs`.
    #[serde(default)]
    think: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct OllamaGenerateRequest {
    model: String,
    #[serde(default)]
    prompt: String,
    #[serde(default = "bool_true")]
    stream: bool,
    options: Option<serde_json::Value>,
    /// keep_alive: 0 with an empty prompt is the Ollama unload signal
    #[serde(default)]
    keep_alive: Option<serde_json::Value>,
    /// See `OllamaChatRequest::think`.
    #[serde(default)]
    think: Option<serde_json::Value>,
}

/// Translates Ollama's own `think` request field into the
/// `chat_template_kwargs` llama-server's `/v1/chat/completions` expects to
/// actually toggle a Qwen3-style template's thinking block (verified
/// directly against a running llama-server: a request-level
/// `"reasoning_budget"` field, despite mirroring the CLI flag of the same
/// name, is *not* respected — only `chat_template_kwargs.enable_thinking`
/// is).
///
/// Only handles the plain-boolean case of `think` (`true`/`false`) —
/// Ollama's documented string thinking levels (`"low"`/`"medium"`/
/// `"high"`/`"max"`) have no equivalent in `enable_thinking`'s plain
/// on/off, so those (along with `think` being absent entirely) are left
/// as a no-op: the template's own default applies, exactly as if this
/// field didn't exist.
fn think_to_chat_template_kwargs(think: &Option<serde_json::Value>) -> Option<serde_json::Value> {
    match think {
        Some(serde_json::Value::Bool(b)) => Some(serde_json::json!({ "enable_thinking": b })),
        _ => None,
    }
}

#[derive(Debug, Serialize)]
struct OllamaChatChunk {
    model: String,
    created_at: String,
    message: OllamaMessage,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    done_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct OllamaGenerateChunk {
    model: String,
    created_at: String,
    response: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    done_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModelInfo>,
}

#[derive(Debug, Serialize)]
struct OllamaModelInfo {
    name: String,
    model: String,
    size: u64,
    digest: String,
    modified_at: String,
    details: OllamaModelDetails,
}

#[derive(Debug, Serialize)]
struct OllamaModelDetails {
    format: String,
    family: String,
    parameter_size: String,
    quantization_level: String,
}

#[derive(Debug, Serialize)]
struct OllamaPsResponse {
    models: Vec<OllamaRunningModelInfo>,
}

#[derive(Debug, Serialize)]
struct OllamaRunningModelInfo {
    name: String,
    model: String,
    // Real Ollama /api/ps shape ends here (see api.ProcessModelResponse in
    // ollama/api/types.go); the fields below are llmman-specific additions
    // for `llmman ps` — safe for any other Ollama-API client to ignore.
    digest: String,
    size: u64,
    size_vram: u64,
    pid: Option<u32>,
    port: u16,
    processor: String,
    context_length: Option<u64>,
    started_at: String,
}

#[derive(Debug, Deserialize)]
struct OllamaShowRequest {
    model: String,
    name: Option<String>,
}

#[derive(Debug, Serialize)]
struct OllamaShowResponse {
    model_info: serde_json::Value,
    details: OllamaModelDetails,
}

#[derive(Debug, Deserialize)]
struct OllamaDeleteRequest {
    model: String,
    name: Option<String>,
}

// ---------------------------------------------------------------------------
// Anthropic API types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: Option<u32>,
    #[serde(default)]
    stream: bool,
    // Anthropic's real API accepts `system` as either a plain string or an
    // array of content blocks (the same shape as message content) — real
    // Claude Code always sends the array form, carrying its system prompt
    // as one or more {"type":"text","text":"..."} blocks, so a bare
    // Option<String> here 422s on every real request.
    system: Option<AnthropicContent>,
    temperature: Option<f32>,
    top_p: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicBlock>),
}

#[derive(Debug, Deserialize)]
struct AnthropicBlock {
    #[serde(rename = "type")]
    type_: String,
    text: Option<String>,
}

impl AnthropicContent {
    fn as_text(&self) -> String {
        match self {
            AnthropicContent::Text(s) => s.clone(),
            AnthropicContent::Blocks(blocks) => blocks
                .iter()
                .filter(|b| b.type_ == "text")
                .filter_map(|b| b.text.as_deref())
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAI types (internal proxy use)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, PartialEq, Eq)]
struct OAIMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct OAIChatRequest {
    model: String,
    messages: Vec<OAIMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    // llama-server's own default (1.0) disables repetition penalty
    // entirely — unlike Ollama, whose default is 1.1 (see
    // DEFAULT_REPEAT_PENALTY's doc comment). Every caller below sends
    // this explicitly rather than omitting it, so llmman's actual runtime
    // behavior matches Ollama's documented default instead of silently
    // falling back to llama-server's much more repetition-prone one.
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_penalty: Option<f32>,
    // See think_to_chat_template_kwargs. Omitted entirely (rather than
    // sent as `null`) when the caller didn't ask to override thinking, so
    // the template's own default applies exactly as if this field never
    // existed.
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<serde_json::Value>,
}

/// Ollama's documented default for `repeat_penalty` (see
/// docs/modelfile.mdx's PARAMETER table: "Default: 1.1"). llama-server's
/// own built-in default is 1.0 — repetition penalty fully disabled —
/// which measurably risks small/quantized models (observed firsthand with
/// qwen3.5:0.8b's "thinking" mode) looping on the same handful of
/// reasoning sentences indefinitely, since nothing then discourages the
/// sampler from repeating exact prior tokens. Used as the fallback
/// whenever a caller doesn't supply its own `options.repeat_penalty`.
const DEFAULT_REPEAT_PENALTY: f32 = 1.1;

#[derive(Debug, Deserialize)]
struct OAIChunk {
    choices: Vec<OAIChunkChoice>,
}

#[derive(Debug, Deserialize)]
struct OAIChunkChoice {
    delta: OAIChunkDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OAIChunkDelta {
    content: Option<String>,
    /// llama-server (Homebrew b8880) sends reasoning content in this field.
    /// The git repo uses "thinking" — accept both for forward compatibility.
    reasoning_content: Option<String>,
    thinking: Option<String>,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

struct AppError(anyhow::Error);

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(e: E) -> Self {
        Self(e.into())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({ "error": format!("{:#}", self.0) });
        (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bool_true() -> bool {
    true
}

fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn gen_id() -> String {
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{secs:032x}")
}

// ---------------------------------------------------------------------------
// Process management
// ---------------------------------------------------------------------------

fn find_free_port() -> anyhow::Result<u16> {
    let l = TcpListener::bind("127.0.0.1:0")?;
    Ok(l.local_addr()?.port())
}

/// Shared handle onto the last few lines a spawned inference backend wrote
/// to stdout/stderr — see `spawn_tail_relay`'s own doc comment for why
/// this exists and `wait_for_ready`'s use of it.
type OutputTail = Arc<StdMutex<VecDeque<String>>>;

/// How many trailing output lines `OutputTail` keeps — enough to catch a
/// one-or-two-line startup failure (a dynamic-linker error, "no such
/// file", an out-of-memory abort, ...) without holding onto an unbounded
/// amount of a chatty child's output.
const TAIL_LINES: usize = 20;

/// Relays a spawned child's piped stdout/stderr line-by-line to this
/// process's own stdout/stderr — preserving exactly what an inherited
/// (the previous default) stdio handle would have shown up as in
/// `llmman serve`'s own log (see daemon.rs's redirection of that to
/// serve.log) — while also appending each line to `tail` (bounded to the
/// last `TAIL_LINES`), so a caller that only learns of a crash after the
/// fact (see `wait_for_ready`) can still report *why*, instead of just
/// "the process exited" with the actual reason sitting only in a log file
/// the caller (an HTTP client, ultimately a chat UI) never sees.
fn spawn_tail_relay(
    reader: impl AsyncRead + Unpin + Send + 'static,
    tail: OutputTail,
    to_stderr: bool,
) {
    tokio::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if to_stderr {
                eprintln!("{line}");
            } else {
                println!("{line}");
            }
            if let Ok(mut buf) = tail.lock() {
                if buf.len() >= TAIL_LINES {
                    buf.pop_front();
                }
                buf.push_back(line);
            }
        }
    });
}

async fn spawn_llama_server(
    bin: &Path,
    model: &Path,
    port: u16,
    ctx_size: Option<u32>,
) -> anyhow::Result<(tokio::process::Child, OutputTail)> {
    let mut cmd = tokio::process::Command::new(bin);
    cmd.args([
        "--model",
        model.to_str().context("non-UTF-8 model path")?,
        "--port",
        &port.to_string(),
        "--host",
        "127.0.0.1",
    ]);
    // See ServeArgs::ctx_size's doc comment — unset leaves llama-server's
    // own default (each model's trained n_ctx_train) untouched. llama-server
    // itself caps this back down to n_ctx_train when it's smaller, with a
    // warning, the same way Ollama's own server does — this is intentional,
    // not a bug to work around (see ServeArgs::ctx_size).
    if let Some(n) = ctx_size {
        cmd.args(["--ctx-size", &n.to_string()]);
    }
    // See GPU_VISIBLE_DEVICE_VARS's own doc comment — already inherited
    // by default, forwarded explicitly here for clarity.
    for var in GPU_VISIBLE_DEVICE_VARS {
        if let Ok(val) = std::env::var(var) {
            cmd.env(var, val);
        }
    }
    // Piped (not inherited) so a startup crash's own explanation — e.g. a
    // dynamic linker's "error while loading shared libraries" — can be
    // captured into `tail` and surfaced by `wait_for_ready`, not just
    // dropped into a log file nobody making the request ever sees. See
    // `spawn_tail_relay`'s own doc comment for how this keeps showing up
    // in that log too.
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    let mut child = cmd
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("spawn llama-server from {}", bin.display()))?;

    let tail: OutputTail = Arc::new(StdMutex::new(VecDeque::with_capacity(TAIL_LINES)));
    if let Some(stdout) = child.stdout.take() {
        spawn_tail_relay(stdout, tail.clone(), false);
    }
    if let Some(stderr) = child.stderr.take() {
        spawn_tail_relay(stderr, tail.clone(), true);
    }
    Ok((child, tail))
}

async fn spawn_vllm_server(
    model_dir: &Path,
    port: u16,
    model_name: &str,
) -> anyhow::Result<tokio::process::Child> {
    let vllm = which_binary("vllm")?;
    let mut cmd = tokio::process::Command::new(&vllm);
    cmd.args([
        "serve",
        model_dir.to_str().context("non-UTF-8 model path")?,
        "--port",
        &port.to_string(),
        "--host",
        "127.0.0.1",
        // Register the model under the same name used in API requests so
        // {"model": "<ref>"} is accepted by vllm's OpenAI-compatible API.
        "--served-model-name",
        model_name,
    ]);
    // vllm's default --gpu-memory-utilization (0.9 of the *device's
    // total* memory) routinely exceeds what's actually free on a
    // unified-memory host or any box already running other GPU
    // workloads, so it refuses to start. Let a user work around it.
    if let Ok(extra) = std::env::var("LLMMAN_VLLM_ARGS") {
        cmd.args(extra.split_whitespace());
    }
    // Own process group so ModelProcess's Drop impl can kill vllm's whole
    // worker tree, not just this one pid, without also killing ourselves.
    #[cfg(unix)]
    cmd.process_group(0);
    cmd.kill_on_drop(true)
        .spawn()
        .with_context(|| format!("spawn vllm from {}", vllm.display()))
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        // On Windows the executable must carry the .exe suffix.
        #[cfg(windows)]
        let candidate = dir.join(format!("{name}.exe"));
        #[cfg(not(windows))]
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn which_binary(name: &str) -> anyhow::Result<PathBuf> {
    find_on_path(name).ok_or_else(|| anyhow::anyhow!("{name} not found on PATH"))
}

/// Polls `process`'s `/health` endpoint until it reports ready, bailing
/// out immediately — instead of only after the full 600s deadline below —
/// the moment `process` itself has already exited. Without this check, a
/// backend that crashes on startup (a missing shared library, a bad
/// model, an out-of-memory abort, ...) left `llmman launch`/any HTTP
/// client hanging for up to 10 minutes on a port nothing was ever going
/// to answer on again, with the real reason sitting only in `serve.log`
/// (see `ModelProcess::is_alive`'s doc comment on the same non-blocking
/// `try_wait` this reuses). `stderr_tail`, when given (currently only for
/// a local llama-server child — see `spawn_llama_server`), lets that
/// reason be included right in the error instead of just "the process
/// exited", so it reaches whatever's actually waiting on this (a chat UI
/// via the HTTP response), not only the log file.
async fn wait_for_ready(
    client: &Client,
    port: u16,
    process: &mut ModelProcess,
    stderr_tail: Option<&OutputTail>,
) -> anyhow::Result<()> {
    let url = format!("http://127.0.0.1:{port}/health");
    // vllm can take several minutes to load large models.
    let deadline = Instant::now() + Duration::from_secs(600);
    loop {
        if Instant::now() > deadline {
            return Err(anyhow!(
                "inference server on port {port} did not become ready within 600s"
            ));
        }
        if !process.is_alive() {
            let detail = stderr_tail.and_then(|t| {
                let lines = t.lock().ok()?;
                (!lines.is_empty()).then(|| lines.iter().cloned().collect::<Vec<_>>().join(" | "))
            });
            return Err(match detail {
                Some(detail) => anyhow!(
                    "inference server on port {port} exited before becoming ready: {detail}"
                ),
                None => anyhow!("inference server on port {port} exited before becoming ready"),
            });
        }
        if let Ok(resp) = client.get(&url).send().await {
            // llama-server: 200 + {"status":"ok"}   vllm: 200 + {}
            // Both return HTTP 200 only when fully ready.
            if resp.status().is_success() {
                return Ok(());
            }
        }
        sleep(Duration::from_millis(500)).await;
    }
}

/// Per-model registry of locks serializing every call into the Go shim's
/// `llmman_pull`/`llmman_push` (see `crate::ffi::pull`/`push`) for a given
/// model reference — replacing what used to be one `PULL_LOCK` mutex
/// shared by every model in the process.
///
/// go-shim/progress_state.go's `progressState` used to track only one
/// transfer at a time process-wide; it's now keyed per model reference
/// (see that file's own doc comment), so two *different* models pulling
/// or pushing at once no longer interleave or corrupt each other's
/// progress numbers the way they would have under the old global lock —
/// only concurrent operations on the *same* model reference still need to
/// be serialized. Three call sites can independently decide "not in
/// store, pull it" for the same model at once (this fallback in
/// `ensure_model`, `handle_pull`, and — since `launch` started calling
/// `daemon::ensure_model_pulled` itself — a concurrent client's own
/// explicit `/api/pull`), and without a per-model lock, two such calls
/// racing for the *same* model still means a redundant full download of
/// the same multi-GB blob. See also go-shim's `blobFetchGroup`
/// (shared_oci.go), which separately deduplicates two *different* models'
/// concurrent pulls that happen to share an underlying blob — a case this
/// per-model registry can't catch on its own since it only locks by
/// reference, not by content digest.
static MODEL_LOCKS: LazyLock<StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> =
    LazyLock::new(|| StdMutex::new(HashMap::new()));

/// Separate from `MODEL_LOCKS`: `ensure_model` holds a load lock across a
/// call that itself takes a `MODEL_LOCKS` lock (`pull_serialized`), so
/// sharing one map would re-enter the same non-reentrant mutex and deadlock.
static LOAD_LOCKS: LazyLock<StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> =
    LazyLock::new(|| StdMutex::new(HashMap::new()));

fn keyed_lock(
    registry: &StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    key: &str,
) -> Arc<tokio::sync::Mutex<()>> {
    let mut locks = registry.lock().unwrap();
    locks
        .entry(key.to_owned())
        .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
}

/// Removes `key` once nothing but `registry` itself still holds a clone —
/// call after dropping your own clone.
fn release_keyed_lock(
    registry: &StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    key: &str,
) {
    let mut locks = registry.lock().unwrap();
    if let Some(arc) = locks.get(key) {
        if Arc::strong_count(arc) <= 1 {
            locks.remove(key);
        }
    }
}

/// Returns (creating if absent) the lock serializing pull/push calls for
/// `model`. See `keyed_lock`.
fn model_lock(model: &str) -> Arc<tokio::sync::Mutex<()>> {
    keyed_lock(&MODEL_LOCKS, model)
}

/// See `release_keyed_lock`.
fn release_model_lock(model: &str) {
    release_keyed_lock(&MODEL_LOCKS, model)
}

/// Serializes `ensure_model`'s load phase (pull-if-missing, spawn,
/// wait-until-ready) per model, instead of `state.0.manager`.
fn load_lock(model: &str) -> Arc<tokio::sync::Mutex<()>> {
    keyed_lock(&LOAD_LOCKS, model)
}

/// See `release_keyed_lock`.
fn release_load_lock(model: &str) {
    release_keyed_lock(&LOAD_LOCKS, model)
}

/// RAII handle for `load_lock`: releases the mutex and the registry entry
/// in `Drop`, so cleanup still runs if the holding task is cancelled
/// (e.g. an axum request future dropped mid-`.await`) rather than only on
/// a normal return — code placed after an `.await` doesn't run when the
/// future holding it is dropped instead of polled to completion.
struct LoadLockGuard {
    model: String,
    guard: Option<tokio::sync::OwnedMutexGuard<()>>,
}

impl Drop for LoadLockGuard {
    fn drop(&mut self) {
        self.guard.take(); // drop the Mutex guard (and its Arc clone) first
        release_load_lock(&self.model);
    }
}

async fn acquire_load_lock(model: &str) -> LoadLockGuard {
    let guard = load_lock(model).lock_owned().await;
    LoadLockGuard {
        model: model.to_owned(),
        guard: Some(guard),
    }
}

/// Pulls `model` into `layout_dir` if (still, after acquiring model's own
/// lock) missing from the local store — shared by `ensure_model`'s
/// fallback and `handle_pull` so both funnel through the same
/// single-flight check instead of each deciding "not present" from a
/// snapshot taken before waiting on the lock, then redundantly re-pulling
/// once it's their turn.
///
/// Must be called from a blocking context (`spawn_blocking`): blocks the
/// current thread on model's lock, not just this async task.
fn pull_serialized(store_path: &std::path::Path, model: &str) -> anyhow::Result<()> {
    let lock = model_lock(model);
    let result = (|| {
        let _guard = lock.blocking_lock();
        if OciStore::open(store_path)
            .and_then(|s| s.find(model))
            .is_ok()
        {
            return Ok(()); // someone else already pulled it while we waited
        }
        let layout_dir = store_path
            .to_str()
            .ok_or_else(|| anyhow!("store path is not valid UTF-8"))?;
        crate::ffi::pull(model, layout_dir)
    })();
    drop(lock);
    release_model_lock(model);
    result
}

/// Resolve a user-supplied model ref to the canonical reference stored in
/// the OCI index (e.g. "hf.co/repo" → "hf.co/repo:latest"). No-ops before
/// the model is pulled — `ensure_model` also runs `default_tag` up front
/// to cover that gap.
fn canonical_ref(store_path: &std::path::Path, model_ref: &str) -> String {
    let Ok(store) = crate::storage::OciStore::open(store_path) else {
        return model_ref.to_owned();
    };
    let Ok(desc) = store.find(model_ref) else {
        return model_ref.to_owned();
    };
    desc.annotations
        .as_ref()
        .and_then(|a| a.get("org.opencontainers.image.ref.name"))
        .cloned()
        .unwrap_or_else(|| model_ref.to_owned())
}

/// Is `model_ref` already running and alive? See `ModelProcess::is_alive`.
async fn check_running(state: &AppState, model_ref: &str) -> Option<u16> {
    let mut mgr = state.0.manager.lock().await;
    if let Some(m) = mgr.running.get_mut(model_ref) {
        if m.process.is_alive() {
            return Some(m.port);
        }
        eprintln!(
            "[llmman] {model_ref} was marked running on port {} but its process has exited — reloading",
            m.port
        );
        mgr.running.remove(model_ref);
    }
    None
}

/// Ensures `model_ref` is loaded and returns `(canonical_ref, port)`. The
/// canonical name is what it's actually registered under with its backend
/// (`--served-model-name`), which can differ from a tagless `model_ref`
/// (e.g. `hf.co/owner/repo` canonicalizes to `...:latest`). Callers must
/// forward this canonical name, not their own input, as the "model" field
/// sent to the backend — vllm validates it strictly and 404s otherwise
/// (llama-server doesn't, so this went unnoticed for GGUF models).
async fn ensure_model(state: &AppState, model_ref: &str) -> Result<(String, u16), AppError> {
    let model_ref = crate::shortnames::resolve_ollama_api(model_ref);
    // Default the tag before the lock below: otherwise two concurrent
    // first-pulls of e.g. "gemma4" and "gemma4:latest" take different
    // locks and both spawn a process for the same model.
    let model_ref = crate::storage::default_tag(&model_ref);
    let model_ref = canonical_ref(&state.0.store_path, &model_ref);
    let model_ref = model_ref.as_str();

    if let Some(port) = check_running(state, model_ref).await {
        return Ok((model_ref.to_string(), port));
    }

    let _guard = acquire_load_lock(model_ref).await;

    // Someone else may have finished loading this model while we
    // waited for the lock above.
    if let Some(port) = check_running(state, model_ref).await {
        return Ok((model_ref.to_string(), port));
    }

    // If the model is not in the local store, pull it now.
    if crate::storage::OciStore::open(&state.0.store_path)
        .and_then(|s| s.find(model_ref))
        .is_err()
    {
        eprintln!("[llmman] {model_ref} not in store — pulling");
        let store_path = state.0.store_path.clone();
        let model_ref_owned = model_ref.to_owned();
        tokio::task::spawn_blocking(move || pull_serialized(&store_path, &model_ref_owned))
            .await
            .context("pull task panicked")?
            .context("pull failed")?;
    }

    // Re-canonicalise after the pull: default_tag already fixed the lock
    // key, so this only refines to a more specific stored form.
    let model_ref = canonical_ref(&state.0.store_path, model_ref);
    let model_ref = model_ref.as_str();

    // Re-check in case that stored form differs from the key above.
    if let Some(port) = check_running(state, model_ref).await {
        return Ok((model_ref.to_string(), port));
    }

    let model_path = resolve_model(&state.0.store_path, &state.0.cache_path, model_ref)
        .with_context(|| format!("resolve model {model_ref}"))?;
    // Best-effort — used only to populate `llmman ps`'s ID/SIZE columns;
    // resolve_model above already established the model exists, so a
    // failure here (e.g. a race with a concurrent `rm`) just means those
    // columns show as empty/zero rather than failing the whole request.
    let (digest, size) = OciStore::open(&state.0.store_path)
        .and_then(|s| {
            s.find(model_ref).map(|d| {
                let size = s.total_size(&d);
                (d.digest, size)
            })
        })
        .unwrap_or_default();
    let port = find_free_port()?;
    eprintln!("[llmman] loading {model_ref} on port {port}");
    // Only a local llama-server child gets a captured stderr tail today
    // (see spawn_llama_server) — container/vllm startup failures still
    // fail fast via ModelProcess::is_alive below, just without an inline
    // "here's why" (their own stdio is still inherited straight into
    // serve.log, same as before).
    let mut stderr_tail: Option<OutputTail> = None;
    let mut process = match (&model_path, state.0.ociman) {
        (ModelPath::Gguf(path), Some(ociman)) => ModelProcess::Container(
            ociman,
            crate::container::spawn(
                ociman,
                path,
                port,
                state.0.llama_cpp_version.as_deref(),
                state.0.ctx_size,
            )?,
        ),
        (ModelPath::Gguf(path), None) => {
            let bin = local_llama_server_bin(state).await?;
            let (child, tail) = spawn_llama_server(&bin, path, port, state.0.ctx_size).await?;
            stderr_tail = Some(tail);
            ModelProcess::Local(Engine::LlamaServer, child, None)
        }
        (ModelPath::SafeTensors(dir), _) => {
            let child = spawn_vllm_server(dir, port, model_ref).await?;
            let pid = child.id();
            ModelProcess::Local(Engine::Vllm, child, pid)
        }
    };
    wait_for_ready(&state.0.client, port, &mut process, stderr_tail.as_ref()).await?;
    eprintln!("[llmman] {model_ref} ready on port {port}");

    state.0.manager.lock().await.running.insert(
        model_ref.to_string(),
        RunningModel {
            process,
            port,
            digest,
            size,
            started_at: now_rfc3339(),
        },
    );
    Ok((model_ref.to_string(), port))
}

/// Returns the local llama-server binary to spawn: the one resolved at
/// startup, unless that file has since disappeared from disk (the install
/// that provided it was upgraded or removed while this daemon kept
/// running), in which case it is re-resolved from the current PATH (or
/// re-downloaded) and the replacement remembered for subsequent loads —
/// instead of failing every model load forever with a spawn error against
/// a path that no longer exists.
async fn local_llama_server_bin(state: &AppState) -> anyhow::Result<PathBuf> {
    let current = state
        .0
        .llama_server_bin
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let Some(bin) = current else {
        anyhow::bail!("no local llama-server binary resolved and --ociman was not set")
    };
    if bin.exists() {
        return Ok(bin);
    }
    eprintln!(
        "[llmman] llama-server at {} no longer exists; re-resolving",
        bin.display()
    );
    let pinned = state.0.llama_cpp_version.clone();
    let resolved = tokio::task::spawn_blocking(move || resolve_llama_server(pinned.as_deref()))
        .await
        .context("resolve llama-server task panicked")??;
    *state
        .0
        .llama_server_bin
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = Some(resolved.clone());
    Ok(resolved)
}

// ---------------------------------------------------------------------------
// Proxy helper – forward raw bytes to llama-server and stream back
// ---------------------------------------------------------------------------

async fn proxy(
    client: &Client,
    url: &str,
    headers: &HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    let mut req = client.post(url).body(body.to_vec());
    if let Some(ct) = headers.get("content-type") {
        req = req.header("content-type", ct);
    }
    let resp = req.send().await.context("proxy request to llama-server")?;
    let status = reqwest::StatusCode::from(resp.status());
    let resp_headers = resp.headers().clone();

    let stream = resp
        .bytes_stream()
        .map(|item| item.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>));

    let mut builder = Response::builder().status(status.as_u16());
    for (k, v) in &resp_headers {
        builder = builder.header(k, v);
    }
    Ok(builder.body(Body::from_stream(stream)).unwrap())
}

// ---------------------------------------------------------------------------
// collect_completion — like ollama's Completion() but in Rust.
//
// Sends a streaming request to llama-server's /v1/chat/completions
// (stream:true, same as ollama always uses), collects every byte until EOF,
// then parses all SSE lines in one pass.  This avoids both the non-streaming
// timeout problem (server must generate everything before sending a byte) and
// the async-streaming fragmentation problem (partial SSE lines across chunks).
// ---------------------------------------------------------------------------

async fn collect_completion(
    _shared_client: &Client,
    url: &str,
    oai: OAIChatRequest,
) -> Result<String, AppError> {
    // Use a fresh client per request.  The shared client's connection pool is
    // polluted by the many health-check GETs in wait_for_ready; reusing those
    // connections for the completion POST can silently produce an empty body
    // when llama-server has already closed the idle connection on its end.
    let client = reqwest::Client::new();

    let resp = post_chat(&client, url, &oai).await?;
    let raw = resp.bytes().await.context("read llama-server response")?;
    eprintln!("[llmman] llama-server raw {} bytes", raw.len());
    if raw.is_empty() {
        return Err(AppError(anyhow!(
            "inference backend returned empty response body"
        )));
    }

    let text = String::from_utf8_lossy(&raw);
    let mut content = String::new();
    for line in text.lines() {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        match oai_chunk_to_content(payload) {
            Some((tok, _thinking, true)) => {
                content.push_str(&tok);
                break;
            }
            Some((tok, _thinking, false)) => content.push_str(&tok),
            None => {}
        }
    }

    if content.is_empty() {
        // Log the raw response for diagnosis so the user can see what came back
        let preview: String = text.chars().take(400).collect();
        eprintln!("[llmman] WARNING: empty content extracted. Raw preview:\n{preview}");
    }
    Ok(content)
}

// ---------------------------------------------------------------------------
// SSE line buffering
//
// reqwest::bytes_stream() delivers raw TCP chunks; a single `data: {json}\n`
// SSE line can be split across two chunks.  bytes_to_lines buffers incomplete
// data and only yields complete newline-terminated lines, so downstream JSON
// parsing never sees a partial line.
// ---------------------------------------------------------------------------

fn bytes_to_lines(
    stream: impl futures::Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
) -> impl futures::Stream<Item = String> + Send + 'static {
    futures::stream::unfold(
        (stream.boxed(), String::new()),
        |(mut stream, mut buf)| async move {
            loop {
                if let Some(pos) = buf.find('\n') {
                    let line = buf[..pos].trim_end_matches('\r').to_string();
                    buf.drain(..=pos);
                    return Some((line, (stream, buf)));
                }
                match futures::StreamExt::next(&mut stream).await {
                    Some(Ok(chunk)) => buf.push_str(&String::from_utf8_lossy(&chunk)),
                    Some(Err(_)) | None => {
                        if buf.is_empty() {
                            return None;
                        }
                        let line = std::mem::take(&mut buf);
                        return Some((line, (stream, buf)));
                    }
                }
            }
        },
    )
}

// ---------------------------------------------------------------------------
// Shared SSE-chunk helper
// ---------------------------------------------------------------------------

/// Returns (content, thinking, done).
fn oai_chunk_to_content(payload: &str) -> Option<(String, Option<String>, bool)> {
    if payload == "[DONE]" {
        return Some((String::new(), None, true));
    }
    let chunk = serde_json::from_str::<OAIChunk>(payload).ok()?;
    let choice = chunk.choices.first()?;
    let content = choice.delta.content.as_deref().unwrap_or("").to_string();
    // Accept both field names: "reasoning_content" (Homebrew llama-server) and "thinking" (git)
    let thinking = choice
        .delta
        .reasoning_content
        .clone()
        .or_else(|| choice.delta.thinking.clone())
        .filter(|s| !s.is_empty());
    let done = choice
        .finish_reason
        .as_deref()
        .map(|r| !r.is_empty() && r != "null")
        .unwrap_or(false);
    Some((content, thinking, done))
}

// ---------------------------------------------------------------------------
// Shared "POST an OpenAI chat request, fail on non-2xx" helper
// ---------------------------------------------------------------------------

/// POSTs oai_req to url and returns the still-streaming response, converting
/// a non-2xx status into an AppError carrying the backend's error body.
/// Shared by every route that streams llama-server's OpenAI-style SSE output
/// back out in some other shape (stream_ollama, stream_anthropic below).
async fn post_chat(
    client: &Client,
    url: &str,
    oai_req: &OAIChatRequest,
) -> Result<reqwest::Response, AppError> {
    let resp = client
        .post(url)
        .json(oai_req)
        .send()
        .await
        .context("send to llama-server")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(AppError(anyhow!("inference backend {status}: {body}")));
    }
    Ok(resp)
}

// ---------------------------------------------------------------------------
// Streaming conversion: OpenAI SSE → Ollama NDJSON (chat + generate)
//
// The chat and generate endpoints differ only in which Ollama chunk struct
// wraps each token (OllamaChatChunk's nested `message.content` vs
// OllamaGenerateChunk's flat `response`), so both go through this one
// generic driver; build_chunk supplies just that piece.
// ---------------------------------------------------------------------------

async fn stream_ollama<T: Serialize + Send + 'static>(
    client: Client,
    url: String,
    oai_req: OAIChatRequest,
    build_chunk: impl Fn(String, Option<String>, bool) -> T + Send + 'static,
) -> Result<Response, AppError> {
    let resp = post_chat(&client, &url, &oai_req).await?;

    let stream = bytes_to_lines(resp.bytes_stream()).map(move |line| {
        let out = line
            .strip_prefix("data: ")
            .and_then(|p| oai_chunk_to_content(p))
            .map(|(content, thinking, done)| {
                let chunk = build_chunk(content, thinking, done);
                serde_json::to_string(&chunk).unwrap_or_default() + "\n"
            })
            .unwrap_or_default();
        Ok::<_, std::convert::Infallible>(Bytes::from(out))
    });

    Ok(Response::builder()
        .header("content-type", "application/x-ndjson")
        .body(Body::from_stream(stream))
        .unwrap())
}

// ---------------------------------------------------------------------------
// Streaming conversion: OpenAI SSE → Anthropic SSE
// ---------------------------------------------------------------------------

async fn stream_anthropic(
    client: Client,
    url: String,
    oai_req: OAIChatRequest,
    model: String,
) -> Result<Response, AppError> {
    let resp = post_chat(&client, &url, &oai_req).await?;

    let msg_id = gen_id();
    let preamble = {
        let start = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": null,
                "usage": { "input_tokens": 0, "output_tokens": 0 }
            }
        });
        let block_start = serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": { "type": "text", "text": "" }
        });
        format!(
            "event: message_start\ndata: {start}\n\nevent: content_block_start\ndata: {block_start}\n\n"
        )
    };

    let preamble_stream =
        futures::stream::once(futures::future::ready(Ok::<_, std::convert::Infallible>(
            Bytes::from(preamble),
        )));

    let sse_stream = bytes_to_lines(resp.bytes_stream()).map(move |line| {
        let out = if let Some(payload) = line.strip_prefix("data: ") {
            if payload == "[DONE]" {
                let msg_delta = serde_json::json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": "end_turn", "stop_sequence": null },
                    "usage": { "output_tokens": 0 }
                });
                let msg_stop = serde_json::json!({ "type": "message_stop" });
                format!(
                    "event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":0}}\n\n\
                     event: message_delta\ndata: {msg_delta}\n\n\
                     event: message_stop\ndata: {msg_stop}\n\n"
                )
            } else if let Ok(chunk) = serde_json::from_str::<OAIChunk>(payload) {
                let content = chunk.choices.first()
                    .and_then(|c| c.delta.content.as_deref())
                    .unwrap_or("")
                    .to_string();
                if content.is_empty() {
                    String::new()
                } else {
                    let delta = serde_json::json!({
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": { "type": "text_delta", "text": content }
                    });
                    format!("event: content_block_delta\ndata: {delta}\n\n")
                }
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        Ok::<_, std::convert::Infallible>(Bytes::from(out))
    });

    Ok(Response::builder()
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(preamble_stream.chain(sse_stream)))
        .unwrap())
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

fn gzipped(body: &'static [u8], content_type: &'static str) -> Response {
    Response::builder()
        .header("content-type", content_type)
        .header("content-encoding", "gzip")
        .header("cache-control", "public, max-age=3600")
        .body(Body::from(body))
        .unwrap()
}

async fn handle_root(headers: HeaderMap) -> impl IntoResponse {
    let wants_html = headers
        .get("accept")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.contains("text/html"))
        .unwrap_or(false);
    if wants_html {
        gzipped(webui::INDEX_HTML, "text/html; charset=utf-8").into_response()
    } else {
        "llmman is running".into_response()
    }
}

async fn handle_bundle_js() -> impl IntoResponse {
    gzipped(webui::BUNDLE_JS, "application/javascript; charset=utf-8")
}

async fn handle_bundle_css() -> impl IntoResponse {
    gzipped(webui::BUNDLE_CSS, "text/css; charset=utf-8")
}

async fn handle_loading_html() -> impl IntoResponse {
    gzipped(webui::LOADING_HTML, "text/html; charset=utf-8")
}

async fn handle_props() -> impl IntoResponse {
    // Return a minimal llama.cpp-compatible /props response in ROUTER mode.
    // The web UI uses `role` to detect multi-model (router) vs single-model mode.
    Json(serde_json::json!({
        "role": "router",
        "total_slots": 0,
        "model_path": "",
        "chat_template": "",
        "bos_token": "",
        "eos_token": "",
        "build_info": env!("LLMMAN_VERSION"),
        "modalities": { "vision": false, "audio": false },
        "default_generation_settings": {
            "id": 0,
            "id_task": 0,
            "n_ctx": 4096,
            "speculative": false,
            "is_processing": false,
            "params": {
                "n_predict": -1,
                "seed": 0,
                "temperature": 0.8,
                "dynatemp_range": 0.0,
                "dynatemp_exponent": 1.0,
                "top_k": 40,
                "top_p": 0.95,
                "min_p": 0.05,
                "top_n_sigma": 0.0,
                "xtc_probability": 0.0,
                "xtc_threshold": 0.1,
                "typ_p": 1.0,
                "repeat_last_n": 64,
                "repeat_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "dry_multiplier": 0.0,
                "dry_base": 1.75,
                "dry_allowed_length": 2,
                "dry_penalty_last_n": -1,
                "dry_sequence_breakers": [],
                "mirostat": 0,
                "mirostat_tau": 5.0,
                "mirostat_eta": 0.1,
                "stop": [],
                "max_tokens": -1,
                "n_keep": 0,
                "n_discard": 0,
                "ignore_eos": false,
                "stream": true,
                "logit_bias": [],
                "n_probs": 0,
                "min_keep": 0,
                "grammar": "",
                "grammar_lazy": false,
                "grammar_triggers": [],
                "preserved_tokens": [],
                "chat_format": "",
                "reasoning_format": "",
                "reasoning_in_content": false,
                "generation_prompt": "",
                "samplers": ["top_k", "top_p", "min_p", "temperature"],
                "backend_sampling": false,
                "speculative.n_max": 16,
                "speculative.n_min": 5,
                "speculative.p_min": 0.9,
                "timings_per_token": false,
                "post_sampling_probs": false,
                "lora": []
            },
            "prompt": "",
            "next_token": {
                "has_next_token": false,
                "has_new_line": false,
                "n_remain": 0,
                "n_decoded": 0,
                "stopping_word": ""
            }
        }
    }))
}

/// Ollama's GET /api/version, extended with this daemon's own identity —
/// executable path (canonicalized at startup) and pid — so a client can
/// tell whether a daemon it found listening still belongs to a live
/// install (the exe still exists, and is the binary the client would
/// launch) and stop/replace it if not. See daemon::ensure_server.
async fn handle_version(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "version": env!("LLMMAN_VERSION"),
        "exe": state.0.exe.as_ref().map(|p| p.to_string_lossy()),
        "pid": std::process::id(),
    }))
}

async fn handle_tags(State(state): State<AppState>) -> Result<impl IntoResponse, AppError> {
    let store = OciStore::open(&state.0.store_path)?;
    let list = store.list()?;
    let models = list
        .into_iter()
        .map(|img| OllamaModelInfo {
            name: img.reference.clone(),
            model: img.reference,
            size: img.size,
            digest: img.digest,
            modified_at: img
                .modified_at
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .and_then(|d| chrono::DateTime::from_timestamp(d.as_secs() as i64, 0))
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_else(now_rfc3339),
            details: OllamaModelDetails {
                format: "gguf".into(),
                family: String::new(),
                parameter_size: String::new(),
                quantization_level: String::new(),
            },
        })
        .collect();
    Ok(Json(OllamaTagsResponse { models }))
}

/// The subset of a [`RunningModel`] `handle_ps` needs, cloned out while
/// holding `manager`'s lock (see `handle_ps`) so the per-model `/props`
/// round trips afterward don't hold that lock for the duration.
struct PsEntry {
    name: String,
    digest: String,
    size: u64,
    port: u16,
    pid: Option<u32>,
    processor: String,
    started_at: String,
}

async fn handle_ps(State(state): State<AppState>) -> impl IntoResponse {
    let entries: Vec<PsEntry> = {
        let mgr = state.0.manager.lock().await;
        mgr.running
            .iter()
            .map(|(name, m)| PsEntry {
                name: name.clone(),
                digest: m.digest.clone(),
                size: m.size,
                port: m.port,
                pid: m.pid(),
                processor: m.processor(),
                started_at: m.started_at.clone(),
            })
            .collect()
    };

    let mut models = Vec::with_capacity(entries.len());
    for entry in entries {
        let context_length = query_context_length(&state.0.client, entry.port).await;
        models.push(OllamaRunningModelInfo {
            name: entry.name.clone(),
            model: entry.name,
            digest: entry.digest,
            size: entry.size,
            size_vram: 0, // not tracked — see RunningModel::processor's doc comment
            pid: entry.pid,
            port: entry.port,
            processor: entry.processor,
            context_length,
            started_at: entry.started_at,
        });
    }
    Json(OllamaPsResponse { models })
}

/// Best-effort live context-length lookup via the running llama-server's own
/// `/props` endpoint (`default_generation_settings.n_ctx`) — mirrors
/// Ollama's own preference for live runner data over anything cached (see
/// server.PsHandler's use of `v.llama.ContextLength()`). Returns `None` on
/// any failure (short timeout, connection error, unexpected shape, or a
/// vllm-backed model, which doesn't expose this endpoint at all) rather
/// than failing the whole `ps` response over one unreachable model.
async fn query_context_length(client: &Client, port: u16) -> Option<u64> {
    let url = format!("http://127.0.0.1:{port}/props");
    let resp = client
        .get(&url)
        .timeout(Duration::from_millis(500))
        .send()
        .await
        .ok()?;
    let body: serde_json::Value = resp.json().await.ok()?;
    body.get("default_generation_settings")?
        .get("n_ctx")?
        .as_u64()
}

async fn handle_show(
    State(state): State<AppState>,
    Json(req): Json<OllamaShowRequest>,
) -> Result<impl IntoResponse, AppError> {
    // ollama sends either {"name":"..."} or {"model":"..."} depending on call site;
    // filter out empty strings so we always fall back to whichever field is populated.
    let model_ref = req
        .name
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(&req.model);
    // Resolve the same way handle_pull stored it — otherwise a bare name
    // (e.g. "gemma4", pulled and stored as "docker.io/ai/gemma4") would
    // never be found by show/delete even though it's in the local store.
    let model_ref = crate::shortnames::resolve_ollama_api(model_ref);
    let model_ref = model_ref.as_str();
    eprintln!("[llmman] /api/show model={model_ref:?}");
    let store = OciStore::open(&state.0.store_path)?;
    let desc = store
        .find(model_ref)
        .map_err(|_| AppError(anyhow!("model not found: {model_ref}")))?;
    Ok(Json(OllamaShowResponse {
        model_info: serde_json::json!({ "digest": desc.digest, "size": desc.size }),
        details: OllamaModelDetails {
            format: "gguf".into(),
            family: String::new(),
            parameter_size: String::new(),
            quantization_level: String::new(),
        },
    }))
}

// -- Ollama /api/pull ---------------------------------------------------------
// Mirrors `ollama.PullHandler`: streams newline-delimited JSON status objects
// (`{"status": "..."}`, matching api.ProgressResponse) ending in either
// `{"status": "success"}` or `{"error": "..."}`. Real Ollama also reports
// per-layer `digest`/`total`/`completed` fields for a byte-level progress
// bar; the Go shim's `llmman_pull` is a single opaque blocking call with no
// progress callback, so this reports coarse status only — every field is
// `omitempty` on the client side, so callers that only render `status` (as
// `llmman pull`'s own CLI progress text does) see accurate text throughout.

#[derive(Debug, Deserialize)]
struct OllamaPullRequest {
    #[serde(default)]
    model: String,
    // Real Ollama keeps `Name` as a deprecated fallback for `Model`
    // (server/routes.go's `cmp.Or(req.Model, req.Name)`) — some clients
    // only ever send `name`, which used to 422 outright since `model`
    // was required. Falls back below like handle_show/handle_delete
    // already do.
    #[serde(default)]
    name: String,
}

async fn handle_pull(
    State(state): State<AppState>,
    Json(req): Json<OllamaPullRequest>,
) -> impl IntoResponse {
    let model_ref = if req.model.is_empty() {
        req.name.as_str()
    } else {
        req.model.as_str()
    };
    if model_ref.is_empty() {
        let body = serde_json::json!({"error": "model is required"});
        return (StatusCode::BAD_REQUEST, Json(body)).into_response();
    }
    let model = crate::shortnames::resolve_ollama_api(model_ref);
    eprintln!("[llmman] /api/pull model={model:?}");
    let store_path = state.0.store_path.clone();

    let already_present = OciStore::open(&store_path)
        .and_then(|s| s.find(&model))
        .is_ok();
    if already_present {
        let line = serde_json::json!({"status": "success"}).to_string() + "\n";
        return Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/x-ndjson")
            .body(Body::from(line))
            .unwrap();
    }

    // Not in the local store: actually pull it (the previous behavior only
    // ever 404'd here, so no real Ollama client's "pull if missing, then
    // use" flow — e.g. `ollama run <model>` — ever worked against llmman).
    //
    // pull_serialized (not a bare crate::ffi::pull call) re-checks presence
    // after acquiring PULL_LOCK: this request's own `already_present` check
    // above ran before that wait, so a concurrent pull of the same model
    // (from another client, or from ensure_model's own fallback below) can
    // finish while this one was waiting its turn — see PULL_LOCK's doc
    // comment for why two callers must never invoke the actual FFI pull at
    // the same time.
    let model_for_task = model.clone();
    let pull_task =
        tokio::task::spawn_blocking(move || pull_serialized(&store_path, &model_for_task));

    stream_ffi_progress(model, "pull", "pulling manifest", pull_task)
}

// -- Ollama /api/push ---------------------------------------------------------
// Ollama's own /api/push has no equivalent in llmman's original design (the
// route didn't exist at all before), but it's the same shape as /api/pull —
// a streamed NDJSON status sequence — so `llmman push` becoming a thin
// client of this endpoint (like `llmman pull`) gets both operations onto
// the exact same Ollama-protocol wire format.

#[derive(Debug, Deserialize)]
struct OllamaPushRequest {
    #[serde(default)]
    model: String,
    // See OllamaPullRequest's `name` field doc comment: same deprecated
    // `Name`-falls-back-to-`Model` shape as real Ollama's PushRequest.
    #[serde(default)]
    name: String,
}

async fn handle_push(
    State(state): State<AppState>,
    Json(req): Json<OllamaPushRequest>,
) -> impl IntoResponse {
    let model_ref = if req.model.is_empty() {
        req.name.as_str()
    } else {
        req.model.as_str()
    };
    if model_ref.is_empty() {
        let body = serde_json::json!({"error": "model is required"});
        return (StatusCode::BAD_REQUEST, Json(body)).into_response();
    }
    let model = crate::shortnames::resolve_ollama_api(model_ref);
    eprintln!("[llmman] /api/push model={model:?}");
    let store_path = state.0.store_path.clone();

    // Unlike pull, there's nothing sensible to do if the model isn't
    // already in the local store — push has no "fetch it first" fallback.
    if OciStore::open(&store_path)
        .and_then(|s| s.find(&model))
        .is_err()
    {
        let body = serde_json::json!({"error": format!("model not found: {model}")});
        return (StatusCode::NOT_FOUND, Json(body)).into_response();
    }

    // See MODEL_LOCKS' doc comment: a push shares the same Go-side
    // progressState entry (keyed by this model reference) as a pull of
    // the same model, so they need the same per-model mutual exclusion —
    // but a push of one model no longer blocks a pull/push of another.
    let model_for_task = model.clone();
    let push_task = tokio::task::spawn_blocking(move || {
        let lock = model_lock(&model_for_task);
        let result = (|| {
            let _guard = lock.blocking_lock();
            let layout_dir = store_path
                .to_str()
                .ok_or_else(|| anyhow!("store path is not valid UTF-8"))?;
            crate::ffi::push(layout_dir, &model_for_task)
        })();
        drop(lock);
        release_model_lock(&model_for_task);
        result
    });

    stream_ffi_progress(model, "push", "retrieving manifest", push_task).into_response()
}

/// Runs `task` (a blocking FFI call already dispatched via spawn_blocking)
/// to completion, streaming an immediate `first_status` line, then polling
/// `ffi::progress(&model)` every 200ms (matching the Go shim's own mpb
/// refresh rate) until the task finishes, then a final `{"status": "success"}` or
/// `{"error": ...}` line. Shared by handle_pull and handle_push.
///
/// Each polled line includes real `total`/`completed` byte counts (mirroring
/// Ollama's own api.ProgressResponse fields) once the shim's shared
/// `progressState` (go-shim/progress_state.go) has learned a nonzero total
/// — before that, or if the FFI call is a kind that doesn't track
/// byte-level progress at all, only `status` text is included, exactly
/// like the old heartbeat-only version of this function. This is what
/// lets `llmman pull`/`llmman push` render a real progress bar instead of
/// just printing status text: the Go shim's own mpb bars
/// (go-shim/shared_oci.go) already draw real bars for these exact
/// numbers, but only reach an interactive terminal when the FFI call runs
/// in the foreground CLI process (e.g. `llmman transfer`) — here it runs
/// inside the daemon, whose stdio is redirected to a log file (see
/// daemon::ensure_server), so polling and relaying over this NDJSON
/// stream is the only way those numbers reach `llmman pull`/`llmman push`.
fn stream_ffi_progress(
    model: String,
    verb: &'static str,
    first_status: &'static str,
    task: tokio::task::JoinHandle<anyhow::Result<()>>,
) -> Response {
    let first_line = serde_json::json!({"status": first_status}).to_string() + "\n";
    let stream = futures::stream::once(futures::future::ready(Bytes::from(first_line)))
        .chain(futures::stream::unfold(Some(task), move |task| {
            let model = model.clone();
            async move {
                let mut task = task?;
                tokio::select! {
                    result = &mut task => {
                        let line = match result {
                            Ok(Ok(())) => serde_json::json!({"status": "success"}).to_string(),
                            Ok(Err(e)) => serde_json::json!({"error": format!("{e:#}")}).to_string(),
                            Err(e) => serde_json::json!({"error": format!("{verb} task panicked: {e}")}).to_string(),
                        };
                        Some((Bytes::from(line + "\n"), None))
                    }
                    _ = sleep(Duration::from_millis(200)) => {
                        let line = match crate::ffi::progress(&model) {
                            Ok(p) if p.total > 0 => serde_json::json!({
                                "status": if p.status.is_empty() { format!("{verb}ing {model}") } else { p.status },
                                "total": p.total.max(0),
                                "completed": p.completed.clamp(0, p.total),
                            }),
                            Ok(p) if !p.status.is_empty() => serde_json::json!({"status": p.status}),
                            _ => serde_json::json!({"status": format!("{verb}ing {model}")}),
                        };
                        Some((Bytes::from(line.to_string() + "\n"), Some(task)))
                    }
                }
            }
        }))
        .map(Ok::<_, std::convert::Infallible>);

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/x-ndjson")
        .body(Body::from_stream(stream))
        .unwrap()
}

async fn handle_delete(
    State(state): State<AppState>,
    Json(req): Json<OllamaDeleteRequest>,
) -> Result<impl IntoResponse, AppError> {
    let model_ref = req
        .name
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(&req.model);
    // See handle_show: resolve the same way handle_pull stored it.
    let model_ref = crate::shortnames::resolve_ollama_api(model_ref);
    let store = OciStore::open(&state.0.store_path)?;
    store.remove(&model_ref)?;
    Ok(StatusCode::OK)
}

// -- Ollama /api/chat ---------------------------------------------------------

async fn handle_ollama_chat(
    State(state): State<AppState>,
    Json(req): Json<OllamaChatRequest>,
) -> Result<Response, AppError> {
    eprintln!(
        "[llmman] /api/chat model={:?} messages={}",
        req.model,
        req.messages.len()
    );
    let (model, port) = ensure_model(&state, &req.model).await?;
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let oai = OAIChatRequest {
        model: model.clone(),
        messages: req
            .messages
            .iter()
            .map(|m| OAIMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect(),
        stream: true,
        temperature: opt_f64(&req.options, "temperature"),
        top_p: opt_f64(&req.options, "top_p"),
        max_tokens: opt_u32(&req.options, "num_predict"),
        repeat_penalty: opt_f64(&req.options, "repeat_penalty").or(Some(DEFAULT_REPEAT_PENALTY)),
        chat_template_kwargs: think_to_chat_template_kwargs(&req.think),
    };
    stream_ollama(
        state.0.client.clone(),
        url,
        oai,
        move |content, thinking, done| OllamaChatChunk {
            model: model.clone(),
            created_at: now_rfc3339(),
            message: OllamaMessage {
                role: "assistant".into(),
                content,
                thinking,
            },
            done,
            done_reason: done.then_some("stop".into()),
        },
    )
    .await
}

// -- Ollama /api/generate -----------------------------------------------------

async fn handle_ollama_generate(
    State(state): State<AppState>,
    Json(req): Json<OllamaGenerateRequest>,
) -> Result<Response, AppError> {
    eprintln!(
        "[llmman] /api/generate model={:?} prompt_len={}",
        req.model,
        req.prompt.len()
    );

    // Empty prompt + keep_alive:0 = unload request (ollama server/routes.go:354).
    let is_unload = req.prompt.is_empty()
        && req
            .keep_alive
            .as_ref()
            .and_then(|v| v.as_i64())
            .map(|n| n == 0)
            .unwrap_or(false);
    if is_unload {
        let resolved = crate::shortnames::resolve_ollama_api(&req.model);
        let canonical = canonical_ref(&state.0.store_path, &resolved);
        // Wait for an in-flight load of this model to publish itself first,
        // so it can't race ahead of this remove.
        let _guard = acquire_load_lock(&canonical).await;
        state.0.manager.lock().await.running.remove(&canonical);
        return Ok(Json(OllamaGenerateChunk {
            model: req.model,
            created_at: now_rfc3339(),
            response: String::new(),
            thinking: None,
            done: true,
            done_reason: Some("unload".into()),
        })
        .into_response());
    }

    let (model, port) = ensure_model(&state, &req.model).await?;
    // Empty prompt = load-only request (mirrors ollama server/routes.go:429).
    if req.prompt.is_empty() {
        return Ok(Json(OllamaGenerateChunk {
            model: req.model,
            created_at: now_rfc3339(),
            response: String::new(),
            thinking: None,
            done: true,
            done_reason: Some("load".into()),
        })
        .into_response());
    }

    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let oai = OAIChatRequest {
        model: model.clone(),
        messages: vec![OAIMessage {
            role: "user".into(),
            content: req.prompt.clone(),
        }],
        stream: true,
        temperature: opt_f64(&req.options, "temperature"),
        top_p: opt_f64(&req.options, "top_p"),
        max_tokens: opt_u32(&req.options, "num_predict"),
        repeat_penalty: opt_f64(&req.options, "repeat_penalty").or(Some(DEFAULT_REPEAT_PENALTY)),
        chat_template_kwargs: think_to_chat_template_kwargs(&req.think),
    };
    stream_ollama(
        state.0.client.clone(),
        url,
        oai,
        move |response, thinking, done| OllamaGenerateChunk {
            model: model.clone(),
            created_at: now_rfc3339(),
            response,
            thinking,
            done,
            done_reason: done.then_some("stop".into()),
        },
    )
    .await
}

// -- OpenAI pass-through handlers --------------------------------------------

async fn handle_openai_models(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, AppError> {
    let store = OciStore::open(&state.0.store_path)?;
    let list = store.list()?;
    let mgr = state.0.manager.lock().await;
    let data: Vec<serde_json::Value> = list
        .into_iter()
        .map(|img| {
            let loaded = mgr.running.contains_key(&img.reference);
            serde_json::json!({
                "id": img.reference,
                "object": "model",
                "created": 0,
                "owned_by": "llmman",
                // status field consumed by the web UI to track loaded/unloaded state
                "status": { "value": if loaded { "loaded" } else { "unloaded" } },
            })
        })
        .collect();
    Ok(Json(serde_json::json!({ "object": "list", "data": data })))
}

/// Shared body of every plain OpenAI-passthrough route: parse just enough
/// of the request to find `model`, make sure it's loaded, rewrite `model`
/// to its canonical name (see `ensure_model`), then proxy through to the
/// backend's equivalent endpoint. `llama_path` is the only thing that
/// differs between handle_openai_chat/completions/embeddings below.
async fn proxy_openai(
    state: &AppState,
    headers: &HeaderMap,
    body: Bytes,
    llama_path: &str,
) -> Result<Response, AppError> {
    let mut req: serde_json::Value =
        serde_json::from_slice(&body).context("parse OpenAI request body")?;
    let model = req["model"].as_str().unwrap_or("").to_string();
    let (model, port) = ensure_model(state, &model).await?;
    req["model"] = serde_json::Value::String(model);
    let body = Bytes::from(serde_json::to_vec(&req).context("re-serialize OpenAI request body")?);
    let url = format!("http://127.0.0.1:{port}{llama_path}");
    proxy(&state.0.client, &url, headers, body).await
}

async fn handle_openai_chat(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    proxy_openai(&state, &headers, body, "/v1/chat/completions").await
}

async fn handle_openai_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    proxy_openai(&state, &headers, body, "/v1/completions").await
}

async fn handle_openai_embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    proxy_openai(&state, &headers, body, "/v1/embeddings").await
}

// -- OpenAI Responses API (/v1/responses) ------------------------------------
//
// llama-server (llama.cpp) has its own native /v1/responses implementation
// that converts a Responses-API request into a Chat Completions request
// internally (see server_chat_convert_responses_to_chatcmpl in
// tools/server/server-chat.cpp) — including the exact SSE event sequence
// Codex requires (response.created -> response.output_item.added ->
// response.output_text.delta -> ... -> response.completed, no `[DONE]`) and
// re-mapping of tool_calls into function_call output items. Re-implementing
// that translation here would just duplicate — and risk drifting out of
// sync with — llama.cpp's own logic, so this is a plain pass-through
// exactly like the other /v1/* routes above, apart from
// filter_non_function_tools (see its own doc comment) below.
async fn handle_openai_responses(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    let body = sanitize_responses_request(body)?;
    proxy_openai(&state, &headers, body, "/v1/responses").await
}

async fn handle_openai_responses_input_tokens(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    let body = sanitize_responses_request(body)?;
    proxy_openai(&state, &headers, body, "/v1/responses/input_tokens").await
}

/// Applies both `/v1/responses` request-shape workarounds below and
/// re-serializes once, rather than parsing the body twice.
fn sanitize_responses_request(body: Bytes) -> anyhow::Result<Bytes> {
    let mut req: serde_json::Value =
        serde_json::from_slice(&body).context("parse OpenAI request body")?;
    filter_non_function_tools(&mut req);
    consolidate_responses_instructions(&mut req);
    Ok(Bytes::from(
        serde_json::to_vec(&req).context("re-serialize sanitized request")?,
    ))
}

/// Strips any entry from the request's top-level `tools` array whose
/// `"type"` isn't `"function"` before proxying to llama-server.
///
/// Real Codex always includes Responses-API tool types llama-server's own
/// `/v1/responses` doesn't understand — a `"namespace"`-typed sub-agent
/// tool bundle, the bare `{"type":"web_search"}` entry, etc. — and, unlike
/// this module's other passthrough routes, llama-server hard-rejects the
/// *entire* request the moment even one such entry is present ("'type' of
/// tool must be 'function'"), rather than skipping just that entry. Since
/// Codex's own default toolset always includes at least one of these,
/// every real `codex`/`codex exec` invocation would 400 on its very first
/// turn without this filter. Nested sub-tools inside a dropped
/// `"namespace"` entry (e.g. its own agent-management functions) are
/// dropped along with it rather than hoisted to the top level: the local
/// model losing access to those secondary tools is harmless, whereas
/// guessing how to flatten them would risk silently changing their
/// semantics.
fn filter_non_function_tools(req: &mut serde_json::Value) {
    if let Some(tools) = req.get_mut("tools").and_then(|t| t.as_array_mut()) {
        tools.retain(|t| t.get("type").and_then(|v| v.as_str()) == Some("function"));
    }
}

/// Folds every `developer`/`system`-role item out of the request's `input`
/// array into the top-level `instructions` string, removing them from
/// `input`, before proxying to llama-server.
///
/// llama-server's own `/v1/responses` → chat-completions conversion
/// (`server_chat_convert_responses_to_chatcmpl` in llama.cpp's
/// `tools/server/server-chat.cpp`) unconditionally prepends one
/// `system`-role chat message built from `instructions`, but otherwise
/// forwards every `input` item's `role` field untouched. A later,
/// model-agnostic pass in llama.cpp's own chat-template layer
/// (`workaround::map_developer_role_to_system` in `common/chat.cpp`) then
/// unconditionally rewrites *every* remaining `role: "developer"` message
/// to `role: "system"`, wherever it sits in the array, with no
/// repositioning or merging. Real Codex requests routinely carry a
/// `developer`-role item further into `input` (permissions/skills
/// instructions) alongside the top-level `instructions` string, which
/// after that rewrite leaves two `system`-role messages in the
/// chat-completions request llama-server builds — the second one not at
/// index 0, which strict chat templates (Qwen3.5's included) reject
/// outright with "System message must be at the beginning". This is a
/// confirmed, currently-unresolved upstream llama.cpp gap (e.g.
/// ggml-org/llama.cpp#20733, ggml-org/llama.cpp#23423; a fix was proposed
/// and abandoned in ggml-org/llama.cpp#20079) rather than anything this
/// module's own /v1/messages-style message-building does, so it can't be
/// fixed the same way — this route is a pass-through by design (see the
/// module doc comment above). Folding every developer/system input item
/// into `instructions` here instead keeps the request in a shape
/// llama-server can never turn into more than one system message,
/// regardless of that upstream gap.
fn consolidate_responses_instructions(req: &mut serde_json::Value) {
    let mut instructions = req
        .get("instructions")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if let Some(input) = req.get_mut("input").and_then(|v| v.as_array_mut()) {
        input.retain(|item| {
            let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("");
            if role != "developer" && role != "system" {
                return true;
            }
            if let Some(text) = responses_input_item_text(item) {
                if !text.is_empty() {
                    if !instructions.is_empty() {
                        instructions.push_str("\n\n");
                    }
                    instructions.push_str(&text);
                }
            }
            false
        });
    }

    if !instructions.is_empty() {
        req["instructions"] = serde_json::Value::String(instructions);
    }
}

/// Extracts the plain text of a Responses-API `input` message item —
/// `content` is either a bare string or an array of blocks (each with a
/// `"text"` field, e.g. `{"type":"input_text","text":"..."}`), the same
/// two shapes Anthropic's own message content takes (see
/// `AnthropicContent::as_text` above).
fn responses_input_item_text(item: &serde_json::Value) -> Option<String> {
    match item.get("content")? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(blocks) => Some(
            blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join(""),
        ),
        _ => None,
    }
}

// -- Anthropic /v1/messages --------------------------------------------------

/// Merges every system-role turn in an Anthropic request into a single
/// leading system message, then appends every other message in order.
///
/// Real Claude Code doesn't confine system content to the top-level
/// `system` field: it also injects background reminders (available
/// agents/skills, etc.) as ordinary entries with `"role": "system"`
/// scattered later in `messages`, which the real Anthropic API accepts in
/// any position. llama.cpp's chat templates (Qwen's included) are far
/// stricter and raise "System message must be at the beginning" the
/// moment a `system` role appears anywhere but index 0 — which every
/// sufficiently long real Claude Code session eventually triggers.
/// Concatenating them here keeps every request llama.cpp-template-safe
/// regardless of where the client put its system-role content.
fn build_anthropic_messages(req: &AnthropicRequest) -> Vec<OAIMessage> {
    let mut system_text = String::new();
    if let Some(sys) = &req.system {
        system_text.push_str(&sys.as_text());
    }
    let mut messages: Vec<OAIMessage> = Vec::new();
    for m in &req.messages {
        if m.role == "system" {
            if !system_text.is_empty() {
                system_text.push_str("\n\n");
            }
            system_text.push_str(&m.content.as_text());
            continue;
        }
        messages.push(OAIMessage {
            role: m.role.clone(),
            content: m.content.as_text(),
        });
    }
    if !system_text.is_empty() {
        messages.insert(
            0,
            OAIMessage {
                role: "system".into(),
                content: system_text,
            },
        );
    }
    messages
}

async fn handle_anthropic_messages(
    State(state): State<AppState>,
    Json(req): Json<AnthropicRequest>,
) -> Result<Response, AppError> {
    // Backend needs its canonical name (see ensure_model); the response
    // below still echoes req.model back, unchanged from before.
    let (canonical_model, port) = ensure_model(&state, &req.model).await?;
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");

    let messages = build_anthropic_messages(&req);

    let oai = OAIChatRequest {
        model: canonical_model,
        messages,
        stream: req.stream,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
        // The Anthropic Messages API has no repeat_penalty concept of its
        // own to read an override from — see DEFAULT_REPEAT_PENALTY.
        repeat_penalty: Some(DEFAULT_REPEAT_PENALTY),
        // Nor a `think` override — see think_to_chat_template_kwargs.
        chat_template_kwargs: None,
    };

    if req.stream {
        stream_anthropic(state.0.client.clone(), url, oai, req.model).await
    } else {
        let resp = state
            .0
            .client
            .post(&url)
            .json(&oai)
            .send()
            .await
            .context("send to llama-server")?;
        let body: serde_json::Value = resp.json().await.context("parse llama-server response")?;
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        Ok(Json(serde_json::json!({
            "id": format!("msg_{}", gen_id()),
            "type": "message",
            "role": "assistant",
            "content": [{ "type": "text", "text": content }],
            "model": req.model,
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": { "input_tokens": 0, "output_tokens": 0 }
        }))
        .into_response())
    }
}

// ---------------------------------------------------------------------------
// Option extractors from Ollama options blob
// ---------------------------------------------------------------------------

fn opt_f64(opts: &Option<serde_json::Value>, key: &str) -> Option<f32> {
    opts.as_ref()?.get(key)?.as_f64().map(|f| f as f32)
}

fn opt_u32(opts: &Option<serde_json::Value>, key: &str) -> Option<u32> {
    opts.as_ref()?.get(key)?.as_u64().map(|n| n as u32)
}

// ---------------------------------------------------------------------------
// llama-server binary resolution
// ---------------------------------------------------------------------------

/// Env var names Ollama documents for selecting specific GPU device(s)
/// within whichever backend is active (see docs/gpu.mdx's "Overrides"
/// sections: `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`,
/// `ROCR_VISIBLE_DEVICES`, `GGML_VK_VISIBLE_DEVICES`). A local
/// `llama-server` child already inherits these from `llmman serve`'s own
/// environment with no extra code — they're forwarded explicitly here
/// anyway so intent doesn't silently depend on `Command`'s default
/// env-inheritance behavior, and so the exact same list can be reused
/// as-is by `crate::container::spawn`, whose `docker run`/`podman run`
/// does *not* inherit the host environment into the container on its own.
pub const GPU_VISIBLE_DEVICE_VARS: &[&str] = &[
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GGML_VK_VISIBLE_DEVICES",
];

/// Resolves the `llama-server` binary to run locally (no `--ociman`):
/// prefers whatever is already on `PATH` untouched, unless
/// `pinned_version` explicitly asks for a specific llama.cpp release, in
/// which case that pin always wins. Falls back to downloading and caching
/// a release build matching this host's OS/arch/GPU backend via
/// `crate::llama_release` when nothing suitable is on PATH.
fn resolve_llama_server(pinned_version: Option<&str>) -> anyhow::Result<PathBuf> {
    if pinned_version.is_none() {
        if let Some(p) = find_on_path("llama-server") {
            return Ok(p);
        }
    }
    let resolved = crate::llama_release::ensure_llama_server(pinned_version)
        .context("no llama-server on PATH and automatic download failed")?;
    eprintln!(
        "[llmman] using downloaded llama-server ({}): {}",
        resolved.backend_label,
        resolved.bin.display()
    );
    Ok(resolved.bin)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run(args: &ServeArgs) -> anyhow::Result<()> {
    tokio::runtime::Runtime::new()?.block_on(serve_async(args))
}

async fn serve_async(_args: &ServeArgs) -> anyhow::Result<()> {
    if _args.ociman.is_some() && !cfg!(target_os = "linux") {
        anyhow::bail!("--ociman is only supported on Linux");
    }
    // Must happen before daemon.rs's caller (if any) redirects this
    // process's stdio to a log file — see ServeArgs::pull_oci's doc
    // comment for why that would otherwise hide the pull's progress.
    // This is meant as its own explicit, foreground warm-up step run
    // before a separate, detached `serve` invocation — not a prelude to
    // this same invocation going on to serve — so it returns as soon as
    // the pull finishes instead of falling through into binding the
    // listener and serving forever.
    if _args.pull_oci {
        let ociman = _args.ociman.context("--pull-oci requires --ociman")?;
        crate::container::pull_image(ociman, _args.llama_cpp_version.as_deref())?;
        return Ok(());
    }
    // Same idea as --pull-oci above, but for the local (non-container)
    // llama-server binary path: resolve_llama_server's own download
    // (crate::llama_release) normally happens further down regardless of
    // --pull-bin, but by then this process may already be detached with
    // its stdio redirected to a log file (see daemon.rs) — a caller
    // waiting on the daemon to come up within ensure_server's short
    // timeout would see nothing and could time out mid-download,
    // indistinguishable from a hang. Run in the foreground first instead.
    if _args.pull_bin {
        let pinned_version = _args.llama_cpp_version.clone();
        tokio::task::spawn_blocking(move || resolve_llama_server(pinned_version.as_deref()))
            .await
            .context("resolve llama-server task panicked")??;
        return Ok(());
    }
    // Only resolve (and require) a local llama-server binary when it'll
    // actually be used: --ociman runs llama-server in a container instead,
    // picking the image itself (see crate::container).
    //
    // resolve_llama_server does blocking network I/O (a GitHub API call,
    // and possibly a multi-hundred-MB download) when no llama-server is
    // already on PATH — spawn_blocking so that doesn't stall this async
    // fn's own executor thread while it runs.
    let llama_server_bin = if _args.ociman.is_none() {
        let pinned_version = _args.llama_cpp_version.clone();
        Some(
            tokio::task::spawn_blocking(move || resolve_llama_server(pinned_version.as_deref()))
                .await
                .context("resolve llama-server task panicked")??,
        )
    } else {
        None
    };
    let store_path = default_store(_args.store.as_deref())?;
    let cache_path = store_path.parent().unwrap_or(&store_path).join("cache");
    std::fs::create_dir_all(&cache_path)?;

    let state = AppState(Arc::new(Inner {
        manager: Mutex::new(ModelManager {
            running: HashMap::new(),
        }),
        llama_server_bin: StdMutex::new(llama_server_bin),
        // Canonicalized now, while the file certainly still exists —
        // resolving later (in the handler) could fail once the install is
        // deleted, exactly the situation /api/version exists to expose.
        exe: std::env::current_exe()
            .ok()
            .map(|p| p.canonicalize().unwrap_or(p)),
        ociman: _args.ociman,
        llama_cpp_version: _args.llama_cpp_version.clone(),
        ctx_size: _args.ctx_size,
        store_path,
        cache_path,
        client: Client::new(),
    }));

    let app_state = state.clone();
    let app = Router::new()
        // Web UI
        .route("/", get(handle_root))
        .route("/bundle.js", get(handle_bundle_js))
        .route("/bundle.css", get(handle_bundle_css))
        .route("/loading.html", get(handle_loading_html))
        // llama.cpp-compatible props endpoint (router mode)
        .route("/props", get(handle_props))
        // Ollama API
        .route("/api/version", get(handle_version))
        .route("/api/tags", get(handle_tags))
        .route("/api/ps", get(handle_ps))
        .route("/api/show", post(handle_show))
        .route("/api/pull", post(handle_pull))
        .route("/api/push", post(handle_push))
        .route("/api/delete", delete(handle_delete))
        .route("/api/chat", post(handle_ollama_chat))
        .route("/api/generate", post(handle_ollama_generate))
        // OpenAI API
        .route("/v1/models", get(handle_openai_models))
        .route("/v1/chat/completions", post(handle_openai_chat))
        .route("/v1/completions", post(handle_openai_completions))
        .route("/v1/embeddings", post(handle_openai_embeddings))
        .route("/v1/responses", post(handle_openai_responses))
        .route(
            "/v1/responses/input_tokens",
            post(handle_openai_responses_input_tokens),
        )
        // Anthropic API
        .route("/v1/messages", post(handle_anthropic_messages))
        .with_state(app_state);

    let addr = "127.0.0.1:17434";
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    eprintln!("llmman serve listening on {addr}");

    // If a model was given on the command line, start loading it immediately
    // so the first request finds it already warm.
    if let Some(model) = &_args.model {
        let model = crate::shortnames::resolve_ollama_api(model);
        let state_clone = state.clone();
        tokio::spawn(async move {
            if let Err(e) = ensure_model(&state_clone, &model).await {
                eprintln!("[llmman] pre-load failed: {:#}", e.0);
            }
        });
    }

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // Unload every running inference backend before exiting — the same
    // explicit unload `ollama serve` does when it traps SIGINT/SIGTERM
    // (server/routes.go's signal handler calling sched.unloadAllRunners).
    // Dropping each RunningModel kills local llama-server/vllm children
    // (kill_on_drop) and SIGTERMs container ones (ModelProcess::drop), so
    // nothing is left orphaned with a model still loaded in memory.
    state.0.manager.lock().await.running.clear();
    Ok(())
}

/// Resolves when the daemon is asked to shut down: SIGINT (Ctrl-C) on all
/// platforms, plus SIGTERM on Unix — the same pair `ollama serve` traps
/// (see server/routes.go) and the graceful signal every supervisor sends
/// first (Ollama's app on darwin, llmman's own daemon::stop_stale_daemon,
/// sbx). Trapping it means an in-flight request gets a chance to finish
/// (axum stops accepting and drains) and loaded models are unloaded
/// deliberately, instead of the whole process group being torn down
/// mid-write.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            // Installing the handler failed: never resolve on this arm
            // rather than shutting down immediately for no reason.
            Err(_) => std::future::pending().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    eprintln!("llmman serve shutting down");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the Claude Code bug described on
    /// `build_anthropic_messages`'s own doc comment: a `system`-role
    /// message anywhere in `messages` (not just the top-level `system`
    /// field) must be folded into one message at index 0, never left in
    /// place, or llama.cpp's chat templates raise "System message must be
    /// at the beginning" on the second one.
    #[test]
    fn build_anthropic_messages_merges_system_role_messages_anywhere_in_the_conversation() {
        let req: AnthropicRequest = serde_json::from_value(serde_json::json!({
            "model": "docker.io/ai/qwen3.5:0.8b",
            "system": [{"type": "text", "text": "leading system prompt"}],
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "system", "content": "a mid-conversation reminder"},
                {"role": "user", "content": "bye"}
            ]
        }))
        .unwrap();

        let messages = build_anthropic_messages(&req);

        assert_eq!(
            messages,
            vec![
                OAIMessage {
                    role: "system".into(),
                    content: "leading system prompt\n\na mid-conversation reminder".into(),
                },
                OAIMessage {
                    role: "user".into(),
                    content: "hi".into()
                },
                OAIMessage {
                    role: "assistant".into(),
                    content: "hello".into()
                },
                OAIMessage {
                    role: "user".into(),
                    content: "bye".into()
                },
            ]
        );
    }

    #[test]
    fn build_anthropic_messages_with_no_system_content_has_no_leading_system_message() {
        let req: AnthropicRequest = serde_json::from_value(serde_json::json!({
            "model": "docker.io/ai/qwen3.5:0.8b",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();

        let messages = build_anthropic_messages(&req);

        assert_eq!(
            messages,
            vec![OAIMessage {
                role: "user".into(),
                content: "hi".into()
            }]
        );
    }

    /// Regression test for the Codex tool-type bug described on
    /// `filter_non_function_tools`'s own doc comment.
    #[test]
    fn filter_non_function_tools_drops_non_function_entries_only() {
        let mut req = serde_json::json!({
            "tools": [
                {"type": "function", "name": "exec_command"},
                {"type": "namespace", "name": "multi_agent_v1", "tools": [{"type": "function", "name": "close_agent"}]},
                {"type": "web_search"},
                {"type": "function", "name": "update_plan"}
            ]
        });

        filter_non_function_tools(&mut req);

        assert_eq!(
            req["tools"],
            serde_json::json!([
                {"type": "function", "name": "exec_command"},
                {"type": "function", "name": "update_plan"}
            ])
        );
    }

    #[test]
    fn filter_non_function_tools_is_a_no_op_without_a_tools_field() {
        let mut req = serde_json::json!({"model": "x"});
        filter_non_function_tools(&mut req);
        assert_eq!(req, serde_json::json!({"model": "x"}));
    }

    /// Regression test for the Codex Responses-API bug described on
    /// `consolidate_responses_instructions`'s own doc comment: a
    /// `developer`/`system`-role `input` item must be folded into
    /// `instructions` and removed from `input`, never left in place.
    #[test]
    fn consolidate_responses_instructions_folds_developer_and_system_input_items() {
        let mut req = serde_json::json!({
            "model": "docker.io/ai/qwen3.5:0.8b",
            "instructions": "top-level instructions",
            "input": [
                {"type": "message", "role": "developer", "content": [
                    {"type": "input_text", "text": "permissions instructions"}
                ]},
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "hi"}
                ]},
                {"type": "message", "role": "system", "content": "a plain-string system item"}
            ]
        });

        consolidate_responses_instructions(&mut req);

        assert_eq!(
            req["instructions"],
            "top-level instructions\n\npermissions instructions\n\na plain-string system item"
        );
        assert_eq!(
            req["input"],
            serde_json::json!([
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "hi"}
                ]}
            ])
        );
    }

    #[test]
    fn consolidate_responses_instructions_is_a_no_op_without_developer_or_system_items() {
        let mut req = serde_json::json!({
            "instructions": "top-level instructions",
            "input": [{"type": "message", "role": "user", "content": "hi"}]
        });
        let before = req.clone();
        consolidate_responses_instructions(&mut req);
        assert_eq!(req, before);
    }

    // -- Tests ported from ollama ---------------------------------------------
    //
    // The tests below are ported from ollama's own unit-test suites for the
    // equivalent conversion logic — file references point at ollama/ollama's
    // test files — adapted to llmman's own (narrower) semantics where the two
    // differ; each test's doc comment calls out any such adaptation.

    /// Ported from ollama's openai/openai_test.go
    /// (TestFromChatRequest_ReasoningEffort), adapted to llmman's narrower
    /// mapping: only the plain-boolean `think` forms have an equivalent in
    /// llama-server's `chat_template_kwargs.enable_thinking`; ollama's
    /// string thinking levels ("low"/"medium"/"high"/"max") have no
    /// counterpart there and are deliberately a no-op (None), as is an
    /// absent `think` — the template's own default then applies.
    #[test]
    fn think_to_chat_template_kwargs_maps_booleans_and_ignores_levels() {
        assert_eq!(
            think_to_chat_template_kwargs(&Some(serde_json::json!(true))),
            Some(serde_json::json!({ "enable_thinking": true }))
        );
        assert_eq!(
            think_to_chat_template_kwargs(&Some(serde_json::json!(false))),
            Some(serde_json::json!({ "enable_thinking": false }))
        );
        for level in ["low", "medium", "high", "max", "minimal", "none"] {
            assert_eq!(
                think_to_chat_template_kwargs(&Some(serde_json::json!(level))),
                None,
                "string level {level:?} must be a no-op"
            );
        }
        assert_eq!(think_to_chat_template_kwargs(&None), None);
    }

    /// Ported from ollama's api/client_test.go (TestClientStream /
    /// TestClientDo malformed-payload cases) and openai streaming-chunk
    /// tests: each SSE payload either yields (content, thinking, done) or
    /// is skipped entirely (None) when malformed — a bad chunk must never
    /// abort the whole stream.
    #[test]
    fn oai_chunk_to_content_ported_ollama_stream_decoding_cases() {
        // Plain content token, stream not finished.
        assert_eq!(
            oai_chunk_to_content(
                r#"{"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#
            ),
            Some(("hi".into(), None, false))
        );
        // finish_reason "stop" marks the stream done.
        assert_eq!(
            oai_chunk_to_content(
                r#"{"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}"#
            ),
            Some((String::new(), None, true))
        );
        // The [DONE] sentinel also marks the stream done.
        assert_eq!(
            oai_chunk_to_content("[DONE]"),
            Some((String::new(), None, true))
        );
        // llama-server's two reasoning field spellings both surface as
        // thinking: "reasoning_content" (Homebrew builds) and "thinking"
        // (git builds).
        assert_eq!(
            oai_chunk_to_content(
                r#"{"choices":[{"delta":{"reasoning_content":"hmm"},"finish_reason":null}]}"#
            ),
            Some((String::new(), Some("hmm".into()), false))
        );
        assert_eq!(
            oai_chunk_to_content(
                r#"{"choices":[{"delta":{"thinking":"hmm"},"finish_reason":null}]}"#
            ),
            Some((String::new(), Some("hmm".into()), false))
        );
        // An empty reasoning string is filtered out rather than surfaced.
        assert_eq!(
            oai_chunk_to_content(
                r#"{"choices":[{"delta":{"content":"x","reasoning_content":""},"finish_reason":null}]}"#
            ),
            Some(("x".into(), None, false))
        );
        // Malformed JSON and an empty choices array are skipped, not fatal.
        assert_eq!(oai_chunk_to_content("not json"), None);
        assert_eq!(oai_chunk_to_content(r#"{"choices":[]}"#), None);
    }

    /// Ported from ollama's api/client_test.go (TestClientStream): SSE
    /// lines split across arbitrary TCP chunk boundaries must be
    /// reassembled, CRLF line endings trimmed, and a trailing
    /// unterminated line flushed when the stream ends.
    #[test]
    fn bytes_to_lines_ported_ollama_client_stream_chunking() {
        let chunks: Vec<reqwest::Result<Bytes>> = vec![
            // One logical line split across two chunks.
            Ok(Bytes::from("data: {\"a\":")),
            // ...ending CRLF, plus a complete LF-terminated line.
            Ok(Bytes::from("1}\r\ndata: {\"b\":2}\n")),
            // A trailing line with no terminator at all.
            Ok(Bytes::from("data: tail")),
        ];
        let stream = bytes_to_lines(futures::stream::iter(chunks));
        let lines: Vec<String> = futures::executor::block_on(StreamExt::collect::<Vec<_>>(stream));
        assert_eq!(
            lines,
            vec![
                "data: {\"a\":1}".to_string(),
                "data: {\"b\":2}".to_string(),
                "data: tail".to_string(),
            ]
        );
    }

    /// Ported from ollama's middleware/anthropic_test.go
    /// (TestAnthropicMessagesMiddleware's plain-string `system` case):
    /// Anthropic's `system` field is accepted as either a bare string or
    /// an array of content blocks, and both forms end up as the single
    /// leading system message.
    #[test]
    fn build_anthropic_messages_accepts_a_plain_string_system_field() {
        let req: AnthropicRequest = serde_json::from_value(serde_json::json!({
            "model": "docker.io/ai/qwen3.5:0.8b",
            "system": "you are a helpful assistant",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();

        let messages = build_anthropic_messages(&req);

        assert_eq!(
            messages,
            vec![
                OAIMessage {
                    role: "system".into(),
                    content: "you are a helpful assistant".into()
                },
                OAIMessage {
                    role: "user".into(),
                    content: "hi".into()
                },
            ]
        );
    }

    /// Ported from ollama's middleware/anthropic_test.go content-block
    /// conversion cases: block-array content joins its text blocks in
    /// order and ignores non-text block types entirely.
    #[test]
    fn anthropic_content_as_text_joins_text_blocks_and_ignores_other_types() {
        let plain: AnthropicContent = serde_json::from_value(serde_json::json!("plain")).unwrap();
        assert_eq!(plain.as_text(), "plain");

        let blocks: AnthropicContent = serde_json::from_value(serde_json::json!([
            {"type": "text", "text": "a"},
            {"type": "image", "source": {"type": "base64", "data": "zzzz"}},
            {"type": "text", "text": "b"}
        ]))
        .unwrap();
        assert_eq!(blocks.as_text(), "ab");

        let empty: AnthropicContent = serde_json::from_value(serde_json::json!([])).unwrap();
        assert_eq!(empty.as_text(), "");
    }

    /// Ported from ollama's openai/responses_test.go polymorphic-input
    /// cases: a Responses-API input item's `content` is either a bare
    /// string or an array of text-bearing blocks (`input_text` /
    /// `output_text`), and anything else (a function_call item with no
    /// content, a non-string/array content) yields no text.
    #[test]
    fn responses_input_item_text_ported_ollama_polymorphic_input_cases() {
        assert_eq!(
            responses_input_item_text(&serde_json::json!({"role": "user", "content": "plain"})),
            Some("plain".into())
        );
        assert_eq!(
            responses_input_item_text(&serde_json::json!({"role": "user", "content": [
                {"type": "input_text", "text": "a"},
                {"type": "output_text", "text": "b"}
            ]})),
            Some("ab".into())
        );
        // Blocks without a text field contribute nothing.
        assert_eq!(
            responses_input_item_text(&serde_json::json!({"content": [{"type": "input_image"}]})),
            Some(String::new())
        );
        assert_eq!(
            responses_input_item_text(&serde_json::json!({"type": "function_call", "name": "f"})),
            None
        );
        assert_eq!(
            responses_input_item_text(&serde_json::json!({"content": 42})),
            None
        );
    }

    /// Ported from ollama's server/routes_options_test.go concept
    /// (api.Options blob -> typed option values): numeric options are
    /// pulled out of the Ollama `options` blob by key, and missing keys,
    /// wrong-typed values, or an absent blob all yield None instead of
    /// erroring.
    #[test]
    fn option_extractors_ported_ollama_options_blob_cases() {
        let opts = Some(serde_json::json!({
            "temperature": 0.5,
            "top_p": 0.9,
            "num_predict": 128,
            "stop": ["### User:"]
        }));
        assert_eq!(opt_f64(&opts, "temperature"), Some(0.5));
        assert_eq!(opt_f64(&opts, "top_p"), Some(0.9));
        assert_eq!(opt_u32(&opts, "num_predict"), Some(128));
        // Missing key.
        assert_eq!(opt_f64(&opts, "repeat_penalty"), None);
        // Wrong type for the extractor.
        assert_eq!(opt_u32(&opts, "stop"), None);
        // No options blob at all.
        assert_eq!(opt_f64(&None, "temperature"), None);
        assert_eq!(opt_u32(&None, "num_predict"), None);
    }

    #[test]
    fn keyed_lock_is_per_key_and_release_only_drops_unreferenced_entries() {
        let registry: StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>> =
            StdMutex::new(HashMap::new());

        let a1 = keyed_lock(&registry, "model-a");
        let a2 = keyed_lock(&registry, "model-a");
        assert!(Arc::ptr_eq(&a1, &a2), "same key must return the same lock");

        let b = keyed_lock(&registry, "model-b");
        assert!(
            !Arc::ptr_eq(&a1, &b),
            "different keys must not share a lock"
        );

        // Caller 1 finishes and releases its own clone — but caller 2's
        // clone (a2) is still outstanding, so the entry must survive.
        drop(a1);
        release_keyed_lock(&registry, "model-a");
        assert!(registry.lock().unwrap().contains_key("model-a"));

        // Caller 2 finishes too — now only the registry itself references
        // it, so releasing removes the entry.
        drop(a2);
        release_keyed_lock(&registry, "model-a");
        assert!(!registry.lock().unwrap().contains_key("model-a"));

        drop(b);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn load_lock_serializes_same_model_but_not_different_models() {
        let slow = load_lock("test-load-lock-slow-model");
        let guard = slow.lock().await; // simulates a mid-flight cold start

        // A different model's load must acquire immediately.
        let other = load_lock("test-load-lock-other-model");
        let _other_guard =
            tokio::time::timeout(std::time::Duration::from_millis(200), other.lock())
                .await
                .expect("a different model's load must not block on an unrelated one");

        // The same model's load must not acquire until the first releases.
        let same = load_lock("test-load-lock-slow-model");
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), same.lock())
                .await
                .is_err(),
            "a second load of the same model must block while the first is in flight"
        );

        drop(guard);
        let same_guard = tokio::time::timeout(std::time::Duration::from_millis(200), same.lock())
            .await
            .expect("must acquire promptly once the first load releases");

        drop(same_guard);
        drop(_other_guard);
        drop(same);
        drop(other);
        drop(slow);
        release_load_lock("test-load-lock-slow-model");
        release_load_lock("test-load-lock-other-model");
    }

    /// Regression: aliases of an unpulled model must key into one lock
    /// (see `ensure_model`'s `default_tag` call).
    #[test]
    fn ensure_model_key_pipeline_converges_aliases_before_the_lock() {
        let tagless = crate::storage::default_tag(&crate::shortnames::resolve_ollama_api(
            "regression-test-model",
        ));
        let tagged = crate::storage::default_tag(&crate::shortnames::resolve_ollama_api(
            "regression-test-model:latest",
        ));
        assert_eq!(
            tagless, tagged,
            "tagless and :latest aliases must resolve to one key"
        );

        let a = load_lock(&tagless);
        let b = load_lock(&tagged);
        assert!(
            Arc::ptr_eq(&a, &b),
            "both aliases must take the same load lock"
        );

        drop(a);
        drop(b);
        release_load_lock(&tagless);
    }

    /// Regression: a call site that drops its guard but not its own `Arc`
    /// clone before calling `release_load_lock` leaves the entry stuck.
    #[tokio::test]
    async fn load_lock_release_actually_removes_the_entry_once_unused() {
        let key = "test-load-lock-release-cleanup";
        let lock = load_lock(key);
        let guard = lock.lock().await;
        drop(guard);
        drop(lock);
        release_load_lock(key);
        assert!(
            !LOAD_LOCKS.lock().unwrap().contains_key(key),
            "release_load_lock must drop the registry entry once nothing else references it"
        );
    }

    /// Regression: aborting a task while it holds a `LoadLockGuard` must
    /// still release the registry entry. `acquire_load_lock`'s caller
    /// (`ensure_model`, the unload handler) can itself be cancelled by axum
    /// mid-`.await` (a dropped client connection) — code placed after an
    /// `.await` doesn't run in that case, so cleanup must live in `Drop`.
    #[tokio::test(flavor = "multi_thread")]
    async fn load_lock_guard_releases_on_task_cancellation() {
        let key = "test-load-lock-guard-cancel";
        let started = Arc::new(tokio::sync::Notify::new());
        let started_tx = started.clone();
        let handle = tokio::spawn(async move {
            let _guard = acquire_load_lock("test-load-lock-guard-cancel").await;
            started_tx.notify_one();
            std::future::pending::<()>().await;
        });
        started.notified().await;
        handle.abort();
        let _ = handle.await;

        assert!(
            !LOAD_LOCKS.lock().unwrap().contains_key(key),
            "aborting a task holding LoadLockGuard must still release the registry entry"
        );
    }

    /// Regression test for `OllamaPullRequest`'s `name` field: a body
    /// carrying only `{"name": "..."}` used to fail Axum's `Json`
    /// extraction outright — `model` was a required, non-default field —
    /// before this handler's own name-falls-back-to-model logic ever ran.
    #[test]
    fn ollama_pull_request_accepts_a_name_only_body() {
        let req: OllamaPullRequest =
            serde_json::from_value(serde_json::json!({"name": "docker.io/ai/gemma4:E2B"}))
                .expect("a name-only body must still deserialize");
        assert_eq!(req.model, "");
        assert_eq!(req.name, "docker.io/ai/gemma4:E2B");
    }

    #[test]
    fn ollama_pull_request_accepts_a_model_only_body() {
        let req: OllamaPullRequest =
            serde_json::from_value(serde_json::json!({"model": "docker.io/ai/gemma4:E2B"}))
                .expect("a model-only body must still deserialize");
        assert_eq!(req.model, "docker.io/ai/gemma4:E2B");
        assert_eq!(req.name, "");
    }

    #[test]
    fn ollama_push_request_accepts_a_name_only_body() {
        let req: OllamaPushRequest =
            serde_json::from_value(serde_json::json!({"name": "docker.io/ai/gemma4:E2B"}))
                .expect("a name-only body must still deserialize");
        assert_eq!(req.model, "");
        assert_eq!(req.name, "docker.io/ai/gemma4:E2B");
    }
}

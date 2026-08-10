//! `llmman serve` – HTTP server exposing Ollama, OpenAI (including the
//! Responses API), and Anthropic-compatible APIs backed by `llama-server`
//! sub-processes from llama.cpp.

use std::collections::HashMap;
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
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration, Instant};

use crate::default_store;
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

    /// Pin the ghcr.io/ggml-org/llama.cpp container image to a specific
    /// release tag (e.g. b9994) instead of the floating server/server-cuda/
    /// ... tags — only meaningful together with --ociman, ignored
    /// otherwise. llmman itself has no default or opinion here: pick a
    /// tag that's actually published for every backend variant you might
    /// run (see docs/docker.md in ggml-org/llama.cpp) and pass it
    /// explicitly if you want reproducible behavior across runs.
    #[arg(long, value_name = "TAG", requires = "ociman")]
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
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState(Arc<Inner>);

struct Inner {
    manager: Mutex<ModelManager>,
    // None when --ociman is set: llama-server then runs in a container, so
    // no local binary is resolved (or required on PATH) at all.
    llama_server_bin: Option<PathBuf>,
    ociman: Option<crate::container::ContainerManager>,
    llama_cpp_version: Option<String>,
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
            ModelProcess::Local(Engine::LlamaServer, _) => "llama-server (local)".into(),
            ModelProcess::Local(Engine::Vllm, _) => "vllm (local)".into(),
            ModelProcess::Container(ociman, _) => format!("llama-server (container/{})", ociman.binary()),
        }
    }

    fn pid(&self) -> Option<u32> {
        match &self.process {
            ModelProcess::Local(_, child) => child.id(),
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
/// process (killed via `Child::kill_on_drop`, as before, set at spawn
/// time) or an attached `docker run --rm --init -t`/`podman run` process
/// (see crate::container::spawn's doc comment) — gracefully stopped via
/// SIGTERM on drop, since the default forceful kill `kill_on_drop` would
/// use cannot be forwarded to (and so does not stop) the container.
enum ModelProcess {
    Local(Engine, #[allow(dead_code)] tokio::process::Child),
    Container(crate::container::ContainerManager, tokio::process::Child),
}

impl Drop for ModelProcess {
    fn drop(&mut self) {
        if let ModelProcess::Container(_, child) = self {
            if let Some(pid) = child.id() {
                crate::container::stop(pid);
            }
        }
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
}

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
// Model resolution – GGUF (llama-server) or safetensors (vllm)
// ---------------------------------------------------------------------------

const HF_GGUF_MEDIA_TYPE: &str = "application/vnd.docker.ai.gguf.v3";

/// What kind of model did we find in the OCI store?
enum ModelPath {
    /// A GGUF file — serve with llama-server.
    Gguf(PathBuf),
    /// A safetensors directory — serve with vllm.
    SafeTensors(PathBuf),
}

/// Splits an OCI digest ("sha256:abcd...") down to just its hex portion,
/// which is what the blob store's on-disk layout uses as the filename.
fn digest_hex(digest: &str) -> anyhow::Result<&str> {
    digest
        .split_once(':')
        .map(|(_, hex)| hex)
        .ok_or_else(|| anyhow!("malformed digest: {digest}"))
}

fn layer_filepath(l: &crate::storage::oci::Descriptor) -> Option<&str> {
    l.annotations.as_ref().and_then(|a| {
        a.get("org.cncf.model.filepath")
            .or_else(|| a.get("org.opencontainers.image.title"))
            .map(|s| s.as_str())
    })
}

fn is_gguf_layer(l: &crate::storage::oci::Descriptor) -> bool {
    if l.media_type == HF_GGUF_MEDIA_TYPE { return true; }
    layer_filepath(l).map(|p| p.to_lowercase().ends_with(".gguf")).unwrap_or(false)
}

fn is_safetensors_layer(l: &crate::storage::oci::Descriptor) -> bool {
    layer_filepath(l).map(|p| p.to_lowercase().ends_with(".safetensors")).unwrap_or(false)
}

fn resolve_model(store_path: &Path, cache_path: &Path, model_ref: &str) -> anyhow::Result<ModelPath> {
    let store = OciStore::open(store_path)?;
    let desc = store
        .find(model_ref)
        .with_context(|| format!("model not found in store: {model_ref}"))?;
    let manifest = store.read_manifest(&desc.digest)?;

    // ── GGUF → llama-server ────────────────────────────────────────────────
    if let Some(gguf_layer) = manifest.layers.iter().find(|l| is_gguf_layer(l)) {
        let title = layer_filepath(gguf_layer).unwrap_or("model.gguf").to_owned();
        let layer_hex = digest_hex(&gguf_layer.digest)?;

        // HF blobs are stored as raw GGUF — use directly.
        if gguf_layer.media_type == HF_GGUF_MEDIA_TYPE {
            let blob_path = store_path.join("blobs").join("sha256").join(layer_hex);
            if blob_path.exists() {
                eprintln!("[llmman] using blob directly: {}", blob_path.display());
                return Ok(ModelPath::Gguf(blob_path));
            }
        }

        // Otherwise extract from tar layer.
        let cached_dir = cache_path.join(layer_hex);
        if cached_dir.exists() {
            for e in std::fs::read_dir(&cached_dir)?.flatten() {
                let p = e.path();
                if p.extension().and_then(|e| e.to_str()) == Some("gguf") {
                    return Ok(ModelPath::Gguf(p));
                }
            }
        }
        std::fs::create_dir_all(&cached_dir)?;
        let blob = store.read_blob(&gguf_layer.digest)
            .with_context(|| format!("read blob {}", gguf_layer.digest))?;
        let dest = if blob.len() >= 4 && &blob[..4] == b"GGUF" {
            let name = Path::new(&title).file_name()
                .unwrap_or_else(|| std::ffi::OsStr::new("model.gguf"));
            let p = cached_dir.join(name);
            std::fs::write(&p, &blob)?;
            p
        } else {
            let mut archive = tar::Archive::new(std::io::Cursor::new(&blob));
            let mut extracted = None;
            for entry in archive.entries()? {
                let mut entry = entry?;
                let ep = entry.path()?.to_path_buf();
                if ep.extension().and_then(|e| e.to_str()) == Some("gguf") {
                    let name = ep.file_name()
                        .unwrap_or_else(|| std::ffi::OsStr::new("model.gguf"));
                    let d = cached_dir.join(name);
                    entry.unpack(&d)?;
                    extracted = Some(d);
                    break;
                }
            }
            extracted.ok_or_else(|| anyhow!("no .gguf in tar layer of {model_ref}"))?
        };
        return Ok(ModelPath::Gguf(dest));
    }

    // ── safetensors → vllm ────────────────────────────────────────────────
    if manifest.layers.iter().any(|l| is_safetensors_layer(l)) {
        let model_dir = extract_safetensors_dir(&store, store_path, cache_path, &desc.digest, &manifest)?;
        return Ok(ModelPath::SafeTensors(model_dir));
    }

    // Nothing usable found — report what was present.
    let exts: std::collections::HashSet<String> = manifest.layers.iter()
        .filter_map(|l| layer_filepath(l))
        .filter_map(|p| Path::new(p).extension()?.to_str().map(|e| e.to_lowercase()))
        .collect();
    if exts.is_empty() {
        anyhow::bail!("no servable model layer found in {model_ref}");
    } else {
        anyhow::bail!(
            "no servable model layer in {model_ref} — found {exts:?} files; \
             llmman serve supports GGUF (llama-server) and safetensors (vllm)"
        );
    }
}

/// Extract CNCF-format safetensors layers to a cache directory and return the
/// model directory (parent of `config.json`).
fn extract_safetensors_dir(
    store: &OciStore,
    store_path: &Path,
    cache_path: &Path,
    manifest_digest: &str,
    manifest: &crate::storage::oci::Manifest,
) -> anyhow::Result<PathBuf> {
    let hex = digest_hex(manifest_digest)?;
    let cache_dir = cache_path.join(hex);

    for layer in &manifest.layers {
        // Only extract config and weight files; skip code/docs.
        let include = matches!(
            layer.media_type.as_str(),
            "application/vnd.cncf.model.weight.config.v1.raw"
            | "application/vnd.cncf.model.weight.v1.raw"
        );
        if !include { continue; }

        let Some(rel_path) = layer_filepath(layer) else { continue };
        let dest = cache_dir.join(rel_path);
        if dest.exists() { continue; }

        std::fs::create_dir_all(dest.parent().context("no parent")?)?;
        let layer_hex = digest_hex(&layer.digest)?;
        let blob = store_path.join("blobs").join("sha256").join(layer_hex);
        std::fs::copy(&blob, &dest)
            .with_context(|| format!("copy {rel_path} from blob store"))?;
        eprintln!("[llmman] extracted {rel_path}");
    }

    // Model dir = parent of config.json
    for layer in &manifest.layers {
        let Some(rel_path) = layer_filepath(layer) else { continue };
        if Path::new(rel_path).file_name().map(|n| n == "config.json").unwrap_or(false) {
            let config = cache_dir.join(rel_path);
            return config.parent().map(|p| p.to_path_buf())
                .ok_or_else(|| anyhow!("config.json has no parent directory"));
        }
    }
    Ok(cache_dir)
}

// ---------------------------------------------------------------------------
// Process management
// ---------------------------------------------------------------------------

fn find_free_port() -> anyhow::Result<u16> {
    let l = TcpListener::bind("127.0.0.1:0")?;
    Ok(l.local_addr()?.port())
}

async fn spawn_llama_server(
    bin: &Path,
    model: &Path,
    port: u16,
) -> anyhow::Result<tokio::process::Child> {
    tokio::process::Command::new(bin)
        .args([
            "--model",
            model.to_str().context("non-UTF-8 model path")?,
            "--port",
            &port.to_string(),
            "--host",
            "127.0.0.1",
        ])
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("spawn llama-server from {}", bin.display()))
}

async fn spawn_vllm_server(model_dir: &Path, port: u16, model_name: &str) -> anyhow::Result<tokio::process::Child> {
    let vllm = which_binary("vllm")?;
    tokio::process::Command::new(&vllm)
        .args([
            "serve",
            model_dir.to_str().context("non-UTF-8 model path")?,
            "--port", &port.to_string(),
            "--host", "127.0.0.1",
            // Register the model under the same name used in API requests so
            // {"model": "<ref>"} is accepted by vllm's OpenAI-compatible API.
            "--served-model-name", model_name,
        ])
        .kill_on_drop(true)
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

async fn wait_for_ready(client: &Client, port: u16) -> anyhow::Result<()> {
    let url = format!("http://127.0.0.1:{port}/health");
    // vllm can take several minutes to load large models.
    let deadline = Instant::now() + Duration::from_secs(600);
    loop {
        if Instant::now() > deadline {
            return Err(anyhow!("inference server on port {port} did not become ready within 600s"));
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

/// Returns (creating if absent) the lock serializing pull/push calls for
/// `model`. Cheap and non-blocking: it only ever holds `MODEL_LOCKS`'s own
/// short-lived std mutex to look up or insert the entry, never the
/// per-model tokio mutex itself.
fn model_lock(model: &str) -> Arc<tokio::sync::Mutex<()>> {
    let mut locks = MODEL_LOCKS.lock().unwrap();
    locks
        .entry(model.to_owned())
        .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
}

/// Drops model's entry from `MODEL_LOCKS` once nobody else appears to be
/// waiting on it, so a long-running daemon doesn't accumulate one entry
/// per distinct model it has ever pulled/pushed. Called after releasing
/// our own clone of the lock: at that point a strong count of 1 means
/// only `MODEL_LOCKS` itself still references it (safe to remove), while
/// a higher count means another caller is already holding or waiting on
/// this same Arc and should keep using it — removing the map entry in
/// that case wouldn't break anything (that caller's clone stays valid
/// independent of the map), it would just mean the *next* new caller for
/// this model gets handed a fresh, unrelated lock instead of piggybacking
/// on the map's copy of this one, so it's simplest to just leave it.
fn release_model_lock(model: &str) {
    let mut locks = MODEL_LOCKS.lock().unwrap();
    if let Some(arc) = locks.get(model) {
        if Arc::strong_count(arc) <= 1 {
            locks.remove(model);
        }
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
        if OciStore::open(store_path).and_then(|s| s.find(model)).is_ok() {
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

/// Resolve a user-supplied model ref to the canonical reference stored in the
/// OCI index (e.g. "hf.co/repo" → "hf.co/repo:latest").  Using the canonical
/// form as the map key means "hf.co/repo" and "hf.co/repo:latest" both hit
/// the same running process rather than spawning a second one.
fn canonical_ref(store_path: &std::path::Path, model_ref: &str) -> String {
    let Ok(store) = crate::storage::OciStore::open(store_path) else { return model_ref.to_owned() };
    let Ok(desc)  = store.find(model_ref)                        else { return model_ref.to_owned() };
    desc.annotations
        .as_ref()
        .and_then(|a| a.get("org.opencontainers.image.ref.name"))
        .cloned()
        .unwrap_or_else(|| model_ref.to_owned())
}

async fn ensure_model(state: &AppState, model_ref: &str) -> Result<u16, AppError> {
    let model_ref = crate::shortnames::resolve_ollama_api(model_ref);
    let model_ref = canonical_ref(&state.0.store_path, &model_ref);
    let model_ref = model_ref.as_str();

    // Fast path: model already running.
    {
        let mgr = state.0.manager.lock().await;
        if let Some(m) = mgr.running.get(model_ref) {
            return Ok(m.port);
        }
    } // mutex released before any I/O

    // If the model is not in the local store, pull it now.
    // Runs outside the mutex so multi-GB downloads don't block other requests.
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

    // Re-canonicalise after the pull (tag may now be resolvable).
    let model_ref = canonical_ref(&state.0.store_path, model_ref);
    let model_ref = model_ref.as_str();

    let mut mgr = state.0.manager.lock().await;
    // Double-check: another task may have started the server while we were pulling.
    if let Some(m) = mgr.running.get(model_ref) {
        return Ok(m.port);
    }
    let model_path = resolve_model(&state.0.store_path, &state.0.cache_path, model_ref)
        .with_context(|| format!("resolve model {model_ref}"))?;
    // Best-effort — used only to populate `llmman ps`'s ID/SIZE columns;
    // resolve_model above already established the model exists, so a
    // failure here (e.g. a race with a concurrent `rm`) just means those
    // columns show as empty/zero rather than failing the whole request.
    let (digest, size) = OciStore::open(&state.0.store_path)
        .and_then(|s| s.find(model_ref).map(|d| {
            let size = s.total_size(&d);
            (d.digest, size)
        }))
        .unwrap_or_default();
    let port = find_free_port()?;
    eprintln!("[llmman] loading {model_ref} on port {port}");
    let process = match (&model_path, state.0.ociman) {
        (ModelPath::Gguf(path), Some(ociman)) => {
            ModelProcess::Container(
                ociman,
                crate::container::spawn(ociman, path, port, state.0.llama_cpp_version.as_deref())?,
            )
        }
        (ModelPath::Gguf(path), None) => {
            let bin = state.0.llama_server_bin.as_deref().ok_or_else(|| {
                anyhow!("no local llama-server binary resolved and --ociman was not set")
            })?;
            ModelProcess::Local(Engine::LlamaServer, spawn_llama_server(bin, path, port).await?)
        }
        (ModelPath::SafeTensors(dir), _) => {
            ModelProcess::Local(Engine::Vllm, spawn_vllm_server(dir, port, model_ref).await?)
        }
    };
    wait_for_ready(&state.0.client, port).await?;
    eprintln!("[llmman] {model_ref} ready on port {port}");
    mgr.running.insert(
        model_ref.to_string(),
        RunningModel {
            process,
            port,
            digest,
            size,
            started_at: now_rfc3339(),
        },
    );
    Ok(port)
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
        return Err(AppError(anyhow!("inference backend returned empty response body")));
    }

    let text = String::from_utf8_lossy(&raw);
    let mut content = String::new();
    for line in text.lines() {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        match oai_chunk_to_content(payload) {
            Some((tok, _thinking, true)) => { content.push_str(&tok); break; }
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
    let thinking = choice.delta.reasoning_content.clone()
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
async fn post_chat(client: &Client, url: &str, oai_req: &OAIChatRequest) -> Result<reqwest::Response, AppError> {
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
        let out = line.strip_prefix("data: ")
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

    let preamble_stream = futures::stream::once(futures::future::ready(
        Ok::<_, std::convert::Infallible>(Bytes::from(preamble)),
    ));

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
        "build_info": env!("CARGO_PKG_VERSION"),
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

async fn handle_version() -> impl IntoResponse {
    // `build` is llmman-specific (real Ollama's /api/version has no such
    // field) — extra JSON keys are ignored by every well-behaved client, so
    // this stays wire-compatible while letting daemon::ensure_server detect
    // a stale already-running `llmman serve` (see build_fingerprint's own
    // doc comment) and restart it instead of silently talking to
    // already-superseded code forever.
    Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "build": crate::daemon::build_fingerprint(),
    }))
}

/// Handles `POST /api/shutdown` — used only by `daemon::ensure_server` to
/// terminate a stale `llmman serve` it's about to replace (see
/// build_fingerprint's doc comment). Responds first, then exits from a
/// detached task after a brief delay so the response actually reaches the
/// client instead of the connection just dropping mid-write.
async fn handle_shutdown() -> impl IntoResponse {
    tokio::spawn(async {
        tokio::time::sleep(Duration::from_millis(200)).await;
        std::process::exit(0);
    });
    StatusCode::OK
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
            modified_at: img.modified_at
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
    body.get("default_generation_settings")?.get("n_ctx")?.as_u64()
}

async fn handle_show(
    State(state): State<AppState>,
    Json(req): Json<OllamaShowRequest>,
) -> Result<impl IntoResponse, AppError> {
    // ollama sends either {"name":"..."} or {"model":"..."} depending on call site;
    // filter out empty strings so we always fall back to whichever field is populated.
    let model_ref = req.name.as_deref().filter(|s| !s.is_empty())
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
    model: String,
    #[serde(alias = "name", default)]
    _name: String,
}

async fn handle_pull(
    State(state): State<AppState>,
    Json(req): Json<OllamaPullRequest>,
) -> impl IntoResponse {
    let model = crate::shortnames::resolve_ollama_api(&req.model);
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
    model: String,
    #[serde(alias = "name", default)]
    _name: String,
}

async fn handle_push(
    State(state): State<AppState>,
    Json(req): Json<OllamaPushRequest>,
) -> impl IntoResponse {
    let model = crate::shortnames::resolve_ollama_api(&req.model);
    eprintln!("[llmman] /api/push model={model:?}");
    let store_path = state.0.store_path.clone();

    // Unlike pull, there's nothing sensible to do if the model isn't
    // already in the local store — push has no "fetch it first" fallback.
    if OciStore::open(&store_path).and_then(|s| s.find(&model)).is_err() {
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
    let model_ref = req.name.as_deref().filter(|s| !s.is_empty())
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
    eprintln!("[llmman] /api/chat model={:?} messages={}", req.model, req.messages.len());
    let port = ensure_model(&state, &req.model).await?;
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let oai = OAIChatRequest {
        model: req.model.clone(),
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
    };
    let model = req.model;
    stream_ollama(state.0.client.clone(), url, oai, move |content, thinking, done| {
        OllamaChatChunk {
            model: model.clone(),
            created_at: now_rfc3339(),
            message: OllamaMessage { role: "assistant".into(), content, thinking },
            done,
            done_reason: done.then_some("stop".into()),
        }
    })
    .await
}

// -- Ollama /api/generate -----------------------------------------------------

async fn handle_ollama_generate(
    State(state): State<AppState>,
    Json(req): Json<OllamaGenerateRequest>,
) -> Result<Response, AppError> {
    eprintln!("[llmman] /api/generate model={:?} prompt_len={}", req.model, req.prompt.len());

    // Empty prompt + keep_alive:0 = unload request (ollama server/routes.go:354).
    let is_unload = req.prompt.is_empty() && req.keep_alive.as_ref()
        .and_then(|v| v.as_i64()).map(|n| n == 0).unwrap_or(false);
    if is_unload {
        let resolved = crate::shortnames::resolve_ollama_api(&req.model);
        let canonical = canonical_ref(&state.0.store_path, &resolved);
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

    let port = ensure_model(&state, &req.model).await?;
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
        model: req.model.clone(),
        messages: vec![OAIMessage {
            role: "user".into(),
            content: req.prompt.clone(),
        }],
        stream: true,
        temperature: opt_f64(&req.options, "temperature"),
        top_p: opt_f64(&req.options, "top_p"),
        max_tokens: opt_u32(&req.options, "num_predict"),
    };
    let model = req.model;
    stream_ollama(state.0.client.clone(), url, oai, move |response, thinking, done| {
        OllamaGenerateChunk {
            model: model.clone(),
            created_at: now_rfc3339(),
            response,
            thinking,
            done,
            done_reason: done.then_some("stop".into()),
        }
    })
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



/// Shared body of every plain OpenAI-passthrough route: parse just enough of
/// the request to find `model`, make sure it's loaded, then proxy the
/// untouched request body straight through to llama-server's equivalent
/// endpoint. `llama_path` is the only thing that differs between
/// handle_openai_chat/completions/embeddings below.
async fn proxy_openai(
    state: &AppState,
    headers: &HeaderMap,
    body: Bytes,
    llama_path: &str,
) -> Result<Response, AppError> {
    let req: serde_json::Value =
        serde_json::from_slice(&body).context("parse OpenAI request body")?;
    let model = req["model"].as_str().unwrap_or("").to_string();
    let port = ensure_model(state, &model).await?;
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
    let port = ensure_model(&state, &req.model).await?;
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");

    let messages = build_anthropic_messages(&req);

    let oai = OAIChatRequest {
        model: req.model.clone(),
        messages,
        stream: req.stream,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
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

fn resolve_llama_server() -> anyhow::Result<PathBuf> {
    find_on_path("llama-server").ok_or_else(|| {
        anyhow!("llama-server not found; install llama.cpp and ensure it is on PATH")
    })
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
    // Only resolve (and require) a local llama-server binary when it'll
    // actually be used: --ociman runs llama-server in a container instead,
    // picking the image itself (see crate::container).
    let llama_server_bin = if _args.ociman.is_none() {
        Some(resolve_llama_server()?)
    } else {
        None
    };
    let store_path = default_store(_args.store.as_deref())?;
    let cache_path = store_path
        .parent()
        .unwrap_or(&store_path)
        .join("cache");
    std::fs::create_dir_all(&cache_path)?;

    let state = AppState(Arc::new(Inner {
        manager: Mutex::new(ModelManager {
            running: HashMap::new(),
        }),
        llama_server_bin,
        ociman: _args.ociman,
        llama_cpp_version: _args.llama_cpp_version.clone(),
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
        .route("/api/shutdown", post(handle_shutdown))
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

    axum::serve(listener, app).await?;
    Ok(())
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
                OAIMessage { role: "user".into(), content: "hi".into() },
                OAIMessage { role: "assistant".into(), content: "hello".into() },
                OAIMessage { role: "user".into(), content: "bye".into() },
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
            vec![OAIMessage { role: "user".into(), content: "hi".into() }]
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
}

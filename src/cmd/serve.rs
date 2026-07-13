//! `llmman serve` – HTTP server exposing Ollama, OpenAI, and Anthropic-compatible
//! APIs backed by `llama-server` sub-processes from llama.cpp.

use std::collections::HashMap;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState(Arc<Inner>);

struct Inner {
    manager: Mutex<ModelManager>,
    llama_server_bin: PathBuf,
    store_path: PathBuf,
    cache_path: PathBuf,
    client: Client,
}

struct ModelManager {
    running: HashMap<String, RunningModel>,
}

struct RunningModel {
    _child: tokio::process::Child,
    port: u16,
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
    size: u64,
    size_vram: u64,
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
    system: Option<String>,
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

#[derive(Debug, Serialize)]
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
        let (_, layer_hex) = gguf_layer.digest.split_once(':')
            .ok_or_else(|| anyhow!("malformed digest: {}", gguf_layer.digest))?;

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
    let (_, hex) = manifest_digest.split_once(':')
        .ok_or_else(|| anyhow!("malformed manifest digest"))?;
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
        let (_, layer_hex) = layer.digest.split_once(':')
            .ok_or_else(|| anyhow!("malformed layer digest"))?;
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
        tokio::task::spawn_blocking(move || {
            let layout = store_path
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("non-UTF-8 store path"))?;
            crate::ffi::pull(&model_ref_owned, layout)
        })
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
    let port = find_free_port()?;
    eprintln!("[llmman] loading {model_ref} on port {port}");
    let child = match model_path {
        ModelPath::Gguf(ref path) =>
            spawn_llama_server(&state.0.llama_server_bin, path, port).await?,
        ModelPath::SafeTensors(ref dir) =>
            spawn_vllm_server(dir, port, model_ref).await?,
    };
    wait_for_ready(&state.0.client, port).await?;
    eprintln!("[llmman] {model_ref} ready on port {port}");
    mgr.running
        .insert(model_ref.to_string(), RunningModel { _child: child, port });
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

    let resp = client
        .post(url)
        .json(&oai)
        .send()
        .await
        .context("send to llama-server")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(AppError(anyhow!("inference backend {status}: {body}")));
    }
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
// Streaming conversion: OpenAI SSE → Ollama NDJSON (chat)
// ---------------------------------------------------------------------------

async fn stream_ollama_chat(
    client: Client,
    url: String,
    oai_req: OAIChatRequest,
    model: String,
) -> Result<Response, AppError> {
    let resp = client
        .post(&url)
        .json(&oai_req)
        .send()
        .await
        .context("send to llama-server")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(AppError(anyhow!("inference backend {status}: {body}")));
    }

    let stream = bytes_to_lines(resp.bytes_stream()).map(move |line| {
        let out = line.strip_prefix("data: ")
            .and_then(|p| oai_chunk_to_content(p))
            .map(|(content, thinking, done)| {
                let chunk = OllamaChatChunk {
                    model: model.clone(),
                    created_at: now_rfc3339(),
                    message: OllamaMessage { role: "assistant".into(), content, thinking },
                    done,
                    done_reason: done.then_some("stop".into()),
                };
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
// Streaming conversion: OpenAI SSE → Ollama NDJSON (generate)
// ---------------------------------------------------------------------------

async fn stream_ollama_generate(
    client: Client,
    url: String,
    oai_req: OAIChatRequest,
    model: String,
) -> Result<Response, AppError> {
    let resp = client
        .post(&url)
        .json(&oai_req)
        .send()
        .await
        .context("send to llama-server")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(AppError(anyhow!("inference backend {status}: {body}")));
    }

    let stream = bytes_to_lines(resp.bytes_stream()).map(move |line| {
        let out = line.strip_prefix("data: ")
            .and_then(|p| oai_chunk_to_content(p))
            .map(|(response, thinking, done)| {
                let chunk = OllamaGenerateChunk {
                    model: model.clone(),
                    created_at: now_rfc3339(),
                    response,
                    thinking,
                    done,
                    done_reason: done.then_some("stop".into()),
                };
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
    let resp = client
        .post(&url)
        .json(&oai_req)
        .send()
        .await
        .context("send to llama-server")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(AppError(anyhow!("inference backend {status}: {body}")));
    }

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
    Json(serde_json::json!({ "version": env!("CARGO_PKG_VERSION") }))
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

async fn handle_ps(State(state): State<AppState>) -> impl IntoResponse {
    let mgr = state.0.manager.lock().await;
    let models = mgr
        .running
        .keys()
        .map(|k| OllamaRunningModelInfo {
            name: k.clone(),
            model: k.clone(),
            size: 0,
            size_vram: 0,
        })
        .collect();
    Json(OllamaPsResponse { models })
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
    let model_for_task = model.clone();
    let pull_task = tokio::task::spawn_blocking(move || {
        let layout_dir = store_path
            .to_str()
            .ok_or_else(|| anyhow!("store path is not valid UTF-8"))?;
        crate::ffi::pull(&model_for_task, layout_dir)
    });

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

    let model_for_task = model.clone();
    let push_task = tokio::task::spawn_blocking(move || {
        let layout_dir = store_path
            .to_str()
            .ok_or_else(|| anyhow!("store path is not valid UTF-8"))?;
        crate::ffi::push(layout_dir, &model_for_task)
    });

    stream_ffi_progress(model, "push", "retrieving manifest", push_task).into_response()
}

/// Runs `task` (a blocking FFI call already dispatched via spawn_blocking)
/// to completion, streaming an immediate `first_status` line, then a
/// `{"status": "<verb>ing <model>"}` heartbeat every 2s until it finishes,
/// then a final `{"status": "success"}` or `{"error": ...}` line. Shared by
/// handle_pull and handle_push. Real per-layer progress (Ollama's
/// `digest`/`total`/`completed` fields) isn't available: the Go shim's
/// `llmman_pull`/`llmman_push` are single opaque blocking calls with no
/// progress callback, so this reports coarse status only — every other
/// field on api.ProgressResponse is `omitempty` on the client side.
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
                    _ = sleep(Duration::from_secs(2)) => {
                        let line = serde_json::json!({"status": format!("{verb}ing {model}")}).to_string() + "\n";
                        Some((Bytes::from(line), Some(task)))
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
    stream_ollama_chat(state.0.client.clone(), url, oai, req.model).await
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
    stream_ollama_generate(state.0.client.clone(), url, oai, req.model).await
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



async fn handle_openai_chat(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    let req: serde_json::Value =
        serde_json::from_slice(&body).context("parse OpenAI request body")?;
    let model = req["model"].as_str().unwrap_or("").to_string();
    let port = ensure_model(&state, &model).await?;
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    proxy(&state.0.client, &url, &headers, body).await
}

async fn handle_openai_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    let req: serde_json::Value =
        serde_json::from_slice(&body).context("parse OpenAI request body")?;
    let model = req["model"].as_str().unwrap_or("").to_string();
    let port = ensure_model(&state, &model).await?;
    let url = format!("http://127.0.0.1:{port}/v1/completions");
    proxy(&state.0.client, &url, &headers, body).await
}

async fn handle_openai_embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    let req: serde_json::Value =
        serde_json::from_slice(&body).context("parse OpenAI request body")?;
    let model = req["model"].as_str().unwrap_or("").to_string();
    let port = ensure_model(&state, &model).await?;
    let url = format!("http://127.0.0.1:{port}/v1/embeddings");
    proxy(&state.0.client, &url, &headers, body).await
}

// -- Anthropic /v1/messages --------------------------------------------------

async fn handle_anthropic_messages(
    State(state): State<AppState>,
    Json(req): Json<AnthropicRequest>,
) -> Result<Response, AppError> {
    let port = ensure_model(&state, &req.model).await?;
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");

    let mut messages: Vec<OAIMessage> = Vec::new();
    if let Some(sys) = &req.system {
        messages.push(OAIMessage {
            role: "system".into(),
            content: sys.clone(),
        });
    }
    for m in &req.messages {
        messages.push(OAIMessage {
            role: m.role.clone(),
            content: m.content.as_text(),
        });
    }

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
    let llama_server_bin = resolve_llama_server()?;
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

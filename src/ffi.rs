//! Safe wrappers around the CGO-exported Go shim functions.
//!
//! Every Go function returns a JSON-encoded `{"ok":bool,"data":"...","error":"..."}`.
//! The wrappers decode this envelope and surface Rust `Result`s.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use anyhow::{anyhow, Context};
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Raw FFI declarations — symbols produced by the Go shim static archive
// ---------------------------------------------------------------------------
extern "C" {
    fn llmman_free(s: *mut c_char);
    fn llmman_login(server: *const c_char, username: *const c_char, password: *const c_char)
        -> *mut c_char;
    fn llmman_logout(server: *const c_char) -> *mut c_char;
    fn llmman_push(layout_dir: *const c_char, reference: *const c_char) -> *mut c_char;
    fn llmman_pull(reference: *const c_char, layout_dir: *const c_char) -> *mut c_char;
    fn llmman_inspect(reference: *const c_char) -> *mut c_char;
    fn llmman_transfer(source: *const c_char, destination: *const c_char) -> *mut c_char;
    fn llmman_progress() -> *mut c_char;
}

// ---------------------------------------------------------------------------
// Response envelope (mirrors the Go `response` struct)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct GoResponse {
    ok: bool,
    #[serde(default)]
    data: String,
    #[serde(default)]
    error: String,
}

/// Consume a raw C string returned by the Go shim and decode the JSON envelope.
/// Returns `Ok(data)` on success, `Err(error)` on failure.
fn consume(raw: *mut c_char) -> anyhow::Result<String> {
    assert!(!raw.is_null(), "Go shim returned a null pointer");
    let json_str = unsafe { CStr::from_ptr(raw).to_string_lossy().into_owned() };
    unsafe { llmman_free(raw) };
    let resp: GoResponse =
        serde_json::from_str(&json_str).context("failed to decode Go shim response")?;
    if resp.ok {
        Ok(resp.data)
    } else {
        Err(anyhow!("{}", resp.error))
    }
}

fn cstr(s: &str) -> anyhow::Result<CString> {
    CString::new(s).context("string contains interior NUL byte")
}

// ---------------------------------------------------------------------------
// Safe public API
// ---------------------------------------------------------------------------

/// Store registry credentials.
pub fn login(server: &str, username: &str, password: &str) -> anyhow::Result<()> {
    let s = cstr(server)?;
    let u = cstr(username)?;
    let p = cstr(password)?;
    consume(unsafe { llmman_login(s.as_ptr(), u.as_ptr(), p.as_ptr()) }).map(|_| ())
}

/// Remove stored registry credentials.
pub fn logout(server: &str) -> anyhow::Result<()> {
    let s = cstr(server)?;
    consume(unsafe { llmman_logout(s.as_ptr()) }).map(|_| ())
}

/// Push the image tagged `reference` from `layout_dir` (OCI layout) to a registry.
pub fn push(layout_dir: &str, reference: &str) -> anyhow::Result<()> {
    let l = cstr(layout_dir)?;
    let r = cstr(reference)?;
    consume(unsafe { llmman_push(l.as_ptr(), r.as_ptr()) }).map(|_| ())
}

/// Pull an image from a registry into `layout_dir` (OCI layout).
pub fn pull(reference: &str, layout_dir: &str) -> anyhow::Result<()> {
    let r = cstr(reference)?;
    let l = cstr(layout_dir)?;
    consume(unsafe { llmman_pull(r.as_ptr(), l.as_ptr()) }).map(|_| ())
}

/// Fetch and return the raw manifest JSON for a remote registry reference.
pub fn inspect_remote(reference: &str) -> anyhow::Result<String> {
    let r = cstr(reference)?;
    consume(unsafe { llmman_inspect(r.as_ptr()) })
}

/// Copy an image directly from `source` to `destination` — the `skopeo
/// copy` equivalent — without going through the local store. See
/// go-shim/transfer_docker.go / transfer_podman.go for how each backend
/// implements this (streamed blob-for-blob where possible, exactly like
/// skopeo itself does, falling back to a throwaway local staging
/// directory only for source kinds that transport has no way to stream).
pub fn transfer(source: &str, destination: &str) -> anyhow::Result<()> {
    let s = cstr(source)?;
    let d = cstr(destination)?;
    consume(unsafe { llmman_transfer(s.as_ptr(), d.as_ptr()) }).map(|_| ())
}

/// A byte-level snapshot of whichever pull/push is currently running
/// inside this process, as tracked by the Go shim's `progressState` (see
/// go-shim/progress_state.go). `total`/`completed` are 0 until the shim
/// learns a blob's size and starts transferring it.
#[derive(Deserialize)]
pub struct ProgressSnapshot {
    pub status: String,
    pub total: i64,
    pub completed: i64,
}

/// Polls the Go shim's process-wide pull/push progress snapshot. Called
/// by `cmd::serve` every ~200ms while a `/api/pull` or `/api/push` task is
/// in flight, to relay real byte counts over its NDJSON stream instead of
/// just a coarse "pulling <model>" heartbeat — see llmman_progress's own
/// doc comment for why the daemon can't just let the shim's own mpb bars
/// (go-shim/shared_oci.go) reach an interactive terminal directly.
pub fn progress() -> anyhow::Result<ProgressSnapshot> {
    let data = consume(unsafe { llmman_progress() })?;
    serde_json::from_str(&data).context("failed to decode progress snapshot")
}

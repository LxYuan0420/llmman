#![recursion_limit = "256"]

pub mod cmd;
pub mod container;
pub mod daemon;
pub mod ffi;
pub mod fmt;
pub mod hf;
pub mod hostgpu;
pub mod llama_release;
pub mod modelpack;
pub mod oauth;
pub mod shortnames;
pub mod storage;
pub mod webui;

use std::path::{Path, PathBuf};

/// Return the path to the default local OCI store, or a caller-supplied override.
///
/// Linux and macOS both use `~/.local/share/llmman/store`.
/// Windows uses `%LOCALAPPDATA%\llmman\store`.
pub fn default_store(override_path: Option<&Path>) -> anyhow::Result<PathBuf> {
    if let Some(p) = override_path {
        return Ok(p.to_path_buf());
    }
    #[cfg(not(target_os = "windows"))]
    let base = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("could not determine home directory"))?
        .join(".local")
        .join("share");
    #[cfg(target_os = "windows")]
    let base = dirs::data_local_dir()
        .ok_or_else(|| anyhow::anyhow!("could not determine local data directory"))?;
    Ok(base.join("llmman").join("store"))
}

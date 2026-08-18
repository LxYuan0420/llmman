//! Reusable "resolve a local OCI-store reference to servable model files"
//! logic — the CNCF ModelPack (<https://github.com/modelpack/model-spec>)
//! equivalent of `huggingface_hub.snapshot_download`.
//!
//! Originally private to `cmd::serve` (which uses it to decide whether to
//! spawn `llama-server` or `vllm` as its backend for a given model), this
//! module is `pub` so it also backs `cmd::resolve` (`llmman resolve`) — a
//! standalone, scriptable entry point that other tools (e.g. a vLLM plugin
//! that wants vLLM itself, not `llmman`, to be the one serving the model)
//! can shell out to, without needing `llmman serve`'s HTTP daemon or its
//! opinions about which inference backend to launch.
//!
//! Everything here assumes the reference has already been pulled into the
//! local `OciStore` at `store_path` (see `crate::ffi::pull`) — this module
//! only resolves+extracts, it never talks to a registry itself.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context};

use crate::storage::OciStore;

const HF_GGUF_MEDIA_TYPE: &str = "application/vnd.docker.ai.gguf.v3";

/// What kind of model did we find in the OCI store?
pub enum ModelPath {
    /// A GGUF file — serve with llama-server.
    Gguf(PathBuf),
    /// A safetensors directory — serve with vllm.
    SafeTensors(PathBuf),
}

impl ModelPath {
    /// The local filesystem path this variant resolved to — either a
    /// single `.gguf` file, or the model directory (parent of
    /// `config.json`) for a safetensors checkout.
    pub fn path(&self) -> &Path {
        match self {
            ModelPath::Gguf(p) => p,
            ModelPath::SafeTensors(p) => p,
        }
    }

    /// A short, stable string identifying which variant this is — used by
    /// `cmd::resolve`'s JSON output and any other consumer that wants to
    /// branch on format without matching the enum directly.
    pub fn format(&self) -> &'static str {
        match self {
            ModelPath::Gguf(_) => "gguf",
            ModelPath::SafeTensors(_) => "safetensors",
        }
    }
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

// gguf_architecture/gguf_context_length_override (a GGUF metadata reader
// + --override-kv builder that let --ctx-size force a context above a
// model's own trained length) were tried and removed: llama-server's own
// capping of --ctx-size back down to a model's trained context — see
// ServeArgs::ctx_size's doc comment — is deliberate, matching behavior
// Ollama's own server independently implements (see llm/server.go in
// ollama/ollama: it clamps num_ctx to n_ctx_train with a warning, with no
// override of any kind). Defeating that safety net via --override-kv
// produces the same NaN/incoherent-output risk for out-of-distribution
// RoPE positions that both llama-server's warning and Ollama's clamp
// exist to prevent, for a use case (fitting a real coding agent's system
// prompt) that a model whose trained context is that tight was never
// going to serve well regardless — see docker/sandboxes' own
// llmmanCtxSize doc comment for the model-selection fix that replaced
// this instead.

/// Resolve `model_ref` (already present in the `OciStore` at `store_path`)
/// to either a `.gguf` file or an extracted safetensors directory, caching
/// any extraction under `cache_path`.
pub fn resolve_model(store_path: &Path, cache_path: &Path, model_ref: &str) -> anyhow::Result<ModelPath> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_matches_variant() {
        assert_eq!(ModelPath::Gguf(PathBuf::from("/x/m.gguf")).format(), "gguf");
        assert_eq!(ModelPath::SafeTensors(PathBuf::from("/x")).format(), "safetensors");
    }

    #[test]
    fn path_returns_inner_pathbuf() {
        let p = ModelPath::SafeTensors(PathBuf::from("/models/foo"));
        assert_eq!(p.path(), Path::new("/models/foo"));
    }
}

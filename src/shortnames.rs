//! Short-name alias resolution — loaded from config files at runtime.
//!
//! Mirrors podman's approach: TOML files are read from a priority-ordered set
//! of locations; all files are merged with higher-priority entries winning.
//! Nothing is compiled into the binary.
//!
//! Search order (ascending priority — later files override earlier ones):
//!   1. /usr/share/llmman/shortnames.conf          distro / package default
//!   2. /etc/llmman/shortnames.conf                 system-admin override
//!   3. <binary>/../share/llmman/shortnames.conf    install-tree relative path
//!   4. <binary-dir>/shortnames.conf                development (conf beside binary)
//!   5. ~/.config/llmman/shortnames.conf            per-user aliases
//!   6. $LLMMAN_SHORTNAMES_CONF                     env-var override

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use serde::Deserialize;

#[derive(Deserialize, Default)]
struct Conf {
    #[serde(default)]
    aliases: HashMap<String, String>,
}

/// Return all candidate config-file paths in ascending priority order.
fn config_paths() -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = vec![
        PathBuf::from("/usr/share/llmman/shortnames.conf"),
        PathBuf::from("/etc/llmman/shortnames.conf"),
    ];

    // Paths relative to the running binary.
    if let Ok(exe) = std::env::current_exe() {
        // <binary>/../share/llmman/shortnames.conf  (standard install layout)
        if let Some(parent) = exe.parent() {
            paths.push(parent.join("../share/llmman/shortnames.conf"));
            // <binary-dir>/shortnames.conf  (development: cargo run / direct exec)
            paths.push(parent.join("shortnames.conf"));
        }
    }

    // ~/.config/llmman/shortnames.conf
    if let Some(cfg) = dirs::config_dir() {
        paths.push(cfg.join("llmman").join("shortnames.conf"));
    }

    // $LLMMAN_SHORTNAMES_CONF
    if let Ok(env) = std::env::var("LLMMAN_SHORTNAMES_CONF") {
        if !env.is_empty() {
            paths.push(PathBuf::from(env));
        }
    }

    paths
}

/// Load and merge aliases from all config files.
/// Higher-priority files (later in the list) override lower-priority ones.
fn load_aliases() -> HashMap<String, String> {
    let mut merged: HashMap<String, String> = HashMap::new();
    for path in config_paths() {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        match toml::from_str::<Conf>(&text) {
            Ok(conf) => {
                for (k, v) in conf.aliases {
                    merged.insert(k, v);
                }
            }
            Err(e) => {
                eprintln!("[llmman] warning: ignoring {}: {e}", path.display());
            }
        }
    }
    merged
}

fn aliases() -> &'static HashMap<String, String> {
    static CACHE: OnceLock<HashMap<String, String>> = OnceLock::new();
    CACHE.get_or_init(load_aliases)
}

/// Returns true if `reference` already carries an explicit registry host
/// (the first path component contains a dot or equals "localhost").
fn has_host(reference: &str) -> bool {
    let first = reference.split('/').next().unwrap_or("");
    first.contains('.') || first.eq_ignore_ascii_case("localhost")
}

/// Resolve `reference` through the short-name alias table, then default the
/// registry to `hf.co` when no host is present.
///
/// URI scheme handling (processed before alias lookup):
///   hf:// huggingface://  → strip scheme, continue as bare owner/repo
///   ms:// modelscope://   → normalise to ms:// (Go shim routes to ModelScope)
///   ngc:// s3:// gs://    → pass through verbatim (Go shim handles natively)
///   /absolute/path        → pass through verbatim (local directory import)
///
/// Resolution order for everything else:
///   1. Exact alias match  → return the mapped value
///   2. Has a registry host → return as-is
///   3. No host            → prepend `hf.co/`
pub fn resolve(reference: &str) -> String {
    // ── URI schemes that bypass alias lookup and hf.co defaulting ─────────
    // Local absolute paths and object-store URIs are forwarded as-is to the
    // Go shim which dispatches them to the appropriate source handler.
    for passthrough in &["ngc://", "s3://", "gs://"] {
        if reference.starts_with(passthrough) {
            return reference.to_owned();
        }
    }
    if reference.starts_with('/') {
        return reference.to_owned();
    }

    // ── Normalise well-known URI schemes to canonical form ─────────────────
    // hf:// and huggingface:// are stripped; the remainder is treated as a
    // bare HuggingFace owner/repo reference through the normal path below.
    let reference = if let Some(r) = reference
        .strip_prefix("hf://")
        .or_else(|| reference.strip_prefix("huggingface://"))
    {
        r
    }
    // ms:// and modelscope:// are normalised to ms:// so the Go shim can
    // detect the scheme and route to the ModelScope download path.
    else if let Some(r) = reference.strip_prefix("modelscope://") {
        return format!("ms://{r}");
    } else if reference.starts_with("ms://") {
        return reference.to_owned();
    } else {
        reference
    };

    // ── Alias lookup → hf.co default ──────────────────────────────────────
    if let Some(mapped) = aliases().get(reference) {
        return mapped.clone();
    }
    if has_host(reference) {
        return reference.to_owned();
    }
    format!("hf.co/{reference}")
}

/// Returns true if `reference` is *completely* bare: no "/" (no owner/repo
/// or registry-host structure) and no "." (no host-like dot and no
/// dotted-version tag such as "3.5"). This is deliberately stricter than
/// `has_host` — a HuggingFace-style "owner/repo" reference has a "/" but no
/// dot, and must NOT be treated as bare here.
fn is_bare(reference: &str) -> bool {
    !reference.contains('/') && !reference.contains('.')
}

/// Resolve `reference` the way every Ollama-API-facing path in `cmd::serve`
/// does (handle_pull, handle_show, handle_delete, ensure_model, and the
/// `--model` preload in serve_async): identical to `resolve`, except a
/// *completely bare* reference — e.g. "gemma4", no "/" and no "." anywhere —
/// defaults to Docker's official curated-model namespace on Docker Hub,
/// `docker.io/ai/<reference>` (e.g. "gemma4" -> "docker.io/ai/gemma4"),
/// instead of `resolve`'s general `hf.co/<reference>` default. Any
/// reference with more structure than that (an owner/repo path, a URI
/// scheme, an explicit host, a dotted version like "gemma4:3.5") is left to
/// `resolve`'s normal rules unchanged.
///
/// CLI subcommands that talk to a local server over the Ollama API (pull,
/// push) go through this same resolution server-side, so the docker.io/ai/
/// default is consistent regardless of whether a bare name reaches llmman
/// via the CLI or directly over HTTP.
pub fn resolve_ollama_api(reference: &str) -> String {
    if is_bare(reference) {
        if let Some(mapped) = aliases().get(reference) {
            return mapped.clone();
        }
        return format!("docker.io/ai/{reference}");
    }
    resolve(reference)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_ollama_api_defaults_bare_names_to_docker_ai() {
        assert_eq!(resolve_ollama_api("gemma4"), "docker.io/ai/gemma4");
        // A tag with no dot is still "bare" by this rule (only "/" and "."
        // disqualify it) — matches the ai/<name>:<tag> shape on Docker Hub.
        assert_eq!(resolve_ollama_api("gemma4:e4b"), "docker.io/ai/gemma4:e4b");
    }

    #[test]
    fn resolve_ollama_api_leaves_structured_references_to_resolve() {
        // Owner/repo (has a "/") falls back to resolve()'s hf.co default.
        assert_eq!(resolve_ollama_api("unsloth/Qwen3.5-0.8B-GGUF"), resolve("unsloth/Qwen3.5-0.8B-GGUF"));
        // Already has an explicit host.
        assert_eq!(resolve_ollama_api("hf.co/foo/bar"), "hf.co/foo/bar");
        assert_eq!(resolve_ollama_api("docker.io/ai/gemma4"), "docker.io/ai/gemma4");
        // A dot (e.g. a dotted version number) disqualifies "bare" even
        // without a "/".
        assert_eq!(resolve_ollama_api("qwen3.5"), resolve("qwen3.5"));
    }

    #[test]
    fn resolve_ollama_api_matches_resolve_for_uri_schemes_and_paths() {
        assert_eq!(resolve_ollama_api("hf://unsloth/Qwen3.5-0.8B-GGUF"), resolve("hf://unsloth/Qwen3.5-0.8B-GGUF"));
        assert_eq!(resolve_ollama_api("/abs/path/model.gguf"), "/abs/path/model.gguf");
    }

    #[test]
    fn is_bare_rejects_slashes_and_dots() {
        assert!(is_bare("gemma4"));
        assert!(is_bare("gemma4:e4b"));
        assert!(!is_bare("unsloth/gemma4"));
        assert!(!is_bare("qwen3.5"));
        assert!(!is_bare("hf.co/gemma4"));
    }
}

//! `llmman launch` — launch AI coding-assistant integrations backed by llmman serve.
//!
//! Mirrors `ollama launch`: sets integration-specific environment variables
//! pointing at the local inference server, then exec's the integration binary.

use std::io::{self, IsTerminal};
use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;
use clap::Args;

use crate::daemon::SERVER;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Args, Debug)]
pub struct LaunchArgs {
    /// Integration to launch (claude, opencode, codex, cline, aider, …)
    /// Omit to list available integrations.
    #[arg(value_name = "INTEGRATION")]
    pub integration: Option<String>,

    /// Model to use
    #[arg(long, short, value_name = "MODEL")]
    pub model: Option<String>,

    /// Extra arguments forwarded to the integration binary (after --)
    #[arg(last = true, value_name = "ARGS")]
    pub extra_args: Vec<String>,
}

pub fn run(args: &LaunchArgs) -> anyhow::Result<()> {
    let Some(ref name) = args.integration else {
        print_integrations();
        return Ok(());
    };

    // resolve_ollama_api, not resolve: every integration this launches talks
    // to serve's Ollama/OpenAI/Anthropic-compat surfaces, all of which
    // resolve model names the same way (see ensure_model in cmd::serve), so
    // a bare name here must match what the daemon resolves it to at
    // request time.
    let model = args
        .model
        .as_deref()
        .map(|m| crate::shortnames::resolve_ollama_api(m))
        .unwrap_or_default();

    // Ensure serve is running (start it in background if needed), preloading
    // the requested model so the integration's first request finds it warm.
    crate::daemon::ensure_server(&model)?;

    // serve's preload above is fire-and-forget and only fires on a cold
    // `serve` start (see run() in cmd/serve.rs) — if the daemon was already
    // running from a previous invocation, a missing model would otherwise
    // only surface as an opaque failure once the integration made its first
    // request. Mirror `llmman run`'s behavior and pull it here instead,
    // synchronously and with progress, before ever handing off to the
    // integration.
    if !model.is_empty() {
        crate::daemon::ensure_model_pulled(&model)?;
    }

    launch(name, &model, &args.extra_args)
}

// ---------------------------------------------------------------------------
// Integration registry
// ---------------------------------------------------------------------------

struct Integration {
    name: &'static str,
    description: &'static str,
    binary: &'static str,
    install_hint: &'static str,
}

const INTEGRATIONS: &[Integration] = &[
    Integration {
        name: "claude",
        description: "Claude Code",
        binary: "claude",
        install_hint: "https://code.claude.com/docs/en/quickstart",
    },
    Integration {
        name: "opencode",
        description: "OpenCode",
        binary: "opencode",
        install_hint: "https://opencode.ai",
    },
    Integration {
        name: "codex",
        description: "OpenAI Codex CLI",
        binary: "codex",
        install_hint: "npm install -g @openai/codex",
    },
    Integration {
        name: "cline",
        description: "Cline",
        binary: "cline",
        install_hint: "npm install -g cline",
    },
    Integration {
        name: "aider",
        description: "Aider AI pair programmer",
        binary: "aider",
        install_hint: "pip install aider-install && aider-install",
    },
    Integration {
        name: "copilot",
        description: "GitHub Copilot CLI",
        binary: "gh",
        install_hint: "https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli",
    },
    Integration {
        name: "kimi",
        description: "Kimi Code CLI",
        binary: "kimi",
        install_hint: "https://kimi.ai",
    },
    Integration {
        name: "gemini",
        description: "Gemini CLI",
        binary: "gemini",
        install_hint: "npm install -g @google/gemini-cli",
    },
];

fn print_integrations() {
    println!("Available integrations:\n");
    for i in INTEGRATIONS {
        if find_on_path(i.binary).is_some() {
            println!("  {:<12} {}", i.name, i.description);
        } else {
            println!("  {:<12} {} (not installed — {})", i.name, i.description, i.install_hint);
        }
    }
    println!("\nUsage: llmman launch <integration> [--model <model>]");
}

fn find_on_path(binary: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = if cfg!(windows) {
            dir.join(format!("{binary}.exe"))
        } else {
            dir.join(binary)
        };
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Launch dispatcher
// ---------------------------------------------------------------------------

fn launch(name: &str, model: &str, extra_args: &[String]) -> anyhow::Result<()> {
    match name.to_lowercase().as_str() {
        "claude" => launch_claude(model, extra_args),
        "opencode" => launch_opencode(model, extra_args),
        "codex" => launch_codex(model, extra_args),
        "cline" => launch_simple("cline", "cline is not installed: npm install -g cline", model, extra_args),
        "aider" => launch_aider(model, extra_args),
        "copilot" | "copilot-cli" => launch_copilot(model, extra_args),
        "kimi" => launch_simple("kimi", "kimi is not installed: https://kimi.ai", model, extra_args),
        "gemini" => launch_gemini(model, extra_args),
        other => anyhow::bail!(
            "unknown integration {:?}\nRun 'llmman launch' without arguments to list supported integrations.",
            other
        ),
    }
}

// ---------------------------------------------------------------------------
// Per-integration launchers
// ---------------------------------------------------------------------------

/// claude: set ANTHROPIC_BASE_URL and a dummy ANTHROPIC_API_KEY so it talks to
/// our server's Anthropic-compatible API.
fn launch_claude(model: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin = find_on_path("claude")
        .ok_or_else(|| anyhow::anyhow!("claude is not installed — https://code.claude.com/docs/en/quickstart"))?;

    let mut args: Vec<String> = Vec::new();
    if !model.is_empty() {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    args.extend_from_slice(extra_args);

    exec_with_env(
        &bin,
        &args,
        &[
            ("ANTHROPIC_BASE_URL", SERVER),
            ("ANTHROPIC_API_KEY", "llmman"),
        ],
    )
}

/// opencode: pass a JSON config via OPENCODE_CONFIG_CONTENT pointing at our
/// /v1 endpoint, matching exactly what ollama launch does.
fn launch_opencode(model: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin = find_on_path("opencode").or_else(|| {
        dirs::home_dir().and_then(|h| {
            let p = h.join(".opencode").join("bin").join("opencode");
            p.exists().then_some(p)
        })
    });
    let bin = bin.ok_or_else(|| {
        anyhow::anyhow!("opencode is not installed — https://opencode.ai")
    })?;

    let effective_model = if model.is_empty() { "default" } else { model };
    let config = opencode_config(effective_model);

    exec_with_env(
        &bin,
        extra_args,
        &[("OPENCODE_CONFIG_CONTENT", &config)],
    )
}

fn opencode_config(model: &str) -> String {
    let base_url = format!("{SERVER}/v1");
    serde_json::json!({
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "ollama": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Ollama",
                "options": {
                    "baseURL": base_url
                },
                "models": {
                    model: { "name": model }
                }
            }
        },
        "model": format!("ollama/{model}")
    })
    .to_string()
}

/// codex: set OPENAI_API_KEY=llmman and write ~/.codex/config.toml with the
/// ollama provider pointing at our /v1 endpoint.
fn launch_codex(model: &str, extra_args: &[String]) -> anyhow::Result<()> {
    // Write codex config
    write_codex_config()?;

    let mut args: Vec<String> = Vec::new();
    if !model.is_empty() {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    // codex profile flag
    args.extend(["--profile".to_string(), "llmman".to_string()]);
    args.extend_from_slice(extra_args);

    exec_with_env(
        &PathBuf::from("codex"),
        &args,
        &[("OPENAI_API_KEY", "llmman")],
    )
}

/// Writes codex's `llmman` profile.
///
/// Codex 0.134+ dropped support for `--profile <name>` reading a
/// `[profiles.<name>]` table out of `config.toml`: it now only overlays a
/// sibling `~/.codex/<name>.config.toml`, using top-level keys instead of a
/// `[profiles.<name>]` wrapper (see
/// <https://developers.openai.com/codex/config-advanced#profiles>). An
/// older llmman wrote the now-unsupported `[profiles.llmman]` form directly
/// into `config.toml`, which current codex refuses to start with at all
/// ("cannot be used while config.toml contains legacy ... table") — so any
/// leftover copy of that table is stripped from `config.toml` first, then
/// the real settings are (re)written to the profile overlay file codex
/// actually reads.
fn write_codex_config() -> anyhow::Result<()> {
    let home = dirs::home_dir().context("no home directory")?;
    let config_dir = home.join(".codex");
    std::fs::create_dir_all(&config_dir)?;

    let config_path = config_dir.join("config.toml");
    if let Ok(existing) = std::fs::read_to_string(&config_path) {
        if existing.contains("[profiles.llmman]") {
            std::fs::write(&config_path, strip_legacy_llmman_profile(&existing))?;
        }
    }

    let profile_path = config_dir.join("llmman.config.toml");
    let contents = format!("openai_base_url = \"{SERVER}/v1\"\n");
    // Avoid rewriting (and bumping the mtime of) a file that's already
    // correct.
    if std::fs::read_to_string(&profile_path).ok().as_deref() != Some(contents.as_str()) {
        std::fs::write(&profile_path, contents)?;
    }
    Ok(())
}

/// Removes a `[profiles.llmman]` table (and everything up to the next
/// top-level `[...]` header or end of file) from `config.toml`'s text —
/// the shape an older llmman wrote there, now rejected by current codex.
/// Line-based rather than a real TOML parser: this only ever needs to
/// undo llmman's own prior output, not handle arbitrary user TOML.
fn strip_legacy_llmman_profile(existing: &str) -> String {
    let mut out = String::new();
    let mut skipping = false;
    for line in existing.lines() {
        let trimmed = line.trim();
        if trimmed == "[profiles.llmman]" {
            skipping = true;
            continue;
        }
        if skipping && trimmed.starts_with('[') {
            skipping = false;
        }
        if skipping {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// aider: set OPENAI_API_KEY and OPENAI_BASE_URL.
fn launch_aider(model: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let mut args: Vec<String> = Vec::new();
    if !model.is_empty() {
        args.extend(["--model".to_string(), format!("openai/{model}")]);
    }
    args.extend(["--openai-api-base".to_string(), format!("{SERVER}/v1")]);
    args.extend_from_slice(extra_args);

    exec_with_env(
        &PathBuf::from("aider"),
        &args,
        &[
            ("OPENAI_API_KEY", "llmman"),
            ("OPENAI_BASE_URL", &format!("{SERVER}/v1")),
        ],
    )
}

/// copilot: passes COPILOT_PROVIDER_BASE_URL via env.
fn launch_copilot(model: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin = find_on_path("gh")
        .ok_or_else(|| anyhow::anyhow!("gh (GitHub CLI) is not installed — https://cli.github.com"))?;

    let base_url = format!("{SERVER}/v1");
    let mut args = vec!["copilot".to_string()];
    if !model.is_empty() {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    args.extend_from_slice(extra_args);

    exec_with_env(
        &bin,
        &args,
        &[("COPILOT_PROVIDER_BASE_URL", &base_url)],
    )
}

/// gemini: set GOOGLE_GENAI_BASE_URL pointing at our Anthropic-compatible endpoint.
fn launch_gemini(model: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin = find_on_path("gemini")
        .ok_or_else(|| anyhow::anyhow!("gemini is not installed — npm install -g @google/gemini-cli"))?;

    let mut args: Vec<String> = Vec::new();
    if !model.is_empty() {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    args.extend_from_slice(extra_args);

    exec_with_env(
        &bin,
        &args,
        &[
            ("GEMINI_BASE_URL", &format!("{SERVER}/v1")),
            ("GEMINI_API_KEY", "llmman"),
        ],
    )
}

/// Generic launcher: just set OLLAMA_HOST and run the binary.
fn launch_simple(binary: &str, install_hint: &str, _model: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin = find_on_path(binary)
        .ok_or_else(|| anyhow::anyhow!("{binary} is not installed — {install_hint}"))?;
    exec_with_env(&bin, extra_args, &[("OLLAMA_HOST", SERVER)])
}

// ---------------------------------------------------------------------------
// Process execution helper
// ---------------------------------------------------------------------------

fn exec_with_env(
    bin: &PathBuf,
    args: &[String],
    extra_env: &[(&str, &str)],
) -> anyhow::Result<()> {
    let mut cmd = Command::new(bin);
    cmd.args(args);
    cmd.stdin(std::process::Stdio::inherit());
    cmd.stdout(std::process::Stdio::inherit());
    cmd.stderr(std::process::Stdio::inherit());

    // Inherit the current environment and overlay OLLAMA_HOST + integration vars.
    let mut env: std::collections::HashMap<String, String> = std::env::vars().collect();
    env.insert("OLLAMA_HOST".to_string(), SERVER.to_string());
    for (k, v) in extra_env {
        env.insert(k.to_string(), v.to_string());
    }
    cmd.envs(&env);

    let status = cmd.status()
        .with_context(|| format!("failed to run {}", bin.display()))?;

    std::process::exit(status.code().unwrap_or(1));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the codex config bug described on
    /// `write_codex_config`'s own doc comment: an older llmman's
    /// `[profiles.llmman]` table (a format current codex refuses to load
    /// at all) must be fully removed, leaving everything else in
    /// `config.toml` untouched.
    #[test]
    fn strip_legacy_llmman_profile_removes_only_that_table() {
        let existing = "\
[some_other_setting]
foo = \"bar\"

[profiles.llmman]
openai_base_url = \"http://127.0.0.1:17434/v1\"

[profiles.other]
model = \"gpt-5\"
";
        let cleaned = strip_legacy_llmman_profile(existing);
        assert!(!cleaned.contains("[profiles.llmman]"));
        assert!(!cleaned.contains("openai_base_url"));
        assert!(cleaned.contains("[some_other_setting]"));
        assert!(cleaned.contains("foo = \"bar\""));
        assert!(cleaned.contains("[profiles.other]"));
        assert!(cleaned.contains("model = \"gpt-5\""));
    }

    #[test]
    fn strip_legacy_llmman_profile_is_a_no_op_without_the_legacy_table() {
        let existing = "[profiles.other]\nmodel = \"gpt-5\"\n";
        assert_eq!(strip_legacy_llmman_profile(existing), existing);
    }

    #[test]
    fn strip_legacy_llmman_profile_handles_the_table_at_end_of_file() {
        let existing = "[profiles.llmman]\nopenai_base_url = \"http://127.0.0.1:17434/v1\"\n";
        assert_eq!(strip_legacy_llmman_profile(existing), "");
    }
}

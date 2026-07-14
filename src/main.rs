#![recursion_limit = "256"]

mod cmd;
mod container;
mod daemon;
mod ffi;
mod fmt;
mod shortnames;
mod storage;
pub mod webui;

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "llmman",
    about = "LLM model image manager",
    version,
    propagate_version = true
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Launch an integration
    Launch(cmd::launch::LaunchArgs),
    /// Run a model interactively or with a one-shot prompt
    Run(cmd::run::RunArgs),
    /// Package model files into a local OCI image
    Build(cmd::build::BuildArgs),
    /// Log in to a container registry
    Login(cmd::login::LoginArgs),
    /// Log out from a container registry
    Logout(cmd::logout::LogoutArgs),
    /// Push a local image to a registry
    Push(cmd::push::PushArgs),
    /// Pull an image from a registry to the local store
    Pull(cmd::pull::PullArgs),
    /// List locally stored images
    #[command(alias = "ls")]
    List(cmd::list::ListArgs),
    /// List models currently loaded by a running `llmman serve`
    Ps(cmd::ps::PsArgs),
    /// Remove a local image
    Rm(cmd::rm::RmArgs),
    /// Show the manifest of a local (or remote with --remote) image
    Inspect(cmd::inspect::InspectArgs),
    /// Start an inference server (Ollama, OpenAI, Anthropic compatible APIs)
    Serve(cmd::serve::ServeArgs),
    /// Create a new local tag pointing to an existing image
    Tag(cmd::tag::TagArgs),
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    let result = match &cli.command {
        Commands::Launch(a)  => cmd::launch::run(a),
        Commands::Run(a)     => cmd::run::run(a),
        Commands::Build(a)   => cmd::build::run(a),
        Commands::Login(a)   => cmd::login::run(a),
        Commands::Logout(a)  => cmd::logout::run(a),
        Commands::Push(a)    => cmd::push::run(a),
        Commands::Pull(a)    => cmd::pull::run(a),
        Commands::List(a)    => cmd::list::run(a),
        Commands::Ps(a)      => cmd::ps::run(a),
        Commands::Rm(a)      => cmd::rm::run(a),
        Commands::Inspect(a) => cmd::inspect::run(a),
        Commands::Serve(a)   => cmd::serve::run(a),
        Commands::Tag(a)     => cmd::tag::run(a),
    };
    if let Err(e) = result {
        eprintln!("Error: {:#}", e);
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

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

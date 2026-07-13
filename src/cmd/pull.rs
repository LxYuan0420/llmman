use clap::Args;

#[derive(Args, Debug)]
pub struct PullArgs {
    /// Registry reference to pull (e.g. registry.example.com/mymodel:latest)
    #[arg(value_name = "REFERENCE")]
    pub reference: String,
}

/// `llmman pull` is a thin client of the local daemon's Ollama-protocol
/// /api/pull (starting one, left running afterwards, if none is running
/// yet — see daemon::ensure_server) — the same wire protocol `sbx` and any
/// other Ollama-API client use, so bare-name resolution (shortnames::
/// resolve_ollama_api) and the model store are always the daemon's.
///
/// This intentionally has no `--store` override anymore: the daemon always
/// uses its own default store (see `llmman serve --store` to change that
/// store for the daemon itself, which then applies to every client).
pub fn run(args: &PullArgs) -> anyhow::Result<()> {
    crate::daemon::ensure_server("")?;
    crate::daemon::stream_progress("/api/pull", &args.reference)?;
    println!("Pulled {}", args.reference);
    Ok(())
}

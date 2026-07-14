use clap::Args;
use serde::Deserialize;

use crate::fmt::{human_size, relative_time_rfc3339, short_id};

#[derive(Args, Debug)]
pub struct PsArgs {
    /// Only show models whose reference starts with this prefix
    #[arg(value_name = "PREFIX")]
    pub prefix: Option<String>,
}

/// Wire shape of GET /api/ps's response — see cmd::serve's
/// `OllamaRunningModelInfo`/`OllamaPsResponse` for the server side these
/// field names must match. Deliberately a separate type (not shared via a
/// common module) rather than reusing the daemon's response, mirroring how
/// none of the other CLI commands (pull/push/run) share request/response
/// types with the daemon either — the two sides only need to agree on the
/// JSON shape, not a Rust type.
#[derive(Debug, Deserialize)]
struct PsResponse {
    models: Vec<PsModel>,
}

#[derive(Debug, Deserialize)]
struct PsModel {
    name: String,
    digest: String,
    size: u64,
    processor: String,
    context_length: Option<u64>,
    started_at: String,
}

/// `llmman ps` — list models currently loaded by a running `llmman serve`,
/// mirroring `ollama ps`'s NAME/ID/SIZE table. Unlike `ollama ps`, there is
/// no PROCESSOR "N%/N% CPU/GPU" split (llmman's local engines don't report
/// VRAM usage back to llmman — see cmd::serve::RunningModel::processor's
/// doc comment) and no UNTIL/keep-alive expiry column (llmman has no
/// idle-unload timer yet — models stay loaded until an explicit unload
/// request or `llmman serve` itself exits). PROCESSOR instead shows which
/// engine loaded the model, and STARTED replaces UNTIL with how long ago
/// it did.
///
/// Unlike `pull`/`push`/`run`, this does not start `llmman serve` if it
/// isn't already running — matching `ollama ps`'s own `checkServerHeartbeat`
/// precondition, since if there's no daemon there's nothing running to list.
pub fn run(args: &PsArgs) -> anyhow::Result<()> {
    if !crate::daemon::server_alive() {
        anyhow::bail!("llmman serve is not running (nothing is loaded) — start it with `llmman serve`");
    }

    let resp: PsResponse = crate::daemon::get_json("/api/ps")?;
    let mut models: Vec<_> = resp
        .models
        .into_iter()
        .filter(|m| args.prefix.as_deref().is_none_or(|p| m.name.starts_with(p)))
        .collect();
    models.sort_by(|a, b| a.name.cmp(&b.name));

    // Always print the header, even with zero rows — matches `ollama ps`,
    // which unconditionally renders its table (unlike `llmman list`, which
    // prints nothing at all for an empty store).

    let name_w = models.iter().map(|m| m.name.len()).max().unwrap_or(4).max(4);
    let proc_w = models.iter().map(|m| m.processor.len()).max().unwrap_or(9).max(9);

    println!(
        "{:<name_w$}    {:<12}    {:<10}    {:<proc_w$}    {:<9}    STARTED",
        "NAME", "ID", "SIZE", "PROCESSOR", "CONTEXT",
        name_w = name_w,
        proc_w = proc_w,
    );

    for m in &models {
        println!(
            "{:<name_w$}    {:<12}    {:<10}    {:<proc_w$}    {:<9}    {}",
            m.name,
            short_id(&m.digest),
            human_size(m.size),
            m.processor,
            m.context_length.map(|c| c.to_string()).unwrap_or_default(),
            relative_time_rfc3339(&m.started_at),
            name_w = name_w,
            proc_w = proc_w,
        );
    }
    Ok(())
}

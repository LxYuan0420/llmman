//! `llmman bench` — measure prefill/decode throughput for one or more
//! locally-served models, the llmman equivalent of Ollama's own
//! `cmd/bench` tool (`ollama-bench`; see that tool's README for the same
//! prefill/generate/ttft/total vocabulary this mirrors) — a first-class
//! subcommand here instead of a separate binary you'd have to build out
//! of tree.
//!
//! Talks to the already-running (or freshly started, via
//! `daemon::ensure_server`) `llmman serve` daemon exactly like any other
//! OpenAI API client would: `POST /v1/chat/completions` with
//! `stream_options.include_usage`. `proxy_openai`/`proxy` in `cmd::serve`
//! forward both the request and the streamed response byte-for-byte to
//! the backend `llama-server`, so its own final `usage` object
//! (`prompt_tokens`/`completion_tokens`) gives real prefill/decode token
//! counts here without needing any change to the daemon's response shape
//! at all — this file only measures wall-clock time client-side and
//! reads the token counts `llama-server` already reports.

use std::time::{Duration, Instant};

use anyhow::Context;
use clap::Args;
use futures::TryStreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncBufReadExt;

use crate::daemon::SERVER;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Default prompt, mirroring `ollama-bench`'s own "default story prompt" —
/// long/open-ended enough to reliably fill `--max-tokens` of generation
/// instead of stopping early on a short factual answer.
const DEFAULT_PROMPT: &str =
    "Write a detailed short story about a robot exploring an abandoned space station.";

#[derive(Args, Debug)]
pub struct BenchArgs {
    /// Model(s) to benchmark. Repeat the flag or pass a comma-separated
    /// list (e.g. `-m gemma3,qwen3`) to compare more than one in a single
    /// run — each is benchmarked in turn, one full result table row (or,
    /// with --format csv, one row) per model.
    #[arg(
        short = 'm',
        long = "model",
        value_name = "MODEL",
        value_delimiter = ',',
        required = true
    )]
    pub model: Vec<String>,

    /// Prompt text sent on every request. Ignored when --prompt-tokens is
    /// non-zero.
    #[arg(short = 'p', long, default_value = DEFAULT_PROMPT)]
    pub prompt: String,

    /// Build a synthetic filler prompt targeting approximately this many
    /// tokens instead of --prompt, for a prefill length that's
    /// reproducible independent of any particular prompt's wording. 0 (the
    /// default) uses --prompt as-is. This is only a target: the real
    /// prefill token count actually measured (`usage.prompt_tokens` from
    /// the backend) is what gets reported, not this estimate.
    #[arg(long, default_value_t = 0, value_name = "N")]
    pub prompt_tokens: u32,

    /// Maximum tokens to generate per request.
    #[arg(long, default_value_t = 200, value_name = "N")]
    pub max_tokens: u32,

    /// Timed iterations per model, averaged in the reported result —
    /// more epochs trade a longer run for a result less skewed by any one
    /// request's jitter.
    #[arg(long, default_value_t = 3, value_name = "N")]
    pub epochs: u32,

    /// Untimed requests sent before the timed epochs, letting a cold
    /// model finish loading (and any first-request-only backend costs
    /// settle) without that one-time cost skewing the timed results.
    #[arg(long, default_value_t = 1, value_name = "N")]
    pub warmup: u32,

    /// Sampling temperature. Defaults to 0 (greedy) for reproducible
    /// generation length/timing across epochs.
    #[arg(long, default_value_t = 0.0, value_name = "N")]
    pub temperature: f32,

    /// Per-request timeout, in seconds.
    #[arg(long, default_value_t = 300, value_name = "SECONDS")]
    pub timeout: u64,

    /// Output format: `text` (aligned table) or `csv`.
    #[arg(long, default_value = "text", value_name = "text|csv")]
    pub format: String,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run(args: &BenchArgs) -> anyhow::Result<()> {
    if !matches!(args.format.as_str(), "text" | "csv") {
        anyhow::bail!(
            "--format must be \"text\" or \"csv\" (got {:?})",
            args.format
        );
    }
    if args.epochs == 0 {
        // Sample::mean's `len().max(1)` divisor otherwise turns 0 timed
        // epochs into a silent, misleadingly "successful" all-zero
        // result instead of measuring nothing at all.
        anyhow::bail!("--epochs must be at least 1");
    }

    // See run::run's own doc comment on why "" (no preload) here: this
    // command loads whichever model(s) it's about to benchmark itself,
    // right below, so a daemon it starts shouldn't look pinned to just
    // the first one.
    crate::daemon::ensure_server("")?;

    let prompt = if args.prompt_tokens > 0 {
        synthetic_prompt(args.prompt_tokens)
    } else {
        args.prompt.clone()
    };

    let rt = tokio::runtime::Runtime::new().context("start tokio runtime")?;
    let client = Client::builder()
        .timeout(Duration::from_secs(args.timeout))
        .build()
        .context("build http client")?;

    let mut results = Vec::with_capacity(args.model.len());
    for raw_model in &args.model {
        let model = crate::shortnames::resolve_ollama_api(raw_model);
        // Fail fast on a bad/unresolvable reference — mirrors run::run's
        // own use of this same helper — rather than only discovering it
        // partway through warmup below.
        crate::daemon::ensure_model_pulled(&model)?;

        eprintln!(
            "[llmman bench] {model}: {} warmup + {} timed epoch(s)...",
            args.warmup, args.epochs
        );
        for _ in 0..args.warmup {
            rt.block_on(one_request(&client, &model, &prompt, args))?;
        }

        let mut samples = Vec::with_capacity(args.epochs as usize);
        for epoch in 0..args.epochs {
            let sample = rt.block_on(one_request(&client, &model, &prompt, args))?;
            eprintln!(
                "[llmman bench]   epoch {}/{}: prefill {:.1} tok/s, decode {:.1} tok/s",
                epoch + 1,
                args.epochs,
                sample.prefill_toks_per_sec(),
                sample.decode_toks_per_sec(),
            );
            samples.push(sample);
        }
        results.push((raw_model.clone(), Sample::mean(&samples)));
    }

    match args.format.as_str() {
        "csv" => print_csv(&results),
        _ => print_table(&results),
    }
    Ok(())
}

/// A synthetic filler prompt targeting roughly `target_tokens` tokens —
/// see `BenchArgs::prompt_tokens`'s doc comment for why this is only an
/// approximation (llmman has no local tokenizer of its own to measure
/// against; whatever the backend actually counts is what gets reported).
/// Repeating a single short, common word keeps the approximation
/// reasonably close for most BPE tokenizers, which encode it as one
/// token each.
fn synthetic_prompt(target_tokens: u32) -> String {
    "hello ".repeat(target_tokens.max(1) as usize)
}

// ---------------------------------------------------------------------------
// One timed request
// ---------------------------------------------------------------------------

/// One epoch's measurements. `ttft` (time to first streamed token — see
/// `BenchDelta::has_any_token`) is treated as this request's prefill time
/// — the two are inseparable from a plain HTTP client's point of view,
/// since nothing is received at all until the backend has both finished
/// processing the whole prompt *and* generated the first output token —
/// matching how `ollama-bench` and most other black-box LLM benchmarks
/// approximate the same split.
#[derive(Debug, Clone, Copy)]
struct Sample {
    ttft: Duration,
    total: Duration,
    prompt_tokens: u32,
    completion_tokens: u32,
}

impl Sample {
    fn prefill_toks_per_sec(&self) -> f64 {
        let secs = self.ttft.as_secs_f64();
        if secs <= 0.0 {
            0.0
        } else {
            self.prompt_tokens as f64 / secs
        }
    }

    /// Excludes the one token already counted within `ttft` above, so
    /// this and `prefill_toks_per_sec` don't double-count it.
    fn decode_toks_per_sec(&self) -> f64 {
        let decode_tokens = self.completion_tokens.saturating_sub(1);
        let secs = (self.total - self.ttft.min(self.total)).as_secs_f64();
        if decode_tokens == 0 || secs <= 0.0 {
            0.0
        } else {
            decode_tokens as f64 / secs
        }
    }

    /// Averages every field across `samples` — the per-model result this
    /// command actually reports, smoothing out any one epoch's jitter.
    /// Panics on an empty slice; every call site here always has at least
    /// one epoch (clap's own `default_value_t = 3` combined with
    /// `--epochs 0` would just report an average of zero samples, so this
    /// guards that instead of dividing by zero silently).
    fn mean(samples: &[Sample]) -> Sample {
        let n = samples.len().max(1) as u32;
        let sum = samples.iter().fold(
            Sample {
                ttft: Duration::ZERO,
                total: Duration::ZERO,
                prompt_tokens: 0,
                completion_tokens: 0,
            },
            |acc, s| Sample {
                ttft: acc.ttft + s.ttft,
                total: acc.total + s.total,
                prompt_tokens: acc.prompt_tokens + s.prompt_tokens,
                completion_tokens: acc.completion_tokens + s.completion_tokens,
            },
        );
        Sample {
            ttft: sum.ttft / n,
            total: sum.total / n,
            prompt_tokens: sum.prompt_tokens / n,
            completion_tokens: sum.completion_tokens / n,
        }
    }
}

#[derive(Serialize)]
struct BenchRequest<'a> {
    model: &'a str,
    messages: [BenchMessage; 1],
    stream: bool,
    temperature: f32,
    max_tokens: u32,
    stream_options: StreamOptions,
}

#[derive(Serialize)]
struct BenchMessage {
    role: &'static str,
    content: String,
}

/// `include_usage` asks llama-server's (OpenAI-compatible) streaming
/// endpoint to append one final chunk carrying a `usage` object with real
/// `prompt_tokens`/`completion_tokens` counts — otherwise omitted, same
/// as OpenAI's own API.
#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Deserialize, Default)]
struct BenchChunk {
    #[serde(default)]
    choices: Vec<BenchChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize, Default)]
struct BenchChoice {
    #[serde(default)]
    delta: BenchDelta,
}

#[derive(Deserialize, Default)]
struct BenchDelta {
    #[serde(default)]
    content: Option<String>,
    // A "thinking"-capable model (see cmd::serve's own
    // think_to_chat_template_kwargs/oai_chunk_to_content) streams its
    // reasoning under one of these two field names — not `content` —
    // often for the model's *entire* token budget on a short
    // `--max-tokens` run (observed firsthand with Qwen3.5's default
    // thinking behavior: zero `content` tokens at all within 300).
    // TTFT has to count either: prefill ends at the first token emitted
    // of *any* kind, not specifically the first non-reasoning one, or a
    // thinking-heavy response would misreport its entire generation
    // phase as "prefill".
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
}

impl BenchDelta {
    fn has_any_token(&self) -> bool {
        [&self.content, &self.reasoning_content, &self.thinking]
            .into_iter()
            .any(|f| f.as_deref().is_some_and(|s| !s.is_empty()))
    }
}

#[derive(Deserialize, Default, Clone, Copy)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

/// Sends one streamed chat completion and measures it — see `Sample`'s
/// own doc comment for what each field means. Async (not
/// `reqwest::blocking`) purely so `run` can share one `tokio::Runtime`
/// across every warmup/timed request instead of spinning one up per
/// call.
async fn one_request(
    client: &Client,
    model: &str,
    prompt: &str,
    args: &BenchArgs,
) -> anyhow::Result<Sample> {
    let start = Instant::now();
    let resp = client
        .post(format!("{SERVER}/v1/chat/completions"))
        .json(&BenchRequest {
            model,
            messages: [BenchMessage {
                role: "user",
                content: prompt.to_string(),
            }],
            stream: true,
            temperature: args.temperature,
            max_tokens: args.max_tokens,
            stream_options: StreamOptions {
                include_usage: true,
            },
        })
        .send()
        .await
        .context("connect to llmman serve")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("{model}: server returned {status}: {body}");
    }

    let byte_stream = resp.bytes_stream().map_err(std::io::Error::other);
    let mut lines =
        tokio::io::BufReader::new(tokio_util::io::StreamReader::new(byte_stream)).lines();

    let mut ttft: Option<Duration> = None;
    let mut usage: Option<Usage> = None;
    // Set only by the `[DONE]` sentinel below — not by the read loop
    // simply running out of lines, which also happens on a connection
    // dropped mid-response. Checked (alongside `usage`) once the loop
    // exits, so a truncated stream reports as an error instead of a
    // silently partial/zeroed-out Sample (e.g. `total_tokens` and
    // `ttft`/`total` measuring only however much arrived before the
    // drop, understating both throughput figures without any indication
    // the request never actually finished).
    let mut saw_done = false;
    while let Some(line) = lines.next_line().await.context("read response stream")? {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload == "[DONE]" {
            saw_done = true;
            break;
        }
        let Ok(chunk) = serde_json::from_str::<BenchChunk>(payload) else {
            continue;
        };
        if let Some(u) = chunk.usage {
            usage = Some(u);
        }
        if ttft.is_none()
            && chunk
                .choices
                .first()
                .is_some_and(|c| c.delta.has_any_token())
        {
            ttft = Some(start.elapsed());
        }
    }
    let total = start.elapsed();

    if !saw_done {
        anyhow::bail!(
            "{model}: stream ended without a [DONE] terminator (connection dropped mid-response?)"
        );
    }
    let usage = usage.ok_or_else(|| {
        anyhow::anyhow!(
            "{model}: stream completed without a usage summary — backend may not support \
             stream_options.include_usage"
        )
    })?;

    Ok(Sample {
        ttft: ttft.unwrap_or(total),
        total,
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
    })
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

fn print_table(results: &[(String, Sample)]) {
    let name_w = results
        .iter()
        .map(|(n, _)| n.len())
        .max()
        .unwrap_or(5)
        .max(5);
    println!(
        "{:<name_w$}  {:>14}  {:>14}  {:>10}  {:>10}  {:>12}  {:>14}",
        "MODEL",
        "PREFILL tok/s",
        "DECODE tok/s",
        "TTFT",
        "TOTAL",
        "PROMPT tok",
        "COMPLETION tok",
        name_w = name_w,
    );
    for (name, s) in results {
        println!(
            "{:<name_w$}  {:>14.1}  {:>14.1}  {:>10}  {:>10}  {:>12}  {:>14}",
            name,
            s.prefill_toks_per_sec(),
            s.decode_toks_per_sec(),
            format!("{:.2}s", s.ttft.as_secs_f64()),
            format!("{:.2}s", s.total.as_secs_f64()),
            s.prompt_tokens,
            s.completion_tokens,
            name_w = name_w,
        );
    }
}

fn print_csv(results: &[(String, Sample)]) {
    println!("model,prefill_toks_per_sec,decode_toks_per_sec,ttft_ms,total_ms,prompt_tokens,completion_tokens");
    for (name, s) in results {
        println!(
            "{},{:.2},{:.2},{},{},{},{}",
            name,
            s.prefill_toks_per_sec(),
            s.decode_toks_per_sec(),
            s.ttft.as_millis(),
            s.total.as_millis(),
            s.prompt_tokens,
            s.completion_tokens,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(ttft_ms: u64, total_ms: u64, prompt_tokens: u32, completion_tokens: u32) -> Sample {
        Sample {
            ttft: Duration::from_millis(ttft_ms),
            total: Duration::from_millis(total_ms),
            prompt_tokens,
            completion_tokens,
        }
    }

    fn args(model: &str, format: &str, epochs: u32) -> BenchArgs {
        BenchArgs {
            model: vec![model.to_string()],
            prompt: DEFAULT_PROMPT.to_string(),
            prompt_tokens: 0,
            max_tokens: 200,
            epochs,
            warmup: 0,
            temperature: 0.0,
            timeout: 300,
            format: format.to_string(),
        }
    }

    #[test]
    fn run_rejects_zero_epochs_before_touching_the_network() {
        // Both checks in `run` happen before `daemon::ensure_server`, so
        // this must fail fast without needing (or starting) a real
        // daemon — otherwise `Sample::mean(&[])`'s `len().max(1)` divisor
        // would silently report an all-zero "successful" result for 0
        // actually-timed epochs instead of refusing to run at all.
        let err = run(&args("unused-model", "text", 0)).unwrap_err();
        assert!(err.to_string().contains("--epochs must be at least 1"));
    }

    #[test]
    fn run_rejects_an_unknown_format_before_touching_the_network() {
        let err = run(&args("unused-model", "yaml", 1)).unwrap_err();
        assert!(err.to_string().contains("--format must be"));
    }

    #[test]
    fn prefill_toks_per_sec_divides_prompt_tokens_by_ttft() {
        // 512 prompt tokens processed in exactly 1s => 512 tok/s.
        let s = sample(1000, 5000, 512, 200);
        assert!((s.prefill_toks_per_sec() - 512.0).abs() < 0.01);
    }

    #[test]
    fn decode_toks_per_sec_excludes_the_first_token_already_counted_in_ttft() {
        // 200 completion tokens total; ttft covers the first one, leaving
        // 199 decoded over (total - ttft) = 4s => 49.75 tok/s.
        let s = sample(1000, 5000, 512, 200);
        assert!((s.decode_toks_per_sec() - 49.75).abs() < 0.01);
    }

    #[test]
    fn decode_toks_per_sec_is_zero_for_a_single_completion_token() {
        // Nothing left to decode after the one token counted in ttft.
        let s = sample(1000, 1000, 512, 1);
        assert_eq!(s.decode_toks_per_sec(), 0.0);
    }

    #[test]
    fn zero_duration_denominators_report_zero_instead_of_dividing_by_zero() {
        let s = sample(0, 0, 512, 200);
        assert_eq!(s.prefill_toks_per_sec(), 0.0);
        assert_eq!(s.decode_toks_per_sec(), 0.0);
    }

    #[test]
    fn mean_averages_every_field_across_samples() {
        let a = sample(1000, 3000, 100, 50);
        let b = sample(2000, 5000, 200, 150);
        let m = Sample::mean(&[a, b]);
        assert_eq!(m.ttft, Duration::from_millis(1500));
        assert_eq!(m.total, Duration::from_millis(4000));
        assert_eq!(m.prompt_tokens, 150);
        assert_eq!(m.completion_tokens, 100);
    }

    #[test]
    fn mean_of_a_single_sample_is_itself() {
        let a = sample(1234, 5678, 111, 222);
        let m = Sample::mean(&[a]);
        assert_eq!(m.ttft, a.ttft);
        assert_eq!(m.total, a.total);
        assert_eq!(m.prompt_tokens, a.prompt_tokens);
        assert_eq!(m.completion_tokens, a.completion_tokens);
    }

    #[test]
    fn synthetic_prompt_repeats_a_filler_word_target_tokens_times() {
        assert_eq!(synthetic_prompt(3), "hello hello hello ");
        // Guards against a 0-length (or negative, pre-u32) prompt.
        assert_eq!(synthetic_prompt(0), "hello ");
    }
}

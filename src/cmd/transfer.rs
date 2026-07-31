use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::Args;

use crate::ffi;

#[derive(Args, Debug)]
pub struct TransferArgs {
    /// Source reference to copy from, e.g. `hf.co/owner/repo`,
    /// `registry.example.com/repo:tag`, or any other reference `llmman
    /// pull` understands (hf://, ms://, ngc://, s3://, gs://, ...)
    #[arg(value_name = "SOURCE")]
    pub source: String,

    /// Destination OCI registry reference to copy to, e.g.
    /// `registry.example.com/repo:tag`
    #[arg(value_name = "DESTINATION")]
    pub destination: String,
}

/// `llmman transfer` is llmman's equivalent of `skopeo copy`: it copies an
/// image directly from one location to another without leaving it behind
/// in the persistent local store (see `cmd::pull`/`cmd::push` for that).
///
/// The motivating case is HuggingFace → OCI registry —
/// `llmman transfer hf.co/owner/model registry.example.com/owner/model` —
/// but any source `llmman_pull` understands (an OCI registry, `hf://`,
/// `ms://`, ...) can be paired with any OCI registry destination, because
/// both halves reuse the exact same containerd/podman-backed FFI calls
/// `llmman pull`/`llmman push` use (`ffi::pull`/`ffi::push`) — no separate
/// "transfer" logic exists in the Go shim at all.
///
/// A private, throwaway OCI layout directory is used for the intermediate
/// hop (removed once the transfer finishes, whether it succeeds or fails)
/// rather than the daemon's persistent, possibly multi-image store: it
/// holds exactly one manifest at a time, so neither backend's "find the
/// manifest to push" lookup — which only works by exact tag-annotation
/// match, or else falls back to "the only manifest present" (see
/// `backend_docker.go`'s `findManifestDesc` / `backend_podman.go`'s `oci:`
/// source reference) — can ever pick up the wrong image, regardless of
/// what the source and destination references happen to be named.
///
/// This intentionally talks to the Go shim directly (like `login`/`logout`
/// and `inspect --remote`) rather than through a running `llmman serve`
/// daemon (like `pull`/`push`): the staging directory below is private to
/// this one invocation, so there's no shared daemon state to coordinate.
pub fn run(args: &TransferArgs) -> anyhow::Result<()> {
    let source = crate::shortnames::resolve(&args.source);
    let destination = crate::shortnames::resolve(&args.destination);

    let staging = staging_dir()?;
    std::fs::create_dir_all(&staging)
        .with_context(|| format!("create staging directory {}", staging.display()))?;
    let result = do_transfer(&source, &destination, &staging);
    let _ = std::fs::remove_dir_all(&staging);
    result?;

    println!("Transferred {source} to {destination}");
    Ok(())
}

fn do_transfer(source: &str, destination: &str, staging: &Path) -> anyhow::Result<()> {
    let layout_dir = staging
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("staging path is not valid UTF-8"))?;

    println!("Pulling {source}");
    ffi::pull(source, layout_dir)?;

    println!("Pushing {destination}");
    ffi::push(layout_dir, destination)?;
    Ok(())
}

/// A unique per-invocation staging directory under the store's parent
/// directory (same filesystem as the persistent store, so blobs could be
/// hard-linked instead of copied there in the future without crossing a
/// mount point).
fn staging_dir() -> anyhow::Result<PathBuf> {
    let store = crate::default_store(None)?;
    let base = store.parent().map(Path::to_path_buf).unwrap_or(store);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    Ok(base.join("transfer").join(format!("{}-{nanos}", std::process::id())))
}

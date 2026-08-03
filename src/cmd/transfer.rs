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
/// but any source the Go shim's pull path understands (an OCI registry,
/// `hf://`, `ms://`, ...) can be paired with any OCI registry destination.
///
/// Unlike an earlier version of this command, the actual copy — and the
/// decision between streaming a blob straight through versus falling back
/// to a throwaway local staging directory — lives entirely in the Go shim
/// (`go-shim/transfer_docker.go` / `transfer_podman.go`), exactly the way
/// `login`/`push`/`pull` already keep their real logic there. See those
/// files for why: skopeo's own `copy.Image` can stream a blob straight
/// from source to destination only because it already knows that blob's
/// digest from the source's OCI manifest before it starts; matching that
/// for a HuggingFace source needs a HEAD request against the file first
/// (to learn its real content digest the same way, before streaming the
/// GET response straight into the registry push), which only makes sense
/// to implement where the registry push itself happens.
///
/// This intentionally talks to the Go shim directly (like `login`/`logout`
/// and `inspect --remote`) rather than through a running `llmman serve`
/// daemon (like `pull`/`push`): a transfer never touches the daemon's
/// persistent store, so there's no shared state to coordinate.
pub fn run(args: &TransferArgs) -> anyhow::Result<()> {
    let source = crate::shortnames::resolve(&args.source);
    let destination = crate::shortnames::resolve(&args.destination);

    let changed = ffi::transfer(&source, &destination)?;

    if changed {
        println!("Transferred {source} to {destination}");
    } else {
        println!("{destination} already up to date with {source}");
    }
    Ok(())
}

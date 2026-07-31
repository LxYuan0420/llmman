use clap::Args;

use crate::{ffi, hf};

#[derive(Args, Debug)]
pub struct LogoutArgs {
    /// Server to log out of: an OCI registry host, or a HuggingFace host
    /// (hf.co, huggingface.co). Defaults to docker.io when omitted.
    #[arg(value_name = "SERVER")]
    pub server: Option<String>,
}

pub fn run(args: &LogoutArgs) -> anyhow::Result<()> {
    let server = args
        .server
        .clone()
        .unwrap_or_else(|| "docker.io".to_string());

    if hf::is_hf_host(&server) {
        hf::logout()?;
        println!("Logged out of {server}");
        return Ok(());
    }

    ffi::logout(&server)?;
    println!("Logged out of {server}");
    Ok(())
}

use clap::Args;

use crate::{ffi, hf};

#[derive(Args, Debug)]
pub struct LoginArgs {
    /// Server to log in to: an OCI registry host (e.g.
    /// registry.example.com) for username/password credentials, or a
    /// HuggingFace host (hf.co, huggingface.co) for a HuggingFace access
    /// token. Defaults to docker.io when omitted.
    #[arg(value_name = "SERVER")]
    pub server: Option<String>,

    /// Registry username (ignored for HuggingFace logins, which are token-only)
    #[arg(short, long)]
    pub username: Option<String>,

    /// Registry password, read from stdin if omitted (ignored for HuggingFace logins)
    #[arg(short, long)]
    pub password: Option<String>,

    /// HuggingFace access token, prompted for if omitted (ignored for registry logins)
    #[arg(short, long)]
    pub token: Option<String>,
}

pub fn run(args: &LoginArgs) -> anyhow::Result<()> {
    let server = args
        .server
        .clone()
        .unwrap_or_else(|| "docker.io".to_string());

    if hf::is_hf_host(&server) {
        let token = match &args.token {
            Some(t) => t.clone(),
            None => {
                eprint!("Token (from https://huggingface.co/settings/tokens): ");
                read_line_stdin()?
            }
        };
        let username = hf::login(token.trim())?;
        println!("Login succeeded for {server} (user: {username})");
        return Ok(());
    }

    let username = args
        .username
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--username is required to log in to a registry"))?;
    let password = match &args.password {
        Some(p) => p.clone(),
        None => {
            eprint!("Password: ");
            read_line_stdin()?
        }
    };
    ffi::login(&server, &username, &password)?;
    println!("Login succeeded for {server}");
    Ok(())
}

/// Read a line from stdin, trimming the trailing newline.
/// Does not suppress echo — callers wanting a TTY UX should pipe through a helper.
fn read_line_stdin() -> anyhow::Result<String> {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    Ok(line.trim_end_matches(['\n', '\r']).to_string())
}

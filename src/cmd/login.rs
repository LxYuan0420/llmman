use std::io::IsTerminal;

use clap::Args;

use crate::{ffi, hf, oauth};

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

/// Mirrors `docker login`'s own decision rule (`cli/command/registry/
/// login.go`'s `loginUser`): the web-based device-authorization flow is
/// only ever attempted for the default registry (docker.io, i.e. no
/// `SERVER` given or `docker.io` given explicitly), and only when neither
/// `--username` nor `--password` was supplied — those two are always an
/// explicit request for the plain credential flow instead. Any other
/// registry always goes straight to username/password, same as before.
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

    let is_default_registry = server == "docker.io";
    if is_default_registry && args.username.is_none() && args.password.is_none() {
        match oauth::login_device() {
            Ok(Some(result)) => {
                ffi::login(&server, &result.username, &result.password)?;
                println!();
                println!("Login Succeeded");
                return Ok(());
            }
            Ok(None) => {
                // Mirrors docker/cli's exact fallback message (the device
                // flow couldn't even be started — e.g. no route to
                // login.docker.com — so drop to the plain credential flow
                // below instead of failing outright).
                println!("Failed to start web-based login - falling back to command line login...");
                println!();
            }
            // The flow *started* but failed afterwards (timeout, denial,
            // network error while polling) — docker/cli does not fall back
            // in that case either, it just reports the error.
            Err(e) => return Err(e),
        }
    }

    let username = match &args.username {
        Some(u) => u.clone(),
        None => {
            if !std::io::stdin().is_terminal() {
                anyhow::bail!("cannot perform an interactive login from a non-TTY device");
            }
            eprint!("Username: ");
            read_line_stdin()?
        }
    };
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

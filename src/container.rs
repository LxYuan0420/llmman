//! Runs `llama-server` inside a container (Linux only, via `--conman
//! docker|podman`) instead of as a local process, auto-selecting the
//! matching `ghcr.io/ggml-org/llama.cpp:server-<backend>` image for
//! whatever GPU acceleration the host actually has — see
//! <https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md> for
//! the full image list and their `docker run` flags, which
//! [`GpuBackend::engine_args`] mirrors for the subset detected here.
//!
//! This is the same problem ggml's own dynamic backend loading
//! (`GGML_BACKEND_DL=ON`, `ggml_backend_load_all` in
//! `ggml/src/ggml-backend-reg.cpp`) solves for shared libraries — given
//! several installed backend libraries, pick the best one for this
//! machine at runtime — except there's no shared library to load and
//! score here, just one container image to run, so detection below is a
//! fixed priority order (CUDA > ROCm > Vulkan > CPU) rather than a
//! numeric score.

use std::path::Path;

use anyhow::{Context, Result};
use clap::ValueEnum;

/// Container engine to run the picked image with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "lower")]
pub enum ContainerManager {
    Docker,
    Podman,
}

impl ContainerManager {
    pub fn binary(self) -> &'static str {
        match self {
            ContainerManager::Docker => "docker",
            ContainerManager::Podman => "podman",
        }
    }
}

/// GPU backends this module can detect and run a matching
/// `ghcr.io/ggml-org/llama.cpp:server-*` image for. Deliberately a subset
/// of every tag llama.cpp publishes (musa/intel/openvino are skipped): as
/// of writing, rocm/vulkan images are amd64-only upstream and cuda/vulkan
/// support arm64 too — see docs/docker.md for the authoritative list if
/// more get added here later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuBackend {
    Cpu,
    Cuda12,
    Cuda13,
    Rocm,
    Vulkan,
}

impl GpuBackend {
    /// The `server-<suffix>` part of the image tag; Cpu has no suffix.
    fn image_tag(self) -> &'static str {
        match self {
            GpuBackend::Cpu => "server",
            GpuBackend::Cuda12 => "server-cuda",
            GpuBackend::Cuda13 => "server-cuda13",
            GpuBackend::Rocm => "server-rocm",
            GpuBackend::Vulkan => "server-vulkan",
        }
    }

    /// The full `ghcr.io/ggml-org/llama.cpp:<tag>` reference. `version`,
    /// when given, pins to that release (e.g. `server-b9994` instead of
    /// the floating `server`) — ghcr.io/ggml-org/llama.cpp publishes a
    /// versioned tag alongside every floating one, built from the same
    /// release. llmman itself has no opinion on which (or whether) to
    /// pin: reproducibility across runs is the caller's concern (see
    /// `ServeArgs::llama_cpp_version` in cmd::serve), not something to
    /// default or hardcode here.
    fn image_ref(self, version: Option<&str>) -> String {
        match version {
            Some(v) => format!("ghcr.io/ggml-org/llama.cpp:{}-{v}", self.image_tag()),
            None => format!("ghcr.io/ggml-org/llama.cpp:{}", self.image_tag()),
        }
    }

    /// Extra `docker run`/`podman run` arguments needed to see the host's
    /// GPU from inside the container, matching docs/docker.md's own
    /// examples for each backend (CUDA: "Docker With CUDA"; ROCm/Vulkan:
    /// the SYCL section's `--device /dev/dri` pattern, extended for ROCm
    /// with the `/dev/kfd` compute device and `video` group every ROCm
    /// container image's own documentation asks for).
    fn engine_args(self) -> Vec<String> {
        match self {
            GpuBackend::Cpu => vec![],
            GpuBackend::Cuda12 | GpuBackend::Cuda13 => vec!["--gpus".into(), "all".into()],
            GpuBackend::Rocm => vec![
                "--device".into(),
                "/dev/kfd".into(),
                "--device".into(),
                "/dev/dri".into(),
                "--group-add".into(),
                "video".into(),
            ],
            GpuBackend::Vulkan => vec!["--device".into(), "/dev/dri".into()],
        }
    }
}

/// Detects the best available GPU backend, in priority order
/// CUDA > ROCm > Vulkan > CPU. Each check only inspects the *host* (via
/// `nvidia-smi`, `/dev/kfd`, `/dev/dri`) — it can't verify the container
/// engine itself is configured to pass a GPU through (e.g. whether
/// nvidia-container-toolkit is actually registered with Docker/Podman).
/// `docker run --gpus all` surfaces that misconfiguration directly and
/// clearly enough on its own if it's missing, so detection here stays a
/// simple, fast host probe rather than trying to fully replicate GPU
/// passthrough validation too.
fn detect_backend() -> GpuBackend {
    if let Some(cuda) = detect_cuda() {
        return cuda;
    }
    if detect_rocm() {
        return GpuBackend::Rocm;
    }
    if detect_vulkan() {
        return GpuBackend::Vulkan;
    }
    GpuBackend::Cpu
}

/// `nvidia-smi` ships with the NVIDIA driver itself (not the container
/// toolkit), so its presence and a successful exit is a reliable "there's
/// an NVIDIA GPU with a working driver here" signal. Its human-readable
/// output is parsed by [`cuda_backend_from_nvidia_smi_output`] to pick
/// cuda12 vs. cuda13.
fn detect_cuda() -> Option<GpuBackend> {
    let output = std::process::Command::new("nvidia-smi").output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(cuda_backend_from_nvidia_smi_output(
        &String::from_utf8_lossy(&output.stdout),
    ))
}

/// Picks cuda12 vs. cuda13 from `nvidia-smi`'s human-readable header, which
/// reports the driver's supported CUDA version as e.g. "CUDA Version: 13.0"
/// — major version 13 and up picks the cuda13 image, matching llama.cpp's
/// own CUDA Dockerfile split between the `cuda`/`cuda12` tag (built against
/// CUDA_VERSION 12.8.1) and `cuda13` (13.3.0); see docs/docker.md. Defaults
/// to cuda12 if the version can't be parsed out of the output at all,
/// since that tag covers a much wider range of driver versions.
fn cuda_backend_from_nvidia_smi_output(text: &str) -> GpuBackend {
    let major: Option<u32> = text
        .split("CUDA Version:")
        .nth(1)
        .and_then(|rest| rest.trim().split(['.', ' ']).next())
        .and_then(|v| v.parse().ok());
    match major {
        Some(m) if m >= 13 => GpuBackend::Cuda13,
        _ => GpuBackend::Cuda12,
    }
}

/// `/dev/kfd` is the ROCm kernel-fusion-driver device node; its presence
/// means the amdgpu kernel driver is loaded *with* ROCm compute support (a
/// plain display-only amdgpu driver doesn't create it) — the same device
/// [`GpuBackend::engine_args`] then mounts into the container.
fn detect_rocm() -> bool {
    Path::new("/dev/kfd").exists()
}

/// Any DRI render node is a reasonable enough signal for "there's a GPU
/// with a kernel driver loaded" to try the generic Vulkan image as a
/// catch-all (Intel iGPUs, and NVIDIA/AMD systems that have a GPU but no
/// CUDA/ROCm toolkit installed) — the same device docs/docker.md's own
/// Vulkan and SYCL examples mount.
fn detect_vulkan() -> bool {
    std::fs::read_dir("/dev/dri")
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .any(|e| e.file_name().to_string_lossy().starts_with("renderD"))
        })
        .unwrap_or(false)
}

/// Runs `llama-server` inside a container: `docker run --rm --init -t`
/// (or `podman run` with the same flags), auto-selecting the image for
/// whatever [`detect_backend`] found. Returns the running child process
/// (the attached `docker`/`podman` CLI itself, not the container) — the
/// container's own stdio is inherited through it, same as a local
/// `llama-server` child's would be.
///
/// This runs *attached* (no `-d`) specifically so it can be managed like
/// a normal child process: `--init` runs a real init (tini) as the
/// container's PID 1, so SIGTERM forwarded to it (e.g. via `docker stop`,
/// or the CLI's own signal forwarding while attached) is actually
/// delivered with default disposition and terminates the container
/// promptly — a bare `sleep`/`llama-server` running *as* PID 1 (no
/// `--init`) does not get default signal handling at all, a well-known
/// Linux PID-1 gotcha, and was verified live to leave the container
/// running indefinitely after being sent SIGTERM. `--rm` then cleans up
/// the stopped container automatically. `-t` allocates a pseudo-tty so
/// the containerized process's own output behaves like a normal
/// interactive process (typically line-buffered) instead of block image
/// buffered as it would through a plain pipe — deliberately *not* paired
/// with `-i`: `-i` needs an actual open, readable stdin to attach, which
/// fails ("cannot attach stdin to a TTY-enabled container because stdin
/// is not a terminal") when combined with `-t` and this process's own
/// stdin isn't a real terminal — the common case, since `llmman serve`
/// itself is normally daemonized with stdin closed (see daemon.rs).
///
/// Pulls the image [`spawn`] would run for the current host's detected GPU
/// backend, with the pull's own progress output (a real `docker pull`/
/// `podman pull` progress bar — not something llmman re-implements)
/// inherited directly to this process's stdout/stderr.
///
/// `spawn`'s underlying `docker run`/`podman run` would pull an image that
/// isn't already cached locally on its own, but silently and without any
/// visible progress from the caller's perspective (its own stdio is
/// redirected to a log file when started detached — see daemon.rs and
/// cmd::serve). A caller that wants to warm this up as its own distinct,
/// visible step first (typically right before starting `llmman serve
/// --conman ...` detached, so a slow first pull doesn't look like a stuck
/// first prompt to whoever's waiting on it) should call this — in the
/// foreground, before `serve` is even started — rather than relying on
/// `spawn`'s own implicit pull.
pub fn pull_image(conman: ContainerManager, llama_cpp_version: Option<&str>) -> Result<()> {
    let backend = detect_backend();
    let image = backend.image_ref(llama_cpp_version);
    eprintln!("[llmman] {}: pulling {image}...", conman.binary());
    let status = std::process::Command::new(conman.binary())
        .args(["pull", &image])
        .status()
        .with_context(|| format!("run {} pull {image}", conman.binary()))?;
    if !status.success() {
        anyhow::bail!("{} pull {image} failed", conman.binary());
    }
    Ok(())
}

/// Callers must stop this gracefully (SIGTERM, not the default
/// `Child::kill()`/`kill_on_drop`, which sends SIGKILL) — see
/// `cmd::serve::ModelProcess`'s Drop impl. SIGKILL cannot be caught or
/// forwarded by the CLI process at all (that's what SIGKILL means), so
/// it was also verified live to leave the container running.
///
/// `llama_cpp_version`, when given, pins the image to that release tag
/// (see [`GpuBackend::image_ref`]) instead of the floating one.
pub fn spawn(
    conman: ContainerManager,
    model_path: &Path,
    port: u16,
    llama_cpp_version: Option<&str>,
) -> Result<tokio::process::Child> {
    let backend = detect_backend();
    let image = backend.image_ref(llama_cpp_version);
    eprintln!(
        "[llmman] {}: detected {:?}, using image {:?}",
        conman.binary(),
        backend,
        image
    );

    let model_dir = model_path
        .parent()
        .context("model path has no parent directory")?;
    let model_file = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .context("model path has no valid UTF-8 filename")?;
    let model_dir_str = model_dir
        .to_str()
        .context("model directory is not valid UTF-8")?;

    let mut args: Vec<String> = vec![
        "run".into(),
        "--rm".into(),
        "--init".into(),
        "-t".into(),
        "-p".into(),
        format!("127.0.0.1:{port}:{port}"),
        "-v".into(),
        format!("{model_dir_str}:/models:ro"),
    ];
    args.extend(backend.engine_args());
    args.push(image);
    args.extend([
        "-m".into(),
        format!("/models/{model_file}"),
        "--port".into(),
        port.to_string(),
        "--host".into(),
        "0.0.0.0".into(),
    ]);

    tokio::process::Command::new(conman.binary())
        .args(&args)
        .stdin(std::process::Stdio::null())
        .spawn()
        .with_context(|| format!("spawn {} {}", conman.binary(), args.join(" ")))
}

/// Gracefully stops a container started by [`spawn`] by sending SIGTERM to
/// the attached `docker`/`podman` CLI process — see `spawn`'s doc comment
/// for why this must be SIGTERM (forwarded to the container's `--init`
/// PID 1) and not the default forceful kill. Best-effort: called from a
/// synchronous `Drop` impl (see `ModelProcess` in cmd::serve), so errors
/// are only logged, never propagated. Unix only (matching `--conman`
/// itself, which cmd::serve::serve_async already rejects on other
/// platforms) — `libc::kill` is not meaningful on Windows.
#[cfg(unix)]
pub fn stop(pid: u32) {
    // SAFETY: kill(2) with an existing pid and a valid signal number is
    // always safe to call; a stale/already-reaped pid just returns ESRCH,
    // which is not a memory-safety concern.
    let result = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
    if result != 0 {
        let err = std::io::Error::last_os_error();
        eprintln!("[llmman] warning: SIGTERM to container process {pid} failed: {err}");
    }
}

/// Unreachable in practice (`--conman` is rejected on non-Linux before
/// `spawn` is ever called — see cmd::serve::serve_async), but this needs
/// to compile on every platform llmman ships for, and a plain forceful
/// kill here is at least no worse than the SIGKILL callers were already
/// relying on before this module existed.
#[cfg(not(unix))]
pub fn stop(_pid: u32) {
    eprintln!("[llmman] warning: container::stop is a no-op on non-Unix platforms");
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real `nvidia-smi` header excerpt shape (whitespace-padded table),
    // trimmed to just the line this module actually parses.
    const SMI_HEADER_124: &str = "\
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15    Driver Version: 550.54.15    CUDA Version: 12.4     |
+-------------------------------+----------------------+----------------------+
";
    const SMI_HEADER_130: &str = "\
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 580.65.06    Driver Version: 580.65.06    CUDA Version: 13.0     |
+-------------------------------+----------------------+----------------------+
";

    #[test]
    fn cuda12_driver_picks_cuda12_image() {
        assert_eq!(
            cuda_backend_from_nvidia_smi_output(SMI_HEADER_124),
            GpuBackend::Cuda12
        );
        assert_eq!(GpuBackend::Cuda12.image_tag(), "server-cuda");
    }

    #[test]
    fn cuda13_driver_picks_cuda13_image() {
        assert_eq!(
            cuda_backend_from_nvidia_smi_output(SMI_HEADER_130),
            GpuBackend::Cuda13
        );
        assert_eq!(GpuBackend::Cuda13.image_tag(), "server-cuda13");
    }

    #[test]
    fn unparseable_output_defaults_to_cuda12() {
        assert_eq!(
            cuda_backend_from_nvidia_smi_output("garbage, no version here"),
            GpuBackend::Cuda12
        );
    }

    #[test]
    fn image_tags_match_docs_docker_md() {
        assert_eq!(GpuBackend::Cpu.image_tag(), "server");
        assert_eq!(GpuBackend::Cuda12.image_tag(), "server-cuda");
        assert_eq!(GpuBackend::Cuda13.image_tag(), "server-cuda13");
        assert_eq!(GpuBackend::Rocm.image_tag(), "server-rocm");
        assert_eq!(GpuBackend::Vulkan.image_tag(), "server-vulkan");
    }

    #[test]
    fn image_ref_uses_floating_tag_when_no_version_given() {
        assert_eq!(GpuBackend::Cpu.image_ref(None), "ghcr.io/ggml-org/llama.cpp:server");
        assert_eq!(
            GpuBackend::Cuda13.image_ref(None),
            "ghcr.io/ggml-org/llama.cpp:server-cuda13"
        );
    }

    #[test]
    fn image_ref_pins_to_the_given_version() {
        assert_eq!(
            GpuBackend::Cpu.image_ref(Some("b9994")),
            "ghcr.io/ggml-org/llama.cpp:server-b9994"
        );
        assert_eq!(
            GpuBackend::Cuda13.image_ref(Some("b9994")),
            "ghcr.io/ggml-org/llama.cpp:server-cuda13-b9994"
        );
    }

    #[test]
    fn cpu_backend_has_no_extra_engine_args() {
        assert!(GpuBackend::Cpu.engine_args().is_empty());
    }

    #[test]
    fn cuda_backend_requests_all_gpus() {
        assert_eq!(GpuBackend::Cuda12.engine_args(), vec!["--gpus", "all"]);
        assert_eq!(GpuBackend::Cuda13.engine_args(), vec!["--gpus", "all"]);
    }

    #[test]
    fn rocm_backend_mounts_kfd_and_dri() {
        let args = GpuBackend::Rocm.engine_args();
        assert_eq!(
            args,
            vec!["--device", "/dev/kfd", "--device", "/dev/dri", "--group-add", "video"]
        );
    }

    #[test]
    fn vulkan_backend_mounts_dri_only() {
        assert_eq!(GpuBackend::Vulkan.engine_args(), vec!["--device", "/dev/dri"]);
    }

    #[test]
    fn container_manager_binary_names() {
        assert_eq!(ContainerManager::Docker.binary(), "docker");
        assert_eq!(ContainerManager::Podman.binary(), "podman");
    }
}

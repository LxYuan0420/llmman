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

/// Runs `llama-server` inside a container: `docker run -d` (or
/// `podman run -d`), auto-selecting the image for whatever
/// [`detect_backend`] found. Returns the running container's id.
///
/// Unlike a local `llama-server` child process (killed via
/// `Child::kill_on_drop`), the caller must explicitly
/// `docker/podman rm -f <id>` this container when the model is unloaded
/// (see `stop` below) — killing the `docker run` CLI process here would
/// not stop the container it started, since the container lives in the
/// engine daemon's own process tree, detached from the CLI that launched
/// it, not as a child of this process.
pub async fn spawn(conman: ContainerManager, model_path: &Path, port: u16) -> Result<String> {
    let backend = detect_backend();
    eprintln!(
        "[llmman] {}: detected {:?}, using image tag {:?}",
        conman.binary(),
        backend,
        backend.image_tag()
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

    let image = format!("ghcr.io/ggml-org/llama.cpp:{}", backend.image_tag());

    let mut args: Vec<String> = vec![
        "run".into(),
        "-d".into(),
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

    let output = tokio::process::Command::new(conman.binary())
        .args(&args)
        .output()
        .await
        .with_context(|| format!("run {} {}", conman.binary(), args.join(" ")))?;
    if !output.status.success() {
        anyhow::bail!(
            "{} {} failed: {}",
            conman.binary(),
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let container_id = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if container_id.is_empty() {
        anyhow::bail!("{} run produced no container id", conman.binary());
    }
    Ok(container_id)
}

/// Force-stops and removes a container started by [`spawn`]. Best-effort:
/// called from a synchronous `Drop` impl (see `ModelProcess` in
/// cmd::serve), so errors are only logged, never propagated.
pub fn stop(conman: ContainerManager, container_id: &str) {
    let result = std::process::Command::new(conman.binary())
        .args(["rm", "-f", container_id])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output();
    match result {
        Ok(output) if !output.status.success() => {
            eprintln!(
                "[llmman] warning: {} rm -f {container_id} failed: {}",
                conman.binary(),
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        Err(e) => {
            eprintln!(
                "[llmman] warning: {} rm -f {container_id} failed: {e}",
                conman.binary()
            );
        }
        Ok(_) => {}
    }
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

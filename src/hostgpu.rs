//! Cross-platform detection of the GPU/accelerator, if any, available on
//! the local host — used by [`crate::llama_release`] to pick which
//! prebuilt `llama-server` release asset to download for this machine.
//!
//! This deliberately mirrors (but doesn't share code with — see
//! `container.rs`'s own `detect_backend`) the same coarse, one-shot style
//! of host probing `container.rs` already uses to pick a
//! `ghcr.io/ggml-org/llama.cpp` Docker image: a fixed priority order
//! (CUDA > ROCm > Vulkan > CPU, plus Metal on macOS) based on driver/device
//! presence, not a live enumeration of every device the way Ollama's own
//! `discover/` package does (see that project's `discover/runner.go`,
//! which actually launches `llama-server` itself to ask it what it sees).
//! A fixed priority probe is enough here for the same reason it's enough
//! in `container.rs`: the result only ever selects *which single prebuilt
//! package to run*, never a live multi-GPU scheduling decision.

use std::path::Path;

/// What kind of GPU acceleration, if any, was detected on the local host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostGpu {
    None,
    /// `major` is the CUDA runtime major version reported by the
    /// installed driver (parsed from `nvidia-smi`'s own header) — used to
    /// decide between llama.cpp's separately published CUDA 12 vs. CUDA 13
    /// Windows builds (see `llama_release::asset_query`).
    Cuda { major: u32 },
    Rocm,
    Vulkan,
    /// macOS (Apple Silicon) only.
    Metal,
}

/// Detects the best available accelerator on this host, in priority order
/// CUDA > ROCm > Vulkan > CPU on Linux/Windows, or Metal > CPU on macOS.
pub fn detect() -> HostGpu {
    #[cfg(target_os = "macos")]
    {
        detect_macos()
    }
    #[cfg(target_os = "linux")]
    {
        detect_linux()
    }
    #[cfg(target_os = "windows")]
    {
        detect_windows()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        HostGpu::None
    }
}

/// Every Mac llama.cpp still publishes a build for is Apple Silicon
/// (arm64), whose official release binary embeds Metal support
/// unconditionally (`GGML_METAL_EMBED_LIBRARY=ON` in llama.cpp's own
/// macOS release job) — Metal is inherent to that build, not something to
/// probe for at runtime the way CUDA/ROCm/Vulkan need to be on
/// Linux/Windows. The x64 (Intel Mac) build is CPU-only upstream
/// (`GGML_METAL=OFF` in that same job's Intel leg), so it gets no Metal
/// detection here either.
#[cfg(target_os = "macos")]
fn detect_macos() -> HostGpu {
    if std::env::consts::ARCH == "aarch64" {
        HostGpu::Metal
    } else {
        HostGpu::None
    }
}

#[cfg(target_os = "linux")]
fn detect_linux() -> HostGpu {
    if let Some(g) = detect_cuda_via_nvidia_smi("nvidia-smi") {
        return g;
    }
    // /dev/kfd is the ROCm kernel-fusion-driver device node; its presence
    // means the amdgpu kernel driver is loaded *with* ROCm compute support
    // (a plain display-only amdgpu driver doesn't create it).
    if Path::new("/dev/kfd").exists() {
        return HostGpu::Rocm;
    }
    // Any DRI render node is a reasonable signal for "there's a GPU with a
    // kernel driver loaded" to try the generic Vulkan build (Intel iGPUs,
    // and NVIDIA/AMD systems with a GPU but no CUDA/ROCm toolkit).
    if has_dri_render_node() {
        return HostGpu::Vulkan;
    }
    HostGpu::None
}

#[cfg(target_os = "linux")]
fn has_dri_render_node() -> bool {
    std::fs::read_dir("/dev/dri")
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .any(|e| e.file_name().to_string_lossy().starts_with("renderD"))
        })
        .unwrap_or(false)
}

#[cfg(target_os = "windows")]
fn detect_windows() -> HostGpu {
    if let Some(g) = detect_cuda_via_nvidia_smi("nvidia-smi.exe") {
        return g;
    }
    // ROCm/HIP on Windows: the AMD driver package installs its HIP runtime
    // DLL system-wide (see AMD's ROCm-on-Windows docs) — the same family
    // of DLLs Ollama's own Windows AMD detection keys off of
    // (`discover/amd.go`'s `detectOldAMDDriverWindows`). Checking for it
    // directly on PATH avoids having to parse the registry.
    if find_on_path("amdhip64_6.dll").is_some() || find_on_path("amdhip64.dll").is_some() {
        return HostGpu::Rocm;
    }
    // vulkan-1.dll ships with essentially every GPU vendor's Windows
    // driver package (NVIDIA, AMD, Intel) once any Vulkan-capable driver
    // is installed — matching Ollama's own docs ("On Windows most GPU
    // vendor drivers come bundled with Vulkan support").
    if find_on_path("vulkan-1.dll").is_some() {
        return HostGpu::Vulkan;
    }
    HostGpu::None
}

#[cfg(target_os = "windows")]
fn find_on_path(name: &str) -> Option<std::path::PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    std::env::split_paths(&path_var)
        .map(|dir| dir.join(name))
        .find(|p| p.is_file())
}

/// Shared by Linux and Windows: `nvidia-smi` ships with the NVIDIA driver
/// itself (not a separate CUDA toolkit install), so its presence and a
/// successful exit reliably means "there's an NVIDIA GPU with a working
/// driver here". Its human-readable header reports the driver's supported
/// CUDA runtime version as e.g. "CUDA Version: 13.0", which becomes
/// [`HostGpu::Cuda`]'s `major` field.
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn detect_cuda_via_nvidia_smi(bin: &str) -> Option<HostGpu> {
    let output = std::process::Command::new(bin).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(HostGpu::Cuda {
        major: parse_nvidia_smi_cuda_major(&String::from_utf8_lossy(&output.stdout)),
    })
}

/// Parses the CUDA runtime major version out of `nvidia-smi`'s
/// human-readable header (e.g. "... CUDA Version: 13.0 ..."). Defaults to
/// 12 if the version can't be found, since that's the more broadly
/// compatible of llama.cpp's two published CUDA builds.
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn parse_nvidia_smi_cuda_major(text: &str) -> u32 {
    text.split("CUDA Version:")
        .nth(1)
        .and_then(|rest| rest.trim().split(['.', ' ']).next())
        .and_then(|v| v.parse().ok())
        .unwrap_or(12)
}

#[cfg(all(test, any(target_os = "linux", target_os = "windows")))]
mod tests {
    use super::*;

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
    fn parses_cuda_major_from_nvidia_smi_header() {
        assert_eq!(parse_nvidia_smi_cuda_major(SMI_HEADER_124), 12);
        assert_eq!(parse_nvidia_smi_cuda_major(SMI_HEADER_130), 13);
    }

    #[test]
    fn defaults_to_12_when_unparseable() {
        assert_eq!(parse_nvidia_smi_cuda_major("garbage, no version here"), 12);
    }
}

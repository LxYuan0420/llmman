# llmman installer (Windows) — downloads the llmman.exe binary matching
# this host's CPU architecture from GitHub Releases and installs it to
# %LOCALAPPDATA%\Microsoft\WindowsApps, which is already on PATH for every
# user account by default on Windows 10/11. For Linux/macOS, use
# install.sh instead.
#
#   irm https://raw.githubusercontent.com/ericcurtin/llmman/main/install.ps1 | iex
#
# Modeled after llama.cpp's own installer
# (https://github.com/ggml-org/llama-install.sh, served at
# https://llama.app/install.ps1) but considerably simpler: that script has
# to probe the host for CUDA/Vulkan itself and download one of several
# GPU-specific `llama` wrapper builds, because llama.cpp publishes a
# separate binary per backend. llmman publishes exactly one binary per
# architecture (see .github/workflows/ci.yml's release job) — the
# GPU/backend detection instead happens *inside* that one binary, at
# runtime, the first time `llmman serve` actually needs a `llama-server`:
# it downloads and caches whichever prebuilt llama.cpp release build
# (CPU, Vulkan, ROCm, CUDA) matches whatever it finds on the host (see
# src/hostgpu.rs and src/llama_release.rs). This script's own job is just
# getting that one llmman.exe itself onto PATH — but, as a convenience
# (and so llama.cpp's own CLI is available too, not just through
# llmman), it finishes by handing off to that same upstream installer
# directly:
#
#   irm https://llama.app/install.ps1 | iex
#
# That's best-effort and non-fatal here: llmman itself is already fully
# installed by the time this runs, and falls back to its own
# hostgpu.rs/llama_release.rs download regardless of whether it succeeds.
#
# Supported today (matches .github/workflows/ci.yml's build matrix):
#   Windows x86_64, aarch64
#
# Env overrides:
#   LLMMAN_VERSION       pin an exact release tag (e.g. "v0.2.0"); default: latest
#   LLMMAN_REPO          "owner/repo" to fetch from; default: ericcurtin/llmman
#   SKIP_INSTALL         download and verify only, don't install
#   SKIP_LLAMA_INSTALL   don't hand off to llama.app/install.ps1 at the end

function Die {
    param([string[]]$Messages)
    $Messages | ForEach-Object { [Console]::Error.WriteLine($_) }
    exit 111
}

function Main {
    $Repo = $env:LLMMAN_REPO
    if (!$Repo) { $Repo = "ericcurtin/llmman" }

    switch ($env:PROCESSOR_ARCHITECTURE) {
        "ARM64" { $Target = "aarch64-pc-windows-msvc" }
        "AMD64" { $Target = "x86_64-pc-windows-msvc" }
        default { Die "Arch not supported: $env:PROCESSOR_ARCHITECTURE" }
    }

    $Asset = "llmman-$Target.exe"
    $Version = $env:LLMMAN_VERSION
    if ($Version) {
        $Url = "https://github.com/$Repo/releases/download/$Version/$Asset"
        "Version: $Version"
    } else {
        $Url = "https://github.com/$Repo/releases/latest/download/$Asset"
        "Version: latest"
    }

    $Dir = Join-Path ([System.IO.Path]::GetTempPath()) "llmman-install-$PID"
    New-Item -Path $Dir -Force -ItemType Directory | Out-Null
    try {
        $Tmp = Join-Path $Dir "llmman.exe"
        "Downloading $Asset..."
        try {
            Invoke-WebRequest -Uri $Url -OutFile $Tmp -UseBasicParsing
        } catch {
            Die "Failed to download $Url" `
                "(has a release for $Target been published yet? see .github/workflows/ci.yml)"
        }

        & $Tmp --version *> $null
        if ($LASTEXITCODE) {
            Die "Downloaded llmman binary failed to run"
        }

        if ($env:SKIP_INSTALL) {
            "Download verified, installation skipped (SKIP_INSTALL is set): $Tmp"
            return
        }

        $InstallDir = Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps"
        $Dest = Join-Path $InstallDir "llmman.exe"
        if (Test-Path $Dest) {
            Remove-Item "$Dest.old" -Force -ErrorAction SilentlyContinue
            Move-Item $Dest "$Dest.old" -Force
        }
        Move-Item $Tmp $Dest -Force

        "Installation completed successfully"
        ""
        "Run the following command to start it:"
        ""
        "  llmman serve"
        ""
    } finally {
        Remove-Item $Dir -Recurse -Force -ErrorAction SilentlyContinue
    }

    Install-Llama
}

# Hands off to llama.cpp's own installer — see this script's header
# comment for why. Best-effort: llmman is already fully installed by the
# time this runs, so a failure here (network hiccup, llama.app
# unreachable, ...) is only a warning, never fatal — `llmman serve` falls
# back to downloading its own llama-server regardless (see
# src/llama_release.rs).
function Install-Llama {
    if ($env:SKIP_LLAMA_INSTALL) { return }
    ""
    "Installing llama.cpp (llama.app)..."
    try {
        Invoke-RestMethod https://llama.app/install.ps1 | Invoke-Expression
    } catch {
        [Console]::Error.WriteLine(
            "Warning: llama.app installer failed ($_); continuing (llmman serve will fetch its own llama-server automatically)"
        )
    }
}

Main

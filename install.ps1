# llmman installer (Windows) — downloads the llmman.exe binary matching
# this host's CPU architecture from GitHub Releases and installs it to
# %LOCALAPPDATA%\Microsoft\WindowsApps, which is already on PATH for every
# user account by default on Windows 10/11. For Linux/macOS, use
# install.sh instead.
#
#   irm https://raw.githubusercontent.com/llmmanorg/llmman/main/install.ps1 | iex
#
# GPU/backend detection happens at runtime inside the llmman binary
# itself, the first time `llmman serve` needs a `llama-server` (see
# src/hostgpu.rs and src/llama_release.rs) — this script just gets
# llmman.exe onto PATH.
#
# Supported today (matches .github/workflows/ci.yml's build matrix):
#   Windows x86_64, aarch64
#
# Env overrides:
#   LLMMAN_VERSION   pin an exact release tag (e.g. "v0.2.0"); default: latest
#   LLMMAN_REPO      "owner/repo" to fetch from; default: llmmanorg/llmman
#   SKIP_INSTALL     download and verify only, don't install

function Die {
    param([string[]]$Messages)
    $Messages | ForEach-Object { [Console]::Error.WriteLine($_) }
    exit 111
}

function Main {
    $Repo = $env:LLMMAN_REPO
    if (!$Repo) { $Repo = "llmmanorg/llmman" }

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
    } finally {
        Remove-Item $Dir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Main

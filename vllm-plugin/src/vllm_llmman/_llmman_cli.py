"""Subprocess wrapper around `llmman resolve` (see llmman's
`src/cmd/resolve.rs`). This is the only way this plugin talks to llmman:
there are no Python bindings, just the CLI binary and its documented JSON
output contract, so any `llmman` release (fetched via its own
`install.sh`/`install.ps1`, exactly as documented in llmman's own README)
already satisfies this plugin's only runtime dependency.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess


class LlmmanNotFoundError(RuntimeError):
    """Raised when the `llmman` binary can't be located on PATH."""


def llmman_binary() -> str:
    """Locate the `llmman` binary: `LLMMAN_BIN` if set, else the first
    `llmman` found on `PATH`.
    """
    override = os.environ.get("LLMMAN_BIN")
    if override:
        return override
    found = shutil.which("llmman")
    if found:
        return found
    raise LlmmanNotFoundError(
        "the `llmman` binary was not found on PATH. Install it with "
        "`curl -fsSL https://raw.githubusercontent.com/llmmanorg/llmman/"
        "main/install.sh | sh` (see https://github.com/llmmanorg/llmman#install "
        "for other platforms), or set LLMMAN_BIN to its full path."
    )


def resolve(
    reference: str,
    *,
    cache: str | None = None,
    binary: str | None = None,
) -> dict:
    """Run `llmman resolve <reference>` and return its parsed JSON output:
    `{"reference": ..., "path": ..., "format": "safetensors" | "gguf"}`.

    Pulls `reference` into llmman's local store first if it isn't already
    present — this call blocks the calling thread until that finishes,
    the same as vLLM's own `huggingface_hub.snapshot_download` call for a
    HuggingFace model does today.

    Raises `RuntimeError` if the subprocess exits non-zero or its stdout
    isn't the expected single line of JSON.
    """
    cmd = [binary or llmman_binary(), "resolve", reference]
    if cache:
        cmd += ["--cache", cache]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"`llmman resolve {reference}` failed (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )

    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(
            f"`llmman resolve {reference}` produced no output on stdout"
        )
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"`llmman resolve {reference}` produced unparseable output: "
            f"{lines[-1]!r}"
        ) from e

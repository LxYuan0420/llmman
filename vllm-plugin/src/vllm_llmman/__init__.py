"""vllm-llmman — a vLLM plugin that lets `--model modelpack://...` /
`--model oci://...` pull a CNCF ModelPack (https://github.com/modelpack/
model-spec) OCI image via `llmman` instead of vLLM's usual HuggingFace
Hub download.

Usage::

    pip install vllm-llmman
    # llmman itself must also be on PATH — see llmman's own install.sh
    vllm serve modelpack://ghcr.io/org/model:tag

No vLLM core files are modified: this package only uses vLLM's existing
`vllm.general_plugins` entry-point mechanism (see
docs/design/plugin_system.md in the vLLM repo) plus a narrowly scoped
runtime monkeypatch — see `_patch.py`'s module docstring for exactly
what and why.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["register"]


def register() -> None:
    """Entry point for vLLM's `vllm.general_plugins` group (see
    `pyproject.toml`). Invoked once per process by
    `vllm.plugins.load_general_plugins()`, before any `ModelConfig` is
    constructed.
    """
    from ._patch import install

    install()

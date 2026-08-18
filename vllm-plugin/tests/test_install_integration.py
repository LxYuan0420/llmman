"""End-to-end check against the real `vllm.config.model.ModelConfig` class.

Skipped unless vLLM is actually importable in the current environment —
`_patch.py`'s unit tests (test_patch.py) cover the resolution logic itself
without needing vLLM installed at all; this file only exists to catch
`vllm.config.model.ModelConfig.maybe_pull_model_tokenizer_for_runai`
disappearing or changing signature in a future vLLM release.
"""

import pytest

vllm_config_model = pytest.importorskip("vllm.config.model")


def test_install_replaces_the_runai_hook_exactly_once():
    from vllm_llmman import _patch

    ModelConfig = vllm_config_model.ModelConfig
    original = ModelConfig.maybe_pull_model_tokenizer_for_runai

    _patch.install()
    patched_once = ModelConfig.maybe_pull_model_tokenizer_for_runai
    assert patched_once is not original

    _patch.install()  # idempotent
    assert ModelConfig.maybe_pull_model_tokenizer_for_runai is patched_once

    # Restore, so this test doesn't leak global state into others.
    ModelConfig.maybe_pull_model_tokenizer_for_runai = original
    delattr(ModelConfig, _patch._PATCHED_ATTR)

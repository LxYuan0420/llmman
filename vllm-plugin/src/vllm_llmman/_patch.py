"""Hooks `model=oci://...` into vLLM's model resolution *before* vLLM's
own HuggingFace-oriented config/tokenizer loading ever runs — without
editing any vLLM core file.

vLLM already has exactly one hook that does "rewrite `self.model`/`self.
tokenizer` from a remote reference to a local directory, early in
`ModelConfig.__post_init__`, before anything else consumes them":
`ModelConfig.maybe_pull_model_tokenizer_for_runai` (today only wired up
for `s3://`/`gs://`/`az://`, via `is_runai_obj_uri`). There is no
equivalent `register_model_source_resolver()` extension point (unlike
`register_model_loader`/`register_config_parser`), so the only way to
reach this point from an out-of-tree plugin is to wrap that method at
runtime. See vLLM's `vllm/config/model.py` (`__post_init__`, line ~565)
and `vllm/transformers_utils/runai_utils.py` for the method being
wrapped.

`_make_patched` is a pure function (no vLLM import) so it's unit
testable against a stub object; `install()` is the only piece that
actually touches `vllm.config.model.ModelConfig`.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol

from ._llmman_cli import resolve
from ._scheme import is_oci_ref, strip_scheme

logger = logging.getLogger("vllm_llmman")

_PATCHED_ATTR = "_vllm_llmman_patched"


class _ModelConfigLike(Protocol):
    model_weights: str | None
    model: str
    tokenizer: str


OriginalHook = Callable[[Any, str, str], None]


def _make_patched(original: OriginalHook) -> OriginalHook:
    """Build the replacement for `ModelConfig.
    maybe_pull_model_tokenizer_for_runai`. Delegates to `original`
    unchanged whenever neither `model` nor `tokenizer` uses a scheme this
    plugin recognizes, so every existing runai (`s3://`/`gs://`/`az://`)
    code path keeps working exactly as before.
    """

    def patched(self: _ModelConfigLike, model: str, tokenizer: str) -> None:
        if self.model_weights:
            return original(self, model, tokenizer)

        model_is_oci = is_oci_ref(model)
        tokenizer_is_oci = is_oci_ref(tokenizer)
        if not (model_is_oci or tokenizer_is_oci):
            return original(self, model, tokenizer)

        if model_is_oci:
            ref = strip_scheme(model)
            logger.info("llmman: resolving model %s", ref)
            result = resolve(ref)
            self.model_weights = model
            self.model = result["path"]

            if model == tokenizer:
                self.tokenizer = result["path"]
                return

        if tokenizer_is_oci and tokenizer != model:
            ref = strip_scheme(tokenizer)
            logger.info("llmman: resolving tokenizer %s", ref)
            result = resolve(ref)
            self.tokenizer = result["path"]

    patched.__name__ = getattr(original, "__name__", "maybe_pull_model_tokenizer_for_runai")
    patched.__doc__ = original.__doc__
    return patched


def install() -> None:
    """Idempotently monkeypatch `vllm.config.model.ModelConfig` in the
    current process. Safe to call more than once (plugins may be loaded
    more than once per process — see `load_general_plugins`'s own
    warning) and safe to call from multiple processes (API server,
    engine core, workers all load `vllm.general_plugins` independently).
    """
    from vllm.config.model import ModelConfig

    if getattr(ModelConfig, _PATCHED_ATTR, False):
        return

    original = ModelConfig.maybe_pull_model_tokenizer_for_runai
    ModelConfig.maybe_pull_model_tokenizer_for_runai = _make_patched(original)
    setattr(ModelConfig, _PATCHED_ATTR, True)

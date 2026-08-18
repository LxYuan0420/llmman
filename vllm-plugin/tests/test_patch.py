from types import SimpleNamespace

import pytest

from vllm_llmman import _patch


def _stub(model, tokenizer, model_weights=None):
    return SimpleNamespace(model=model, tokenizer=tokenizer, model_weights=model_weights)


def test_delegates_to_original_when_model_weights_already_set():
    calls = []
    original = lambda self, model, tokenizer: calls.append((model, tokenizer))
    patched = _patch._make_patched(original)

    cfg = _stub("modelpack://ghcr.io/org/model:tag", "modelpack://ghcr.io/org/model:tag", model_weights="already-pulled")
    patched(cfg, cfg.model, cfg.tokenizer)

    assert calls == [(cfg.model, cfg.tokenizer)]
    assert cfg.model == "modelpack://ghcr.io/org/model:tag"  # untouched


def test_delegates_to_original_for_non_modelpack_refs():
    calls = []
    original = lambda self, model, tokenizer: calls.append((model, tokenizer))
    patched = _patch._make_patched(original)

    cfg = _stub("s3://bucket/model", "s3://bucket/model")
    patched(cfg, cfg.model, cfg.tokenizer)

    assert calls == [("s3://bucket/model", "s3://bucket/model")]


def test_resolves_model_and_shared_tokenizer(monkeypatch):
    seen_refs = []

    def fake_resolve(ref, **kwargs):
        seen_refs.append(ref)
        return {"reference": ref, "path": "/cache/model-dir", "format": "safetensors"}

    monkeypatch.setattr(_patch, "resolve", fake_resolve)
    patched = _patch._make_patched(original=lambda *a: pytest.fail("must not delegate"))

    ref = "modelpack://ghcr.io/org/model:tag"
    cfg = _stub(ref, ref)
    patched(cfg, cfg.model, cfg.tokenizer)

    assert seen_refs == ["ghcr.io/org/model:tag"]
    assert cfg.model == "/cache/model-dir"
    assert cfg.tokenizer == "/cache/model-dir"
    assert cfg.model_weights == ref


def test_resolves_model_and_distinct_tokenizer_separately(monkeypatch):
    seen_refs = []

    def fake_resolve(ref, **kwargs):
        seen_refs.append(ref)
        return {"reference": ref, "path": f"/cache/{ref.split('/')[-1]}", "format": "safetensors"}

    monkeypatch.setattr(_patch, "resolve", fake_resolve)
    patched = _patch._make_patched(original=lambda *a: pytest.fail("must not delegate"))

    model_ref = "modelpack://ghcr.io/org/model:tag"
    tok_ref = "oci://ghcr.io/org/tokenizer:tag"
    cfg = _stub(model_ref, tok_ref)
    patched(cfg, cfg.model, cfg.tokenizer)

    assert seen_refs == ["ghcr.io/org/model:tag", "ghcr.io/org/tokenizer:tag"]
    assert cfg.model == "/cache/model:tag"
    assert cfg.tokenizer == "/cache/tokenizer:tag"
    assert cfg.model_weights == model_ref


def test_resolves_tokenizer_only_when_model_is_not_modelpack(monkeypatch):
    def fake_resolve(ref, **kwargs):
        return {"reference": ref, "path": "/cache/tok", "format": "safetensors"}

    monkeypatch.setattr(_patch, "resolve", fake_resolve)
    patched = _patch._make_patched(original=lambda *a: pytest.fail("must not delegate"))

    cfg = _stub("meta-llama/Llama-3-8B", "modelpack://ghcr.io/org/tok:tag")
    patched(cfg, cfg.model, cfg.tokenizer)

    assert cfg.model == "meta-llama/Llama-3-8B"  # untouched, not a modelpack ref
    assert cfg.tokenizer == "/cache/tok"
    assert cfg.model_weights is None

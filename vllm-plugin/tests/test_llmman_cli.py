import stat
import sys

import pytest

from vllm_llmman._llmman_cli import LlmmanNotFoundError, llmman_binary, resolve


def _write_fake_llmman(path, body: str):
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_llmman_binary_prefers_llmman_bin_env(tmp_path, monkeypatch):
    fake = tmp_path / "llmman"
    fake.write_text("")
    monkeypatch.setenv("LLMMAN_BIN", str(fake))
    assert llmman_binary() == str(fake)


def test_llmman_binary_falls_back_to_path(tmp_path, monkeypatch):
    monkeypatch.delenv("LLMMAN_BIN", raising=False)
    fake = _write_fake_llmman(tmp_path / "llmman", "")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert llmman_binary() == fake


def test_llmman_binary_raises_when_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("LLMMAN_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))  # empty dir, nothing found
    with pytest.raises(LlmmanNotFoundError):
        llmman_binary()


def test_resolve_parses_successful_json_output(tmp_path):
    fake = _write_fake_llmman(
        tmp_path / "llmman",
        """
import sys, json
assert sys.argv[1:] == ["resolve", "ghcr.io/org/model:tag"]
print(json.dumps({"reference": sys.argv[2], "path": "/local/dir", "format": "safetensors"}))
""",
    )
    result = resolve("ghcr.io/org/model:tag", binary=fake)
    assert result == {
        "reference": "ghcr.io/org/model:tag",
        "path": "/local/dir",
        "format": "safetensors",
    }


def test_resolve_passes_cache_flag(tmp_path):
    fake = _write_fake_llmman(
        tmp_path / "llmman",
        """
import sys, json
assert sys.argv[1:] == ["resolve", "m", "--cache", "/cache"]
print(json.dumps({"reference": "m", "path": "/x", "format": "gguf"}))
""",
    )
    result = resolve("m", cache="/cache", binary=fake)
    assert result["format"] == "gguf"


def test_resolve_raises_on_nonzero_exit(tmp_path):
    fake = _write_fake_llmman(
        tmp_path / "llmman",
        """
import sys
sys.stderr.write("pull failed: not found\\n")
sys.exit(1)
""",
    )
    with pytest.raises(RuntimeError, match="pull failed"):
        resolve("missing:ref", binary=fake)


def test_resolve_raises_on_unparseable_output(tmp_path):
    fake = _write_fake_llmman(
        tmp_path / "llmman",
        """
print("not json")
""",
    )
    with pytest.raises(RuntimeError, match="unparseable"):
        resolve("m", binary=fake)

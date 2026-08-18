"""Recognizes the URI schemes this plugin handles.

Deliberately requires an explicit scheme (`modelpack://` or `oci://`)
rather than trying to guess from a bare `registry/name:tag`-shaped
string: that shape is indistinguishable from a HuggingFace repo id
(`org/model`), and silently hijacking those would break every existing
HuggingFace-backed `vllm serve org/model` invocation once this plugin is
installed.
"""

from __future__ import annotations

SCHEMES: tuple[str, ...] = ("modelpack://", "oci://")


def is_modelpack_ref(value: str | None) -> bool:
    """True if `value` uses one of the schemes this plugin resolves."""
    return bool(value) and value.lower().startswith(SCHEMES)


def strip_scheme(value: str) -> str:
    """Drop a recognized scheme prefix, leaving the bare reference `llmman`
    itself understands (e.g. "modelpack://ghcr.io/org/model:tag" ->
    "ghcr.io/org/model:tag"). Returns `value` unchanged if no recognized
    scheme prefixes it.
    """
    lowered = value.lower()
    for scheme in SCHEMES:
        if lowered.startswith(scheme):
            return value[len(scheme) :]
    return value

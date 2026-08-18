"""Recognizes the URI scheme this plugin handles.

Deliberately requires an explicit `oci://` scheme rather than trying to
guess from a bare `registry/name:tag`-shaped string: that shape is
indistinguishable from a HuggingFace repo id (`org/model`), and silently
hijacking those would break every existing HuggingFace-backed
`vllm serve org/model` invocation once this plugin is installed.
"""

from __future__ import annotations

SCHEME = "oci://"


def is_oci_ref(value: str | None) -> bool:
    """True if `value` uses the `oci://` scheme this plugin resolves."""
    return bool(value) and value.lower().startswith(SCHEME)


def strip_scheme(value: str) -> str:
    """Drop the `oci://` prefix, leaving the bare reference `llmman`
    itself understands (e.g. "oci://ghcr.io/org/model:tag" ->
    "ghcr.io/org/model:tag"). Returns `value` unchanged if it isn't
    prefixed with `oci://`.
    """
    if value.lower().startswith(SCHEME):
        return value[len(SCHEME) :]
    return value

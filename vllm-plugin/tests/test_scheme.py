from vllm_llmman._scheme import is_modelpack_ref, strip_scheme


def test_recognizes_modelpack_scheme():
    assert is_modelpack_ref("modelpack://ghcr.io/org/model:tag")


def test_recognizes_oci_scheme():
    assert is_modelpack_ref("oci://ghcr.io/org/model:tag")


def test_recognizes_scheme_case_insensitively():
    assert is_modelpack_ref("MODELPACK://ghcr.io/org/model:tag")


def test_rejects_none():
    assert not is_modelpack_ref(None)


def test_rejects_bare_hf_repo_id():
    # This is the whole reason an explicit scheme is required: an
    # "org/model"-shaped HF repo id must never be silently hijacked.
    assert not is_modelpack_ref("meta-llama/Llama-3-8B")


def test_rejects_plain_local_path():
    assert not is_modelpack_ref("/models/local-checkpoint")


def test_strip_scheme_removes_modelpack_prefix():
    assert strip_scheme("modelpack://ghcr.io/org/model:tag") == "ghcr.io/org/model:tag"


def test_strip_scheme_removes_oci_prefix():
    assert strip_scheme("oci://ghcr.io/org/model:tag") == "ghcr.io/org/model:tag"


def test_strip_scheme_is_noop_without_a_recognized_scheme():
    assert strip_scheme("ghcr.io/org/model:tag") == "ghcr.io/org/model:tag"

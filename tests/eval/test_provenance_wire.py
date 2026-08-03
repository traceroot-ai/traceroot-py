"""structured run provenance is built in the backend's flat wire shape and
reported at registration (git/CI/SDK identity + free-form metadata, honest omission)."""

from traceroot.eval.platform import PlatformTransport
from traceroot.eval.provenance import run_provenance


def test_run_provenance_flat_wire_shape():
    # CI env drives the deterministic fields (branch + CI + SDK identity). git_repository/
    # git_commit come from the SDK's already-resolved git context and are asserted only
    # structurally (machine-dependent), never with a hard-coded value.
    env = {
        "GITHUB_REF_NAME": "main",
        "GITHUB_ACTIONS": "true",
        "GITHUB_RUN_ID": "42",
    }
    prov = run_provenance(env=env, detect_dirty=False)

    assert prov["sdk_language"] == "python"
    assert "sdk_version" in prov and isinstance(prov["sdk_version"], str)
    # git_ref is the BRANCH (from CI env), distinct from the commit SHA (never duplicated).
    assert prov["git_ref"] == "main"
    if "git_commit" in prov:
        assert prov["git_ref"] != prov["git_commit"]
        assert isinstance(prov["git_commit"], str)
    assert prov["ci_provider"] == "github"
    assert prov["ci_build_id"] == "42"
    # dirty not requested -> omitted, never null-filled
    assert "git_dirty" not in prov
    # model/prompt identity is never auto-inferred by the SDK
    assert "declared_model" not in prov
    assert "declared_prompt_version" not in prov


def test_run_provenance_reports_sdk_identity_without_ci():
    prov = run_provenance(env={"PATH": "/usr/bin"}, detect_dirty=False)
    # Even with no CI signal, provenance honestly reports the SDK identity.
    assert prov["sdk_language"] == "python"
    assert "ci_provider" not in prov
    assert "ci_build_id" not in prov


def _capturing_transport() -> tuple[PlatformTransport, dict]:
    t = PlatformTransport(
        dataset_id="ds", scorer_names=["quality"], api_key="k", host_url="http://h"
    )
    captured: dict = {}

    def _req(method, path, body=None):
        captured["method"], captured["path"], captured["body"] = method, path, body
        return {"evaluation_run_id": "run_x"}

    t._request = _req  # type: ignore[method-assign]
    return t, captured


def test_create_run_reports_provenance_and_metadata():
    t, captured = _capturing_transport()
    t.create_run(
        name="eval",
        dataset_name="ds",
        metadata={"team": "quality"},
        client_run_id="crun_1",
        provenance={"sdk_language": "python", "git_commit": "abc"},
    )
    body = captured["body"]
    assert body["provenance"] == {"sdk_language": "python", "git_commit": "abc"}
    assert body["metadata"] == {"team": "quality"}
    assert body["client_run_id"] == "crun_1"
    assert captured["path"] == "/api/v1/public/evaluation-runs"


def test_create_run_omits_absent_provenance_and_metadata():
    t, captured = _capturing_transport()
    t.create_run(name="eval", dataset_name="ds", metadata=None, provenance=None)
    body = captured["body"]
    assert "provenance" not in body  # absent, not null-filled
    assert "metadata" not in body

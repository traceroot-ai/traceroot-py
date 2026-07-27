"""Phase 0B: artifact hygiene -- dir perms, .gitignore, bounded payloads, and a proof
that credentials never land in a written artifact."""

import json
import os
import stat
import sys

from traceroot.eval.results import (
    EvalItemResult,
    EvalRunResult,
    Score,
    UploadState,
)
from traceroot.eval.runner import write_artifacts

SENTINEL_KEY = "tr-SECRET-should-never-persist-abc123"


def _item(case_id, inp, out):
    return EvalItemResult(
        case_id=case_id,
        input=inp,
        output=out,
        expected=out,
        scores=[Score("acc", 1.0)],
        scorer_errors={},
        error=None,
        trace_id=None,
        duration_ms=1.0,
    )


def _run(items):
    return EvalRunResult(
        name="r",
        item_results=items,
        score_summary={},
        upload_state=UploadState(status="local_only"),
        local_run_id="lr_test",
        candidate_version="v1",
    )


_OPTS = dict(
    status="completed",
    run_mode="module",
    is_final=True,
    sample_count=None,
    sample_seed=None,
    candidate_version="v1",
    provenance=None,
)


def _paths(tmp_path):
    return tmp_path / "lr.json", tmp_path / "lr.cases.jsonl"


def test_credentials_never_appear_in_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACEROOT_API_KEY", SENTINEL_KEY)
    run_path, cases_path = _paths(tmp_path)
    write_artifacts(_run([_item("c0", {"q": "hi"}, {"a": "ok"})]), run_path, cases_path, **_OPTS)
    assert SENTINEL_KEY not in run_path.read_text()
    assert SENTINEL_KEY not in cases_path.read_text()


def test_gitignore_written(tmp_path):
    run_path, cases_path = _paths(tmp_path)
    write_artifacts(_run([_item("c0", 1, 1)]), run_path, cases_path, **_OPTS)
    gi = tmp_path / ".gitignore"
    assert gi.exists()
    assert "*" in gi.read_text().splitlines()


def test_artifact_dir_permissions_posix(tmp_path):
    if sys.platform == "win32":
        return
    run_path, cases_path = _paths(tmp_path)
    write_artifacts(_run([_item("c0", 1, 1)]), run_path, cases_path, **_OPTS)
    assert stat.S_IMODE(os.stat(tmp_path).st_mode) == 0o700


def test_payload_truncation_marker(tmp_path):
    run_path, cases_path = _paths(tmp_path)
    big = "x" * 5000
    artifact = write_artifacts(
        _run([_item("c0", {"blob": big}, {"blob": big})]),
        run_path,
        cases_path,
        max_payload_bytes=64,
        **_OPTS,
    )
    assert artifact["payloads"] == "truncated"
    rec = json.loads(cases_path.read_text().strip())
    assert rec["input"]["truncated"] is True
    assert len(rec["input"]["preview"]) <= 64


def test_payloads_complete_by_default(tmp_path):
    run_path, cases_path = _paths(tmp_path)
    artifact = write_artifacts(
        _run([_item("c0", {"blob": "x" * 5000}, 1)]), run_path, cases_path, **_OPTS
    )
    assert artifact["payloads"] == "complete"

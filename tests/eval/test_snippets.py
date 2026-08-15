"""SDK owns the canonical copy-paste snippets so UI text always matches the SDK signature.

Mental model: bringing something local always means pulling DATA (a dataset at a version).
A run is task + scorers + data; task/scorers live in user code, so a run is reproduced by
pulling its dataset_version_id -- there is intentionally no pull_run/pull_evaluation.
"""

import ast

import pytest

from traceroot.eval import (
    dataset_latest_snippet,
    dataset_version_snippet,
    reproduce_run_snippet,
)


def _py_compiles(src: str) -> bool:
    ast.parse(src)  # raises SyntaxError if invalid
    return True


class TestPythonSnippets:
    def test_dataset_latest_uses_pull_dataset(self):
        s = dataset_latest_snippet("ds_1", lang="python")
        assert "pull_dataset(" in s and "pull_dataset_version" not in s
        assert '"ds_1"' in s
        assert _py_compiles(s)

    def test_dataset_version_uses_pull_dataset_version(self):
        s = dataset_version_snippet("dsv_9", lang="python")
        assert "pull_dataset_version(" in s
        assert '"dsv_9"' in s
        assert _py_compiles(s)

    def test_reproduce_run_pulls_version_then_commented_evaluate(self):
        s = reproduce_run_snippet("dsv_9", lang="python")
        # pulls the exact version the run used
        assert 'pull_dataset_version("dsv_9")' in s
        # evaluate is present but COMMENTED (user supplies task + scorers)
        assert "evaluate(" in s
        assert "# " in s and "task" in s and "scorers" in s
        # no pull_run / pull_evaluation anywhere
        assert "pull_run" not in s and "pull_evaluation" not in s
        assert _py_compiles(s)  # uncommented lines (import + pull) are valid


class TestTypeScriptSnippets:
    def test_dataset_latest_ts(self):
        s = dataset_latest_snippet("ds_1", lang="typescript")
        assert "pullDataset(" in s and "pullDatasetVersion" not in s
        assert '"ds_1"' in s
        assert "await" in s
        assert 'from "traceroot"' in s

    def test_dataset_version_ts(self):
        s = dataset_version_snippet("dsv_9", lang="typescript")
        assert "pullDatasetVersion(" in s
        assert '"dsv_9"' in s
        assert "await" in s

    def test_reproduce_run_ts_commented_evaluate(self):
        s = reproduce_run_snippet("dsv_9", lang="typescript")
        assert 'pullDatasetVersion("dsv_9")' in s
        assert "evaluate(" in s
        assert "// " in s  # evaluate block commented out
        assert "candidateVersion" in s  # ts camelCase
        assert "pull_run" not in s


class TestReproduceRunAcceptance:
    """Acceptance: pull_dataset_version(run.dataset_version_id) returns the exact cases the
    run scored, exposes .dataset_version_id (round-trip pinning), and is evaluate-able."""

    def test_reproduce_run_from_dataset_version_id(self, monkeypatch):
        from traceroot.eval import Dataset, evaluate, pull_dataset_version

        # A run recorded this dataset_version_id; the platform returns the exact snapshot.
        run_dsv_id = "dsv_run_42"
        snapshot = {
            "dataset_id": "ds_1",
            "dataset_version_id": run_dsv_id,
            "items": [
                {"test_case_id": "c0", "input": {"q": "a"}, "expected": {"a": 1}},
                {"test_case_id": "c1", "input": {"q": "b"}, "expected": {"a": 2}},
            ],
        }
        monkeypatch.setattr(
            "traceroot.eval.platform._http_get_json",
            lambda url, api_key: snapshot,
        )

        ds = pull_dataset_version(run_dsv_id, api_key="tr-x", host_url="https://h")

        # exact cases, same revision, pinned id round-trips
        assert isinstance(ds, Dataset)
        assert ds.dataset_version_id == run_dsv_id
        assert [c.id for c in ds] == ["c0", "c1"]
        assert ds.get("c0").input == {"q": "a"} and ds.get("c0").expected == {"a": 1}

        # ...and it is directly evaluate-able (bring your own task + scorers)
        result = evaluate(
            name="repro",
            dataset=ds,
            task=lambda x: x,
            scorers=[lambda ctx: 1.0],
        )
        assert [it.case_id for it in result.item_results] == ["c0", "c1"]
        # the run result still carries the pinned version it reproduced
        assert result.dataset.dataset_version_id == run_dsv_id


class TestSnippetPlaceholders:
    def test_placeholder_when_no_id(self):
        assert "<dataset_id>" in dataset_latest_snippet(lang="python")
        assert "<dataset_version_id>" in dataset_version_snippet(lang="python")
        assert "<dataset_version_id>" in reproduce_run_snippet(lang="python")

    def test_unknown_language_raises(self):
        with pytest.raises(ValueError):
            dataset_latest_snippet("ds_1", lang="ruby")

"""Run metadata + provenance collection (machine-independent -- never depends on
the test machine's real git repo or CI env)."""

import traceroot.eval.provenance as prov
from traceroot.eval import Dataset, EvalCase, evaluate
from traceroot.eval.provenance import collect_run_provenance
from traceroot.eval.transport import FakeTransport


def echo(x):
    return x


def acc(ctx):
    return 1.0


class TestCollectRunProvenance:
    def test_ci_github_provider_and_build_id(self, monkeypatch):
        monkeypatch.setattr(prov, "_resolved_git", lambda env: (None, None))
        env = {"GITHUB_ACTIONS": "true", "GITHUB_RUN_ID": "42"}
        meta = collect_run_provenance(env=env, detect_dirty=False)
        assert meta == {"ci": {"provider": "github", "build_id": "42"}}

    def test_generic_ci_without_known_provider(self, monkeypatch):
        monkeypatch.setattr(prov, "_resolved_git", lambda env: (None, None))
        meta = collect_run_provenance(env={"CI": "1"}, detect_dirty=False)
        assert meta == {"ci": {"provider": "ci"}}

    def test_git_block_repository_ref_commit(self, monkeypatch):
        # A SHA-looking ref is exposed as both ref and commit.
        monkeypatch.setattr(
            prov,
            "_resolved_git",
            lambda env: ("owner/repo", "0123456789abcdef0123456789abcdef01234567"),
        )
        meta = collect_run_provenance(env={}, detect_dirty=False)
        assert meta == {
            "git": {
                "repository": "owner/repo",
                "ref": "0123456789abcdef0123456789abcdef01234567",
                "commit": "0123456789abcdef0123456789abcdef01234567",
            }
        }

    def test_git_block_branch_ref_has_no_commit(self, monkeypatch):
        # A branch/tag ref is not a commit SHA -> exposed as ref only, never as commit.
        monkeypatch.setattr(prov, "_resolved_git", lambda env: ("owner/repo", "main"))
        meta = collect_run_provenance(env={}, detect_dirty=False)
        assert meta == {"git": {"repository": "owner/repo", "ref": "main"}}

    def test_dirty_flag_when_requested(self, monkeypatch):
        monkeypatch.setattr(prov, "_resolved_git", lambda env: ("owner/repo", "abc123"))
        monkeypatch.setattr(prov, "_git_dirty", lambda: True)
        meta = collect_run_provenance(env={}, detect_dirty=True)
        assert meta["git"]["dirty"] is True

    def test_user_metadata_preserved_and_wins(self, monkeypatch):
        monkeypatch.setattr(prov, "_resolved_git", lambda env: ("owner/repo", "abc123"))
        meta = collect_run_provenance(
            {"model": "claude-sonnet", "prompt_version": "v12", "git": "USER"},
            env={"GITHUB_ACTIONS": "true", "GITHUB_RUN_ID": "9"},
            detect_dirty=False,
        )
        assert meta["model"] == "claude-sonnet" and meta["prompt_version"] == "v12"
        assert meta["git"] == "USER"  # user key wins over auto git
        assert meta["ci"] == {"provider": "github", "build_id": "9"}

    def test_nothing_available_is_none(self, monkeypatch):
        monkeypatch.setattr(prov, "_resolved_git", lambda env: (None, None))
        assert collect_run_provenance(env={}, detect_dirty=False) is None

    def test_never_raises_on_git_failure(self, monkeypatch):
        # _git_dirty must degrade to None rather than raise, even if git blows up.
        def boom(*a, **k):
            raise OSError("no git")

        monkeypatch.setattr(prov.subprocess, "run", boom)
        assert prov._git_dirty() is None


class TestEngineAttachesProvenance:
    def test_result_metadata_merges_user_and_git(self, monkeypatch):
        monkeypatch.setattr(
            prov,
            "_resolved_git",
            lambda env: ("owner/repo", "0123456789abcdef0123456789abcdef01234567"),
        )
        ds = Dataset("d")
        ds.upsert(EvalCase(input=1, id="c0", expected=1))
        run = evaluate(name="r", dataset=ds, task=echo, scorers=[acc], metadata={"model": "x"})
        assert run.metadata["model"] == "x"
        assert run.metadata["git"] == {
            "repository": "owner/repo",
            "ref": "0123456789abcdef0123456789abcdef01234567",
            "commit": "0123456789abcdef0123456789abcdef01234567",
        }


class TestProvenanceReachesWire:
    """Reproducibility metadata (git commit/branch, CI) is NON-IDENTITY but must reach the
    platform as run metadata -- it was captured locally yet dropped from the wire alongside
    SDK-language identity. Braintrust sends the equivalent `repo_info` for reproducibility."""

    def test_git_and_ci_provenance_ride_the_run_metadata(self, monkeypatch):
        monkeypatch.setattr(
            prov,
            "collect_run_provenance",
            lambda user_metadata=None, **k: {"git": {"commit": "abc123"}, **(user_metadata or {})},
        )
        fake = FakeTransport()
        evaluate(
            name="r",
            dataset=[{"input": {"m": 1}}],
            task=echo,
            scorers=[acc],
            metadata={"team": "eval"},
            report_to=fake,
        )
        # git provenance reached create_run's metadata (not just the raw user dict), user keys kept
        assert fake.last_run_metadata["git"] == {"commit": "abc123"}
        assert fake.last_run_metadata["team"] == "eval"

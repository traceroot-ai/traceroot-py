"""Tests for git context resolution: harvest_ci_git_context and client integration.

Coverage:
- GitHub Actions, Vercel, GitLab CI, CircleCI, Bitbucket, Render
- First-platform-wins precedence (per field)
- Independent repo / ref resolution
- Empty environment → both None
- No fabricated defaults
- Never raises
- Warning emitted once when both values remain unresolved
- Warning suppressed when at least one value resolves
- Explicit client args bypass CI harvest
"""

import logging
from unittest.mock import patch

import pytest

import traceroot.client as client_module
from traceroot.git_context import harvest_ci_git_context

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_warn_guard() -> None:
    """Reset the module-level warn-once guard between tests."""
    client_module._git_context_warning_emitted = False


# ---------------------------------------------------------------------------
# harvest_ci_git_context — platform-specific tests
# ---------------------------------------------------------------------------


class TestHarvestCiGitContextPlatforms:
    """Each platform correctly populates git_repo and git_ref."""

    def test_github_actions_full(self) -> None:
        env = {
            "GITHUB_REPOSITORY": "myorg/myrepo",
            "GITHUB_SHA": "abc1234567890",
        }
        result = harvest_ci_git_context(env)
        assert result["git_repo"] == "myorg/myrepo"
        assert result["git_ref"] == "abc1234567890"

    def test_vercel_full(self) -> None:
        env = {
            "VERCEL_GIT_REPO_OWNER": "myorg",
            "VERCEL_GIT_REPO_SLUG": "myrepo",
            "VERCEL_GIT_COMMIT_SHA": "deadbeef",
        }
        result = harvest_ci_git_context(env)
        assert result["git_repo"] == "myorg/myrepo"
        assert result["git_ref"] == "deadbeef"

    def test_vercel_missing_owner_gives_none_repo(self) -> None:
        """Vercel repo requires both owner and slug; only slug → None."""
        env = {
            "VERCEL_GIT_REPO_SLUG": "myrepo",
            "VERCEL_GIT_COMMIT_SHA": "deadbeef",
        }
        result = harvest_ci_git_context(env)
        # repo cannot be formed without the owner
        assert result["git_repo"] is None
        assert result["git_ref"] == "deadbeef"

    def test_vercel_missing_slug_gives_none_repo(self) -> None:
        """Vercel repo requires both owner and slug; only owner → None."""
        env = {
            "VERCEL_GIT_REPO_OWNER": "myorg",
            "VERCEL_GIT_COMMIT_SHA": "deadbeef",
        }
        result = harvest_ci_git_context(env)
        assert result["git_repo"] is None
        assert result["git_ref"] == "deadbeef"

    def test_gitlab_ci_full(self) -> None:
        env = {
            "CI_PROJECT_PATH": "mygroup/myproject",
            "CI_COMMIT_SHA": "cafebabe",
        }
        result = harvest_ci_git_context(env)
        assert result["git_repo"] == "mygroup/myproject"
        assert result["git_ref"] == "cafebabe"

    def test_circleci_full(self) -> None:
        env = {
            "CIRCLE_PROJECT_USERNAME": "myorg",
            "CIRCLE_PROJECT_REPONAME": "myrepo",
            "CIRCLE_SHA1": "feed1234",
        }
        result = harvest_ci_git_context(env)
        assert result["git_repo"] == "myorg/myrepo"
        assert result["git_ref"] == "feed1234"

    def test_circleci_missing_reponame_gives_none_repo(self) -> None:
        env = {
            "CIRCLE_PROJECT_USERNAME": "myorg",
            "CIRCLE_SHA1": "feed1234",
        }
        result = harvest_ci_git_context(env)
        assert result["git_repo"] is None
        assert result["git_ref"] == "feed1234"

    def test_bitbucket_full(self) -> None:
        env = {
            "BITBUCKET_REPO_FULL_NAME": "myorg/myrepo",
            "BITBUCKET_COMMIT": "0badc0de",
        }
        result = harvest_ci_git_context(env)
        assert result["git_repo"] == "myorg/myrepo"
        assert result["git_ref"] == "0badc0de"

    def test_render_ref_only(self) -> None:
        """Render exposes only a ref; repo must remain None."""
        env = {
            "RENDER_GIT_COMMIT": "11223344",
        }
        result = harvest_ci_git_context(env)
        assert result["git_repo"] is None
        assert result["git_ref"] == "11223344"


# ---------------------------------------------------------------------------
# harvest_ci_git_context — empty environment
# ---------------------------------------------------------------------------


class TestHarvestCiGitContextEmptyEnv:
    def test_empty_env_returns_none_none(self) -> None:
        result = harvest_ci_git_context({})
        assert result == {"git_repo": None, "git_ref": None}


# ---------------------------------------------------------------------------
# harvest_ci_git_context — precedence
# ---------------------------------------------------------------------------


class TestHarvestCiGitContextPrecedence:
    def test_github_repo_wins_over_gitlab(self) -> None:
        """GitHub Actions is listed first; its repo var should win."""
        env = {
            "GITHUB_REPOSITORY": "gh-org/gh-repo",
            "GITHUB_SHA": "ghsha",
            "CI_PROJECT_PATH": "gl-group/gl-project",
            "CI_COMMIT_SHA": "glsha",
        }
        result = harvest_ci_git_context(env)
        assert result["git_repo"] == "gh-org/gh-repo"
        assert result["git_ref"] == "ghsha"

    def test_github_ref_wins_over_circleci_ref(self) -> None:
        env = {
            "GITHUB_SHA": "github-sha",
            "CIRCLE_SHA1": "circle-sha",
        }
        result = harvest_ci_git_context(env)
        assert result["git_ref"] == "github-sha"

    def test_circleci_repo_used_when_github_repo_absent(self) -> None:
        """Only GitHub SHA is set; repo must fall through to CircleCI."""
        env = {
            "GITHUB_SHA": "ghsha",
            "CIRCLE_PROJECT_USERNAME": "circle-org",
            "CIRCLE_PROJECT_REPONAME": "circle-repo",
        }
        result = harvest_ci_git_context(env)
        # GitHub has no repo var → CircleCI repo wins
        assert result["git_repo"] == "circle-org/circle-repo"
        # GitHub ref wins for ref
        assert result["git_ref"] == "ghsha"


# ---------------------------------------------------------------------------
# harvest_ci_git_context — independent resolution
# ---------------------------------------------------------------------------


class TestHarvestCiGitContextIndependentResolution:
    def test_repo_from_github_ref_from_gitlab(self) -> None:
        """Repo and ref can come from different platforms."""
        env = {
            "GITHUB_REPOSITORY": "gh-org/gh-repo",
            # No GITHUB_SHA → ref must come from GitLab
            "CI_COMMIT_SHA": "gl-sha",
        }
        result = harvest_ci_git_context(env)
        assert result["git_repo"] == "gh-org/gh-repo"
        assert result["git_ref"] == "gl-sha"

    def test_only_ref_resolved(self) -> None:
        """Only ref vars set; repo must be None, ref must be resolved."""
        env = {"GITHUB_SHA": "sha-only"}
        result = harvest_ci_git_context(env)
        assert result["git_repo"] is None
        assert result["git_ref"] == "sha-only"

    def test_only_repo_resolved(self) -> None:
        """Only repo vars set; git_ref must be None."""
        env = {"GITHUB_REPOSITORY": "org/repo"}
        result = harvest_ci_git_context(env)
        assert result["git_repo"] == "org/repo"
        assert result["git_ref"] is None


# ---------------------------------------------------------------------------
# harvest_ci_git_context — safety
# ---------------------------------------------------------------------------


class TestHarvestCiGitContextSafety:
    def test_never_raises_on_empty_dict(self) -> None:
        # Should not raise under any circumstance
        result = harvest_ci_git_context({})
        assert isinstance(result, dict)

    def test_never_raises_on_none_env(self) -> None:
        # Passing None should fall back to os.environ without raising
        result = harvest_ci_git_context(None)
        assert isinstance(result, dict)
        assert "git_repo" in result
        assert "git_ref" in result

    def test_no_fabricated_default_ref(self) -> None:
        """Result must never contain a hardcoded fabricated ref."""
        result = harvest_ci_git_context({})
        fabricated = {"main", "master", "HEAD", "develop"}
        assert result["git_ref"] not in fabricated

    def test_no_fabricated_default_repo(self) -> None:
        result = harvest_ci_git_context({})
        assert result["git_repo"] is None

    def test_empty_string_treated_as_absent(self) -> None:
        """Empty-string env vars must not be returned as a resolved value."""
        env = {
            "GITHUB_REPOSITORY": "",
            "GITHUB_SHA": "",
        }
        result = harvest_ci_git_context(env)
        assert result["git_repo"] is None
        assert result["git_ref"] is None


# ---------------------------------------------------------------------------
# client.py integration — warn-once behaviour
# ---------------------------------------------------------------------------


class TestClientWarnOnce:
    """Verify the warning behaviour inside TracerootClient.__init__."""

    @pytest.fixture(autouse=True)
    def reset_warn_flag(self) -> None:
        """Ensure warn-once guard is reset before every test in this class."""
        _reset_warn_guard()

    def _make_client(self, git_repo=None, git_ref=None, **kwargs):
        """Instantiate TracerootClient with subprocess and CI env fully mocked out."""
        with (
            # Prevent actual subprocess calls
            patch("traceroot.git_context.subprocess.check_output", side_effect=OSError),
            # Serve an empty CI environment
            patch(
                "traceroot.git_context.os.environ",
                new_callable=lambda: (lambda: {}),
            ) if False else patch("traceroot.client.harvest_ci_git_context", return_value={"git_repo": None, "git_ref": None}),
            patch("traceroot.client.auto_detect_git_context", return_value={"git_repo": None, "git_ref": None}),
            # Suppress TRACEROOT env vars
            patch.dict("os.environ", {}, clear=True),
        ):
            from traceroot.client import TracerootClient
            return TracerootClient(
                api_key="dummy-key",
                enabled=False,
                git_repo=git_repo,
                git_ref=git_ref,
                **kwargs,
            )

    def test_warning_emitted_when_both_unresolved(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="traceroot.client"):
            self._make_client()
        assert any("Git context could not be resolved" in r.message for r in caplog.records)

    def test_warning_contains_docs_link(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="traceroot.client"):
            self._make_client()
        full_text = " ".join(r.message for r in caplog.records)
        assert "https://docs.traceroot.ai/tracing/git-context" in full_text

    def test_warning_emitted_only_once(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="traceroot.client"):
            self._make_client()
            self._make_client()  # second call must not produce a second warning
        warn_records = [r for r in caplog.records if "Git context could not be resolved" in r.message]
        assert len(warn_records) == 1

    def test_no_warning_when_repo_resolved(self, caplog) -> None:
        with (
            caplog.at_level(logging.WARNING, logger="traceroot.client"),
            patch("traceroot.client.harvest_ci_git_context", return_value={"git_repo": None, "git_ref": None}),
            patch("traceroot.client.auto_detect_git_context", return_value={"git_repo": None, "git_ref": None}),
            patch.dict("os.environ", {}, clear=True),
        ):
            from traceroot.client import TracerootClient
            TracerootClient(api_key="dummy-key", enabled=False, git_repo="org/repo")
        assert not any("Git context could not be resolved" in r.message for r in caplog.records)

    def test_no_warning_when_ref_resolved(self, caplog) -> None:
        with (
            caplog.at_level(logging.WARNING, logger="traceroot.client"),
            patch("traceroot.client.harvest_ci_git_context", return_value={"git_repo": None, "git_ref": None}),
            patch("traceroot.client.auto_detect_git_context", return_value={"git_repo": None, "git_ref": None}),
            patch.dict("os.environ", {}, clear=True),
        ):
            from traceroot.client import TracerootClient
            TracerootClient(api_key="dummy-key", enabled=False, git_ref="abc123")
        assert not any("Git context could not be resolved" in r.message for r in caplog.records)

    def test_no_warning_when_both_resolved_via_ci(self, caplog) -> None:
        with (
            caplog.at_level(logging.WARNING, logger="traceroot.client"),
            patch(
                "traceroot.client.harvest_ci_git_context",
                return_value={"git_repo": "ci-org/ci-repo", "git_ref": "ci-sha"},
            ),
            patch("traceroot.client.auto_detect_git_context", return_value={"git_repo": None, "git_ref": None}),
            patch.dict("os.environ", {}, clear=True),
        ):
            from traceroot.client import TracerootClient
            TracerootClient(api_key="dummy-key", enabled=False)
        assert not any("Git context could not be resolved" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# client.py integration — resolution tier ordering
# ---------------------------------------------------------------------------


class TestClientResolutionOrder:
    """Explicit args and env vars skip lower-priority tiers."""

    @pytest.fixture(autouse=True)
    def reset_warn_flag(self) -> None:
        _reset_warn_guard()

    def test_explicit_args_skip_ci_harvest(self) -> None:
        """When both explicit args are provided, harvest_ci_git_context is not called."""
        with (
            patch("traceroot.client.harvest_ci_git_context") as mock_ci,
            patch("traceroot.client.auto_detect_git_context", return_value={"git_repo": None, "git_ref": None}),
            patch.dict("os.environ", {}, clear=True),
        ):
            from traceroot.client import TracerootClient
            c = TracerootClient(api_key="k", enabled=False, git_repo="org/repo", git_ref="sha")
        mock_ci.assert_not_called()
        assert c.git_repo == "org/repo"
        assert c.git_ref == "sha"

    def test_traceroot_env_vars_skip_ci_harvest(self) -> None:
        """When TRACEROOT_GIT_REPO and TRACEROOT_GIT_REF are set, CI is not harvested."""
        with (
            patch("traceroot.client.harvest_ci_git_context") as mock_ci,
            patch("traceroot.client.auto_detect_git_context", return_value={"git_repo": None, "git_ref": None}),
            patch.dict(
                "os.environ",
                {"TRACEROOT_GIT_REPO": "env-org/env-repo", "TRACEROOT_GIT_REF": "env-sha"},
                clear=True,
            ),
        ):
            from traceroot.client import TracerootClient
            c = TracerootClient(api_key="k", enabled=False)
        mock_ci.assert_not_called()
        assert c.git_repo == "env-org/env-repo"
        assert c.git_ref == "env-sha"

    def test_ci_context_used_when_traceroot_env_absent(self) -> None:
        """CI tier is consulted when TRACEROOT env vars are not set."""
        with (
            patch(
                "traceroot.client.harvest_ci_git_context",
                return_value={"git_repo": "ci-org/ci-repo", "git_ref": "ci-sha"},
            ) as mock_ci,
            patch("traceroot.client.auto_detect_git_context", return_value={"git_repo": None, "git_ref": None}),
            patch.dict("os.environ", {}, clear=True),
        ):
            from traceroot.client import TracerootClient
            c = TracerootClient(api_key="k", enabled=False)
        mock_ci.assert_called_once()
        assert c.git_repo == "ci-org/ci-repo"
        assert c.git_ref == "ci-sha"

    def test_auto_detect_skipped_when_ci_resolves_both(self) -> None:
        """auto_detect_git_context is not called when CI provides both values."""
        with (
            patch(
                "traceroot.client.harvest_ci_git_context",
                return_value={"git_repo": "ci-org/ci-repo", "git_ref": "ci-sha"},
            ),
            patch("traceroot.client.auto_detect_git_context") as mock_auto,
            patch.dict("os.environ", {}, clear=True),
        ):
            from traceroot.client import TracerootClient
            TracerootClient(api_key="k", enabled=False)
        mock_auto.assert_not_called()

    def test_auto_detect_called_when_ci_only_resolves_repo(self) -> None:
        """auto_detect_git_context is still consulted when CI resolves only one field."""
        with (
            patch(
                "traceroot.client.harvest_ci_git_context",
                return_value={"git_repo": "ci-org/ci-repo", "git_ref": None},
            ),
            patch(
                "traceroot.client.auto_detect_git_context",
                return_value={"git_repo": None, "git_ref": "git-sha"},
            ) as mock_auto,
            patch.dict("os.environ", {}, clear=True),
        ):
            from traceroot.client import TracerootClient
            c = TracerootClient(api_key="k", enabled=False)
        mock_auto.assert_called_once()
        assert c.git_repo == "ci-org/ci-repo"
        assert c.git_ref == "git-sha"

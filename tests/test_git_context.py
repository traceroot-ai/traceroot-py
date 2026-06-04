"""Tests for CI/platform git-context harvesting."""

import logging

import traceroot.client as client_mod
from traceroot.git_context import git_context_from_files, harvest_ci_git_context


def test_github_actions():
    r = harvest_ci_git_context({"GITHUB_REPOSITORY": "acme/web", "GITHUB_SHA": "a" * 40})
    assert r == {"git_repo": "acme/web", "git_ref": "a" * 40}


def test_empty_env():
    assert harvest_ci_git_context({}) == {"git_repo": None, "git_ref": None}


def test_empty_string_github_treated_as_absent():
    r = harvest_ci_git_context({"GITHUB_REPOSITORY": ""})
    assert r["git_repo"] is None


# ---------------------------------------------------------------------------
# git_context_from_files tests
# ---------------------------------------------------------------------------
def _write(p, content):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_files_detached_head_and_git_at_url(tmp_path):
    git_dir = tmp_path / ".git"
    _write(git_dir / "config", '[remote "origin"]\n\turl = git@github.com:acme/web.git\n')
    _write(git_dir / "HEAD", "f" * 40 + "\n")
    r = git_context_from_files(str(tmp_path))
    assert r["git_repo"] == "acme/web"
    assert r["git_ref"] == "f" * 40


def test_files_ref_based_head(tmp_path):
    git_dir = tmp_path / ".git"
    _write(git_dir / "HEAD", "ref: refs/heads/main\n")
    _write(git_dir / "refs" / "heads" / "main", "1" * 40 + "\n")
    r = git_context_from_files(str(tmp_path))
    assert r["git_ref"] == "1" * 40


def test_files_packed_refs(tmp_path):
    # HEAD points at a branch whose loose ref file is absent (gc'd / fresh clone);
    # the SHA lives in packed-refs.
    git_dir = tmp_path / ".git"
    _write(git_dir / "HEAD", "ref: refs/heads/main\n")
    _write(
        git_dir / "packed-refs",
        "# pack-refs with: peeled fully-peeled sorted\n" + "2" * 40 + " refs/heads/main\n",
    )
    r = git_context_from_files(str(tmp_path))
    assert r["git_ref"] == "2" * 40


def test_files_https_url(tmp_path):
    git_dir = tmp_path / ".git"
    _write(git_dir / "config", '[remote "origin"]\n\turl = https://github.com/acme/web.git\n')
    r = git_context_from_files(str(tmp_path))
    assert r["git_repo"] == "acme/web"


def test_files_prefers_url_over_pushurl(tmp_path):
    git_dir = tmp_path / ".git"
    _write(
        git_dir / "config",
        '[remote "origin"]\n\tpushurl = git@github.com:acme/push-mirror.git\n\turl = git@github.com:acme/web.git\n',
    )
    r = git_context_from_files(str(tmp_path))
    assert r["git_repo"] == "acme/web"


def test_files_missing_git_dir(tmp_path):
    r = git_context_from_files(str(tmp_path))
    assert r == {"git_repo": None, "git_ref": None}


def test_client_warns_when_git_unresolved(monkeypatch, caplog):
    for v in (
        "TRACEROOT_GIT_REPO",
        "TRACEROOT_GIT_REF",
        "GITHUB_REPOSITORY",
        "GITHUB_SHA",
    ):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(
        client_mod, "auto_detect_git_context", lambda: {"git_repo": None, "git_ref": None}
    )
    monkeypatch.setattr(
        client_mod, "harvest_ci_git_context", lambda: {"git_repo": None, "git_ref": None}
    )
    monkeypatch.setattr(
        client_mod, "git_context_from_files", lambda: {"git_repo": None, "git_ref": None}
    )
    # Disable OTel initialization to avoid corrupting the shared global TracerProvider
    monkeypatch.setattr(client_mod.TracerootClient, "_initialize", lambda self: None)
    with caplog.at_level(logging.WARNING):
        client_mod.TracerootClient(api_key="test", git_repo=None, git_ref=None)
    assert any("git context incomplete" in r.message for r in caplog.records)


def test_client_treats_empty_git_env_as_unset(monkeypatch, caplog):
    # Empty TRACEROOT_GIT_* must behave like unset: fall through and warn,
    # never store "" as a resolved value.
    monkeypatch.setenv("TRACEROOT_GIT_REPO", "")
    monkeypatch.setenv("TRACEROOT_GIT_REF", "")
    for v in ("GITHUB_REPOSITORY", "GITHUB_SHA"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(
        client_mod, "auto_detect_git_context", lambda: {"git_repo": None, "git_ref": None}
    )
    monkeypatch.setattr(
        client_mod, "harvest_ci_git_context", lambda: {"git_repo": None, "git_ref": None}
    )
    monkeypatch.setattr(
        client_mod, "git_context_from_files", lambda: {"git_repo": None, "git_ref": None}
    )
    monkeypatch.setattr(client_mod.TracerootClient, "_initialize", lambda self: None)
    with caplog.at_level(logging.WARNING):
        c = client_mod.TracerootClient(api_key="test", git_repo=None, git_ref=None)
    assert c.git_repo is None and c.git_ref is None
    assert any("git context incomplete" in r.message for r in caplog.records)

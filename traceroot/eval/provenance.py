"""Run metadata + provenance collection for offline evaluation (Phase 4).

Merges caller-supplied run metadata with automatically discovered provenance -- git
(repository/ref/commit/dirty) and CI (provider/build_id) -- when it is available. Reuses
the SDK's existing git-context resolution (and the value it already resolved on the client,
so no extra git warning is printed) and degrades gracefully everywhere: detached HEAD,
dirty trees, no remote, packaged deployments, and CI all just yield partial-or-empty
provenance rather than failing the evaluation. No secrets or arbitrary env vars are
captured.

Shape:
    {
      ...user metadata (wins on key conflicts)...,
      "git": {"repository"?, "ref"?, "commit"?, "dirty"?},
      "ci":  {"provider", "build_id"?},
    }
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any

# (env flag that marks the provider, provider name, env var holding the build/run id)
_CI_PROVIDERS = (
    ("GITHUB_ACTIONS", "github", "GITHUB_RUN_ID"),
    ("GITLAB_CI", "gitlab", "CI_PIPELINE_ID"),
    ("CIRCLECI", "circleci", "CIRCLE_BUILD_NUM"),
    ("BUILDKITE", "buildkite", "BUILDKITE_BUILD_ID"),
    ("JENKINS_URL", "jenkins", "BUILD_NUMBER"),
)


def _resolved_git(env: dict[str, str]) -> tuple[str | None, str | None]:
    """(repository, commit) using the SAME precedence the client uses, preferring the
    value already resolved on the client so no extra git warning is emitted."""
    import traceroot
    from traceroot.env import TRACEROOT_GIT_REF, TRACEROOT_GIT_REPO
    from traceroot.git_context import (
        auto_detect_git_context,
        git_context_from_files,
        harvest_ci_git_context,
    )

    # Read an ALREADY-created client directly — never get_client(), which would lazily construct the
    # global client (extra git probes/warnings, and telemetry init under ambient credentials). This
    # helper must stay side-effect-free and cheap.
    client = getattr(traceroot, "_client", None)
    client_repo = getattr(client, "git_repo", None) if client is not None else None
    client_ref = getattr(client, "git_ref", None) if client is not None else None

    # An explicitly supplied env var wins over the client-resolved value: the client resolved ITS
    # value from os.environ at init, so when `env` IS os.environ the two agree, but when a caller
    # passes a custom `env` mapping to control provenance, that mapping must take effect.
    repo = env.get(TRACEROOT_GIT_REPO) or client_repo or None
    ref = env.get(TRACEROOT_GIT_REF) or client_ref or None
    if repo is None or ref is None:
        # These helpers never raise and never warn (the warning lives in client init).
        for src in (
            harvest_ci_git_context(env),
            git_context_from_files(),
            auto_detect_git_context(),
        ):
            repo = repo or src.get("git_repo")
            ref = ref or src.get("git_ref")
            if repo and ref:
                break
    return repo, ref


def _git_dirty() -> bool | None:
    """Best-effort working-tree cleanliness. None when it cannot be determined (no git
    binary, not a repo, packaged deploy). Bounded; never raises."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def _git_block(env: dict[str, str], *, detect_dirty: bool) -> dict[str, Any] | None:
    repo, ref = _resolved_git(env)
    block: dict[str, Any] = {}
    if repo:
        block["repository"] = repo
    if ref:
        # git_ref may be a branch/tag, not a commit SHA. Always expose it as `ref`, but only
        # populate `commit` when it actually looks like a SHA, so we never report a branch name
        # as a commit.
        block["ref"] = ref
        # Only EXACT OID lengths are commits (SHA-1 = 40, SHA-256 = 64). A shorter hex-looking
        # branch/tag (e.g. "deadbeef") must stay a ref, not be reported as a commit.
        if re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", ref):
            block["commit"] = ref
    if detect_dirty:
        dirty = _git_dirty()
        if dirty is not None:
            block["dirty"] = dirty
    return block or None


def _ci_block(env: dict[str, str]) -> dict[str, Any] | None:
    for flag, provider, build_var in _CI_PROVIDERS:
        if env.get(flag):
            block: dict[str, Any] = {"provider": provider}
            build_id = env.get(build_var)
            if build_id:
                block["build_id"] = build_id
            return block
    if env.get("CI"):  # generic CI with no recognized provider
        return {"provider": "ci"}
    return None


def collect_run_provenance(
    user_metadata: dict[str, Any] | None = None,
    *,
    env: dict[str, str] | None = None,
    detect_dirty: bool = True,
) -> dict[str, Any] | None:
    """Build run metadata = user metadata + auto git/ci provenance (when available).

    User-supplied keys win on conflict. Returns None when there is nothing to record.
    """
    env = env if env is not None else dict(os.environ)
    meta: dict[str, Any] = {}
    git = _git_block(env, detect_dirty=detect_dirty)
    if git:
        meta["git"] = git
    ci = _ci_block(env)
    if ci:
        meta["ci"] = ci
    if user_metadata:
        meta = {**meta, **user_metadata}  # user metadata takes precedence
    return meta or None

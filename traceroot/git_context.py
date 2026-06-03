"""Git context utilities for capturing source location and repo info."""

import inspect
import logging
import os
import subprocess
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Directory of this package — used to identify SDK-internal frames
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# Additional library paths to skip when walking the call stack
_SKIP_LIBRARIES = [
    "opentelemetry",
    "openinference",
]

# Cached git root path (None = not yet detected, "" = detection failed)
_git_root_cache: str | None = None


def _get_git_root() -> str | None:
    """Get the git repository root directory. Cached for performance."""
    global _git_root_cache
    if _git_root_cache is not None:
        return _git_root_cache if _git_root_cache else None

    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        _git_root_cache = git_root
        return git_root
    except Exception:
        _git_root_cache = ""  # Mark as failed
        return None


def capture_source_location() -> dict[str, str | int | None]:
    """Walk up the call stack to find the first frame outside SDK internals.

    Returns dict with git_source_file, git_source_line, git_source_function.
    """
    frame = inspect.currentframe()
    try:
        while frame:
            frame = frame.f_back
            if frame is None:
                break

            filename = frame.f_code.co_filename

            # Skip SDK internal frames (this package)
            if filename.startswith(_PACKAGE_DIR):
                continue

            # Skip known library frames
            if any(lib in filename for lib in _SKIP_LIBRARIES):
                continue

            # Skip frames from frozen/built-in modules
            if filename.startswith("<"):
                continue

            # Found user code
            return {
                "git_source_file": _relative_path(filename),
                "git_source_line": frame.f_lineno,
                "git_source_function": frame.f_code.co_name,
            }
    finally:
        del frame  # Avoid reference cycles

    return {}


def _relative_path(filepath: str) -> str:
    """Convert absolute path to relative (from git root, fallback to cwd)."""
    # Try git root first for correct GitHub links
    git_root = _get_git_root()
    if git_root and filepath.startswith(git_root):
        return filepath[len(git_root) :].lstrip(os.sep)

    # Fallback to cwd
    cwd = os.getcwd()
    if filepath.startswith(cwd):
        return filepath[len(cwd) :].lstrip(os.sep)
    return filepath


def auto_detect_git_context() -> dict[str, str | None]:
    """Auto-detect git_repo and git_ref from local git repo.

    Returns dict with git_repo and git_ref keys (values may be None).
    """
    result: dict[str, str | None] = {"git_repo": None, "git_ref": None}

    try:
        # Get remote URL
        remote = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()

        # Parse owner/repo from URL
        # Handles: https://github.com/o/r.git,
        # git@github.com:o/r.git, ssh://git@github.com/o/r.git
        import re

        match = re.match(
            r"(?:https?://|ssh://git@|git@)github\.com[:/](.+?)(?:\.git)?$",
            remote,
        )
        if match:
            result["git_repo"] = match.group(1).rstrip("/")
    except Exception:
        pass

    try:
        # Get current commit SHA
        ref = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        result["git_ref"] = ref
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# CI/platform environment variable resolution
# ---------------------------------------------------------------------------

# Each entry is (platform_name, repo_resolver, ref_resolver).
# repo_resolver: callable(env) -> str | None
# ref_resolver:  callable(env) -> str | None
#
# Platforms are probed in order; the first non-empty value for each field wins.

def _gh_repo(env: dict[str, str]) -> str | None:
    return env.get("GITHUB_REPOSITORY") or None


def _gh_ref(env: dict[str, str]) -> str | None:
    return env.get("GITHUB_SHA") or None


def _vercel_repo(env: dict[str, str]) -> str | None:
    owner = env.get("VERCEL_GIT_REPO_OWNER", "")
    slug = env.get("VERCEL_GIT_REPO_SLUG", "")
    if owner and slug:
        return f"{owner}/{slug}"
    return None


def _vercel_ref(env: dict[str, str]) -> str | None:
    return env.get("VERCEL_GIT_COMMIT_SHA") or None


def _gitlab_repo(env: dict[str, str]) -> str | None:
    return env.get("CI_PROJECT_PATH") or None


def _gitlab_ref(env: dict[str, str]) -> str | None:
    return env.get("CI_COMMIT_SHA") or None


def _circleci_repo(env: dict[str, str]) -> str | None:
    username = env.get("CIRCLE_PROJECT_USERNAME", "")
    reponame = env.get("CIRCLE_PROJECT_REPONAME", "")
    if username and reponame:
        return f"{username}/{reponame}"
    return None


def _circleci_ref(env: dict[str, str]) -> str | None:
    return env.get("CIRCLE_SHA1") or None


def _bitbucket_repo(env: dict[str, str]) -> str | None:
    return env.get("BITBUCKET_REPO_FULL_NAME") or None


def _bitbucket_ref(env: dict[str, str]) -> str | None:
    return env.get("BITBUCKET_COMMIT") or None


def _render_repo(env: dict[str, str]) -> str | None:  # Render does not expose a repo slug
    return None


def _render_ref(env: dict[str, str]) -> str | None:
    return env.get("RENDER_GIT_COMMIT") or None


_CI_PLATFORMS: list[
    tuple[
        str,
        Callable[[dict[str, str]], str | None],
        Callable[[dict[str, str]], str | None],
    ]
] = [
    ("GitHub Actions",       _gh_repo,        _gh_ref),
    ("Vercel",               _vercel_repo,    _vercel_ref),
    ("GitLab CI",            _gitlab_repo,    _gitlab_ref),
    ("CircleCI",             _circleci_repo,  _circleci_ref),
    ("Bitbucket Pipelines",  _bitbucket_repo, _bitbucket_ref),
    ("Render",               _render_repo,    _render_ref),
]


def harvest_ci_git_context(
    env: dict[str, str] | None = None,
) -> dict[str, str | None]:
    """Harvest git_repo and git_ref from CI/platform environment variables.

    Checks platforms in order (first-platform-wins per field)::

        GitHub Actions → Vercel → GitLab CI → CircleCI
        → Bitbucket Pipelines → Render

    ``git_repo`` and ``git_ref`` are resolved **independently**.  If platform A
    provides only a ref and platform B provides only a repo, both values are
    used regardless of the platform order.

    Args:
        env: Environment mapping to probe.  Defaults to ``os.environ``.
             Pass a plain ``dict`` in tests to avoid touching the real
             process environment.

    Returns:
        ``dict`` with ``"git_repo"`` and ``"git_ref"`` keys.  Either value
        may be ``None`` when it cannot be resolved.  Never raises.
    """
    if env is None:
        env = dict(os.environ)

    result: dict[str, str | None] = {"git_repo": None, "git_ref": None}

    try:
        for _platform, repo_fn, ref_fn in _CI_PLATFORMS:
            if result["git_repo"] is None:
                try:
                    result["git_repo"] = repo_fn(env)
                except Exception:
                    pass

            if result["git_ref"] is None:
                try:
                    result["git_ref"] = ref_fn(env)
                except Exception:
                    pass

            # Short-circuit once both fields are satisfied
            if result["git_repo"] and result["git_ref"]:
                break
    except Exception:
        pass

    return result

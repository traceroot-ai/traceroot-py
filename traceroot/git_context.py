"""Git context utilities for capturing source location and repo info."""

import inspect
import logging
import os
import re
import subprocess
from collections.abc import Mapping

from traceroot.env import GITHUB_REPOSITORY, GITHUB_SHA

logger = logging.getLogger(__name__)

# Directory of this package — used to identify SDK-internal frames
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# Regex to parse owner/repo from GitHub remote URLs.
# Handles: https://github.com/o/r.git, git@github.com:o/r.git,
# ssh://git@github.com/o/r.git
_REPO_URL_RE = re.compile(r"(?:https?://|ssh://git@|git@)github\.com[:/](.+?)(?:\.git)?$")

# Regex to validate a 40-char hex SHA
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

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


def git_context_from_files(cwd: str | None = None) -> dict[str, str | None]:
    """Read git context directly from ``.git`` files (no ``git`` subprocess).

    Works in containers that ship a ``.git`` dir but no ``git`` binary.
    Only inspects ``<cwd>/.git`` (does not walk parent directories).
    """
    base = cwd if cwd is not None else os.getcwd()
    git_dir = os.path.join(base, ".git")
    result: dict[str, str | None] = {"git_repo": None, "git_ref": None}

    try:
        with open(os.path.join(git_dir, "config"), encoding="utf-8") as f:
            config = f.read()
        seen_origin = False
        for line in config.splitlines():
            if re.match(r'^\[remote "origin"\]', line):
                seen_origin = True
                continue
            if seen_origin and line.startswith("["):
                break
            if seen_origin:
                m = re.search(r"\burl\s*=\s*(.+)$", line)
                if m:
                    rm = _REPO_URL_RE.match(m.group(1).strip())
                    if rm:
                        result["git_repo"] = rm.group(1).rstrip("/")
                    break
    except OSError:
        pass

    try:
        with open(os.path.join(git_dir, "HEAD"), encoding="utf-8") as f:
            head = f.read().strip()
        if _SHA_RE.match(head):
            result["git_ref"] = head
        else:
            rm = re.match(r"ref:\s+(\S+)", head)
            if rm:
                ref_path = rm.group(1)  # e.g. refs/heads/main
                try:
                    with open(os.path.join(git_dir, ref_path), encoding="utf-8") as f:
                        loose = f.read().strip()
                    if _SHA_RE.match(loose):
                        result["git_ref"] = loose
                except OSError:
                    pass  # loose ref missing — the ref may be packed
                if result["git_ref"] is None:
                    try:
                        # Packed refs (after `git gc` / fresh clone).
                        with open(os.path.join(git_dir, "packed-refs"), encoding="utf-8") as f:
                            packed = f.read()
                        for line in packed.splitlines():
                            if not line or line[0] in "#^":
                                continue
                            # OIDs are exactly 40 (SHA-1) or 64 (SHA-256) hex.
                            m = re.match(r"^([0-9a-f]{40}|[0-9a-f]{64})\s+(.+)$", line)
                            if m and m.group(2) == ref_path:
                                result["git_ref"] = m.group(1)
                                break
                    except OSError:
                        pass
    except OSError:
        pass

    return result


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

        # Parse owner/repo from URL using shared regex
        match = _REPO_URL_RE.match(remote)
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


def harvest_ci_git_context(
    env: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    """Resolve git context from GitHub Actions environment variables —
    ``GITHUB_REPOSITORY`` (owner/repo) and ``GITHUB_SHA``.

    repo and ref resolve independently (a build may provide one, not both).
    """
    e = os.environ if env is None else env
    return {
        "git_repo": e.get(GITHUB_REPOSITORY) or None,
        "git_ref": e.get(GITHUB_SHA) or None,
    }

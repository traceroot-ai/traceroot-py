"""Client-generated stable identifiers for offline evaluation (DS-1).

ULID: 48-bit millisecond timestamp + 80 bits of randomness, Crockford base32
(26 chars, time-sortable). Stdlib only. Datasets/cases/runs get typed prefixes
(``ds_``/``tc_``/``run_``) so the SDK can start local work without a server
round-trip; the server accepts these ids idempotently on push.
"""

from __future__ import annotations

import hashlib
import os
import time

# Crockford base32 (excludes I, L, O, U).
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _encode(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def ulid() -> str:
    """A 26-char Crockford base32 ULID (time-ordered, unique)."""
    ts = int(time.time() * 1000) & ((1 << 48) - 1)
    rand = int.from_bytes(os.urandom(10), "big")  # 80 random bits
    return _encode(ts, 10) + _encode(rand, 16)


def new_id(prefix: str) -> str:
    return f"{prefix}_{ulid()}"


def new_dataset_id() -> str:
    return new_id("ds")


def stable_dataset_id(key: str) -> str:
    """Deterministic dataset id derived from a semantic key (``ds_``+sha256 prefix).

    A dataset's identity is its key, NOT which SDK or process created it, so this is a
    pure function of ``key``: the SAME key -- in any process, in Python or TypeScript --
    yields the SAME ``client_dataset_id``, which is how the platform converges runs of one
    logical dataset (upsert on ``(project, client_dataset_id)``) instead of forking a new
    dataset each run. Must stay byte-for-byte identical to the TypeScript ``stableDatasetId``.
    """
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"ds_{digest[:26]}"


def stable_case_id(dataset_key: str, index: int) -> str:
    """Deterministic case id from the dataset key + insertion position.

    Convergence needs case ids to be stable across runs, not random per construction:
    the same case authored in the same position -- any process, Python or TypeScript --
    must get the SAME ``tc_`` id so the platform matches it on re-publish (upsert keys on
    id) instead of duplicating it, and so runs pair case-for-case. Position-based (not
    content-based) to stay trivially identical across languages. Must match the TypeScript
    ``stableCaseId``.
    """
    digest = hashlib.sha256(f"{dataset_key}\x00{index}".encode("utf-8")).hexdigest()
    return f"tc_{digest[:20]}"


def new_test_case_id() -> str:
    return new_id("tc")


def new_run_id() -> str:
    return new_id("run")

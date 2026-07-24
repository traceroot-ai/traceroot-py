"""Client-generated stable identifiers for offline evaluation (DS-1).

ULID: 48-bit millisecond timestamp + 80 bits of randomness, Crockford base32
(26 chars, time-sortable). Stdlib only. Datasets/cases/runs get typed prefixes
(``ds_``/``tc_``/``run_``) so the SDK can start local work without a server
round-trip; the server accepts these ids idempotently on push.
"""

from __future__ import annotations

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


def new_test_case_id() -> str:
    return new_id("tc")


def new_run_id() -> str:
    return new_id("run")

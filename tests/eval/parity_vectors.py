"""Decoder for the shared cross-SDK parity fixtures.

``case_id_parity.json`` is byte-identical in traceroot-py and traceroot-ts, so its vectors have
to be expressible in plain JSON -- but the values that actually diverged between the SDKs (dates,
NaN/Infinity, sets, bytes, maps) have no JSON literal. The fixture encodes those in a tagged form
that each suite decodes to its own native value before feeding the SDK. The tags exist ONLY in the
fixture; the canonicalizer never sees them.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "fixtures"

CASE_FIXTURE: dict[str, Any] = json.loads(
    (FIXTURES / "case_id_parity.json").read_text(encoding="utf-8")
)
SCORER_FIXTURE: dict[str, Any] = json.loads(
    (FIXTURES / "scorer_definition_parity.json").read_text(encoding="utf-8")
)


def decode(value: Any) -> Any:
    """Fixture encoding -> native Python value. Mirrors ``decode`` in tests/parity_vectors.ts."""
    if isinstance(value, list):
        return [decode(v) for v in value]
    if isinstance(value, dict):
        if "$nan" in value:
            return float("nan")
        if "$inf" in value:
            return float("inf") if value["$inf"] > 0 else float("-inf")
        if "$date" in value:
            return datetime.fromisoformat(value["$date"].replace("Z", "+00:00")).astimezone(UTC)
        if "$set" in value:
            return {decode(v) for v in value["$set"]}
        if "$bytes" in value:
            return bytes(value["$bytes"])
        if "$map" in value:
            return {decode(k): decode(v) for k, v in value["$map"]}
        return {k: decode(v) for k, v in value.items()}
    return value

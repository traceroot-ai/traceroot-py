"""ONE canonical form for offline evaluation — used for BOTH hashing and the wire.

Byte-identical with the TypeScript ``src/eval/canonical.ts``. Every dataset/case/scorer
identity in the SDK flows through here, so a dataset authored in Python and re-authored in
TypeScript reconciles to one identity on the platform.

The grammar is deliberately **JavaScript's**, because JSON's number space is JavaScript's and
only one of the two languages can be the reference:

* numbers render via the ECMAScript ``Number::toString`` algorithm (ECMA-262 6.1.6.1.20), so
  ``1.0`` is ``1``, ``-0.0`` is ``0``, ``1e-5`` is ``0.00001`` and ``1e-7`` is ``1e-7``;
* object keys sort by Unicode CODE POINT (Python's native string order), and the JSON text is
  built directly from that sorted key list — never by assigning into a dict/object, which in
  JavaScript would silently reorder integer-like keys;
* ``datetime``/``date`` render exactly like JavaScript ``Date.prototype.toISOString()``;
* ``set``/``frozenset`` render as an array sorted by each item's canonical JSON;
* ``bytes`` render as an array of byte integers;
* ``NaN``/``Infinity``/``-Infinity`` render as the string sentinels ``"NaN"``/``"Infinity"``/
  ``"-Infinity"`` (JSON has no such literals and both SDKs must agree on the substitute);
* anything else (class instances, functions, pydantic models, ``Decimal``, ``UUID``, ...) is
  REJECTED with an actionable error rather than hashed in a language-specific shape.

:func:`normalize` produces the value that is hashed AND the value put on the wire, so
hash-form == wire-form == read-back-form and a pushed dataset's revision survives a round trip.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, date, datetime
from typing import Any, NoReturn

# Integers up to 2**53 are exactly representable as an IEEE-754 double, i.e. survive any JSON
# pipeline that parses numbers as JS Numbers. Beyond that a Python int has no faithful JS
# counterpart, so it is rendered as the double it will become — which is also what comes back
# on a read, keeping the round trip stable.
_MAX_EXACT_INT = 2**53


class CanonicalizationError(TypeError):
    """A value cannot be canonicalized (not JSON-serializable, or a reference cycle)."""


# --- ECMAScript Number::toString ------------------------------------------------------------


def _shortest_decimal(x: float) -> tuple[str, int]:
    """Split a positive finite float into ``(digits, n)`` with ``digits`` the shortest
    round-tripping decimal digit string and ``x == 0.<digits> * 10**n``.

    ``repr`` is Python's shortest round-trip representation (the same value V8 computes), so
    only the *formatting* has to be redone below — never the digits.
    """
    text = repr(x)
    mantissa, _, exponent = text.partition("e")
    exp = int(exponent) if exponent else 0
    int_part, _, frac_part = mantissa.partition(".")
    digits = int_part + frac_part
    point = len(int_part) + exp
    stripped = digits.lstrip("0")
    point -= len(digits) - len(stripped)
    return stripped.rstrip("0"), point


def es_number(x: float) -> str:
    """Render a finite float exactly as ECMAScript ``Number::toString`` (i.e. JS ``String(x)``)."""
    if x == 0:  # covers -0.0: ECMAScript renders it "0"
        return "0"
    sign = "-" if x < 0 else ""
    s, n = _shortest_decimal(abs(x))
    k = len(s)
    if k <= n <= 21:
        return sign + s + "0" * (n - k)
    if 0 < n <= 21:
        return sign + s[:n] + "." + s[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + s
    e = n - 1
    exponent = ("e+" if e > 0 else "e-") + str(abs(e))
    return sign + (s if k == 1 else s[0] + "." + s[1:]) + exponent


def _int_token(v: int) -> str:
    if -_MAX_EXACT_INT <= v <= _MAX_EXACT_INT:
        return str(v)
    try:
        return es_number(float(v))
    except OverflowError as e:
        raise CanonicalizationError(
            f"integer {v} is too large to canonicalize (no IEEE-754 double representation)"
        ) from e


# --- normalization --------------------------------------------------------------------------


def _iso(v: date) -> str:
    """ISO-8601 exactly as JavaScript ``Date.prototype.toISOString()``: always UTC, always
    millisecond precision, trailing ``Z``. A naive datetime is read as UTC (a JS Date is an
    instant; there is no naive form to mirror)."""
    if isinstance(v, datetime):
        dt = v if v.tzinfo is not None else v.replace(tzinfo=UTC)
        dt = dt.astimezone(UTC)
    else:
        dt = datetime(v.year, v.month, v.day, tzinfo=UTC)
    return (
        f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"
        f"T{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}.{dt.microsecond // 1000:03d}Z"
    )


def _key(k: Any) -> str:
    """Stringify a mapping key the way a JavaScript object/Map key stringifies."""
    if isinstance(k, str):
        _reject_lone_surrogate(k)
        return k
    if isinstance(k, bool):
        return "true" if k else "false"
    if isinstance(k, int):
        return _int_token(k)
    if isinstance(k, float):
        if math.isnan(k):
            return "NaN"
        if math.isinf(k):
            return "Infinity" if k > 0 else "-Infinity"
        return es_number(k)
    if k is None:
        return "null"
    raise CanonicalizationError(
        f"mapping key of type {type(k).__name__!r} is not JSON-serializable; "
        "use a string, number, bool, or None key"
    )


def _reject_lone_surrogate(s: str) -> None:
    """A lone UTF-16 surrogate (U+D800..U+DFFF) is not valid Unicode text and cannot be encoded
    as UTF-8: hashing it here would raise a raw ``UnicodeEncodeError`` while the TypeScript SDK
    would silently hash the escaped form, so the two SDKs would disagree. Reject it explicitly in
    both, as a ``CanonicalizationError``, before it reaches the hash."""
    if any(0xD800 <= ord(c) <= 0xDFFF for c in s):
        raise CanonicalizationError(
            "string contains an unpaired UTF-16 surrogate and cannot be canonicalized; "
            "it is not valid Unicode text"
        )


def _reject(value: Any) -> NoReturn:
    raise CanonicalizationError(
        f"value of type {type(value).__name__!r} is not JSON-serializable; evaluation payloads "
        "must be built from None/bool/int/float/str/list/tuple/dict/set/bytes/datetime "
        "(convert it yourself, e.g. dataclasses.asdict(x) or x.model_dump())"
    )


def _norm(value: Any, path: list[int]) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        _reject_lone_surrogate(value)
        return value
    if isinstance(value, int):
        if -_MAX_EXACT_INT <= value <= _MAX_EXACT_INT:
            return value
        # Beyond 2**53 the number IS a double (see _int_token / JS). Normalize to it here too, or
        # the WIRE would carry the exact int while the HASH was taken over the rounded double —
        # pushed content that doesn't match its own revision.
        try:
            return float(value)
        except OverflowError as e:
            raise CanonicalizationError(
                f"integer {value} is too large to canonicalize (no IEEE-754 double representation)"
            ) from e
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, (datetime, date)):
        return _iso(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return list(bytes(value))

    if isinstance(value, (dict, list, tuple, set, frozenset)):
        # Track only the ACTIVE recursion path: a true cycle is an error, but a merely repeated
        # (aliased) reference must still canonicalize as its full content.
        if id(value) in path:
            raise CanonicalizationError(
                "value contains a reference cycle and cannot be canonicalized"
            )
        path.append(id(value))
        try:
            if isinstance(value, dict):
                out: dict[str, Any] = {}
                for k, v in value.items():
                    ks = _key(k)
                    if ks in out:
                        raise CanonicalizationError(
                            f"mapping has two keys that stringify to {ks!r}; "
                            "canonical JSON cannot represent both"
                        )
                    out[ks] = _norm(v, path)
                return out
            if isinstance(value, (list, tuple)):
                return [_norm(v, path) for v in value]
            # set/frozenset iteration order is not stable across processes; emit a deterministic
            # array ordered by each item's canonical JSON.
            return sorted((_norm(v, path) for v in value), key=_dump)
        finally:
            path.pop()

    _reject(value)


def normalize(value: Any) -> Any:
    """Canonical plain-JSON form of ``value`` — what gets hashed AND what goes on the wire."""
    return _norm(value, [])


# --- serialization --------------------------------------------------------------------------


def _dump(value: Any) -> str:
    """Serialize an ALREADY-normalized value to canonical JSON text."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return _int_token(value)
    if isinstance(value, float):
        return es_number(value)
    if isinstance(value, list):
        return "[" + ",".join(_dump(v) for v in value) + "]"
    # dict — keys are already strings; sorted() is Unicode code-point order, and the text is
    # built straight from that sorted list so nothing can reorder it.
    return (
        "{"
        + ",".join(json.dumps(k, ensure_ascii=False) + ":" + _dump(value[k]) for k in sorted(value))
        + "}"
    )


def canonical_json(value: Any) -> str:
    """Canonical JSON text for ``value`` (sorted keys, no whitespace, UTF-8)."""
    return _dump(normalize(value))


def canonical_hash(value: Any, length: int) -> str:
    """sha256 of the canonical JSON of ``value``, truncated to ``length`` hex chars."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]

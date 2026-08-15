"""Phase 3 — stable cross-language identity (Python side of the shared fixture).

Proves that a scorer given the same explicit `key` in Python and TypeScript resolves to the SAME
logical scorer identity on the wire, while its provenance (function-name spelling, SDK language,
captured source) is retained and DIFFERS across languages. The TS counterpart
(`eval-cross-sdk-identity.test.ts`) asserts the same `key` from the same fixture.
"""

import json
from pathlib import Path

from traceroot.eval.scorers import scorer, scorer_metadata

FIX = json.loads((Path(__file__).parent / "fixtures" / "cross_sdk_identity.json").read_text())


def covers_both_cities(input, output, expected=None):  # snake_case — a Python spelling
    return 1.0 if output == expected else 0.0


def test_scorer_key_is_stable_across_languages():
    s = scorer(covers_both_cities, key=FIX["scorer_key"], threshold=1.0)
    md = scorer_metadata(s)
    # Stable SEMANTIC identity — identical to the value the TS fixture asserts.
    assert md["key"] == FIX["scorer_key"]
    # Provenance differs from TS and is NOT the identity:
    assert md["name"] == FIX["python"]["fn_name"]        # function-name spelling (snake_case)
    assert md["language"] == FIX["python"]["language"]   # SDK language = python
    assert md["source"] and "def covers_both_cities" in md["source"]


def test_key_defaults_to_definition_name_when_unset():
    def fresh_scorer(input, output):  # a distinct fn (scorer() mutates meta in place)
        return 1.0

    md = scorer_metadata(scorer(fresh_scorer, threshold=1.0))
    assert md["key"] == "fresh_scorer"  # no explicit key -> the definition name (diverges by language)

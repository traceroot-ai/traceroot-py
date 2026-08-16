"""First-class scorer metadata + definition reporting.

A scorer is any callable ``(ScorerContext) -> value | Score``. The optional ``@scorer``
decorator lets an evaluation DESCRIBE a scorer so the platform can (a) compare candidates
correctly (value type, direction, threshold) and (b) render the read-only Scorer detail
that shows *what the scorer is* -- its source (code scorers) or model+messages (LLM judges),
plus output type, threshold, description, and metadata. Plain callables keep working.

    @scorer(value_type="numeric", direction="higher_is_better", threshold=0.9)
    def accuracy(ctx): ...

    @scorer(output_type="score", threshold=1.0, description="Exact match")
    def exact_match(ctx): ...

    concise = llm_judge(name="concise", model="claude-sonnet-5", messages=[...],
                        output_type="score", threshold=0.8)

Defaults: numeric/boolean scorers default to ``higher_is_better``; categorical scorers
default to ``none``. ``output_type`` derives from ``value_type`` (categorical -> classification,
else score) when not given; an explicit value wins. Direction is NEVER inferred from the
scorer's name; a version is only reported when declared. Every definition field is optional
and additive -- an absent field is omitted (never fabricated), so the detail page shows
"Not provided by SDK".
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from collections.abc import Callable, Sequence
from typing import Any

VALUE_TYPES = ("numeric", "boolean", "categorical")
DIRECTIONS = ("higher_is_better", "lower_is_better", "none")
OUTPUT_TYPES = ("score", "classification")
SCORER_TYPES = ("code", "llm_judge")
# The ScorerContext fields a scorer may declare it needs. An extensible descriptor
# (not a narrow reference_based boolean): output-only scorers declare ["output"], a
# reference scorer adds "expected", etc. Absent = unknown (never assumed).
REQUIRED_INPUTS = ("input", "output", "expected", "metadata", "trace")

_META_ATTR = "_traceroot_scorer"


def _validate_required_inputs(value: Any) -> list[str]:
    """Coerce a declared ``required_inputs`` to a validated, canonically-ordered list."""
    if not isinstance(value, (list, tuple)) or not all(isinstance(x, str) for x in value):
        raise ValueError("required_inputs must be a list of strings")
    unknown = [x for x in value if x not in REQUIRED_INPUTS]
    if unknown:
        raise ValueError(f"required_inputs must be a subset of {REQUIRED_INPUTS}, got {unknown!r}")
    present = set(value)
    return [x for x in REQUIRED_INPUTS if x in present]


def scorer(
    fn: Callable | None = None,
    *,
    key: str | None = None,
    name: str | None = None,
    version: str | None = None,
    value_type: str | None = None,
    direction: str | None = None,
    threshold: float | None = None,
    output_type: str | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
    required_inputs: list[str] | None = None,
) -> Callable:
    """Decorator that attaches metadata to a (code) scorer callable.

    Usable bare (``@scorer``) or with arguments. The scorer's SOURCE is captured
    automatically at report time (``inspect.getsource``); ``@scorer`` only adds the
    declared metadata (output type, threshold, description, required inputs, ...).

    ``required_inputs`` declares which ``ScorerContext`` fields the scorer consumes
    (a subset of ``REQUIRED_INPUTS``); an output-only scorer declares ``["output"]``.
    Left unset, a code scorer's requirements are unknown and the field is omitted.
    """
    if value_type is not None and value_type not in VALUE_TYPES:
        raise ValueError(f"value_type must be one of {VALUE_TYPES}, got {value_type!r}")
    if direction is not None and direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if output_type is not None and output_type not in OUTPUT_TYPES:
        raise ValueError(f"output_type must be one of {OUTPUT_TYPES}, got {output_type!r}")
    if required_inputs is not None:
        required_inputs = _validate_required_inputs(required_inputs)

    def apply(f: Callable) -> Callable:
        meta = dict(getattr(f, _META_ATTR, {}))
        for mk, val in (
            ("key", key),
            ("name", name),
            ("version", version),
            ("value_type", value_type),
            ("direction", direction),
            ("threshold", threshold),
            ("output_type", output_type),
            ("description", description),
            ("metadata", metadata),
            ("required_inputs", required_inputs),
        ):
            if val is not None:
                meta[mk] = val
        setattr(f, _META_ATTR, meta)
        return f

    return apply(fn) if fn is not None else apply


def _declared(fn: Callable, key: str) -> Any:
    meta = getattr(fn, _META_ATTR, None)
    if isinstance(meta, dict) and meta.get(key) is not None:
        return meta[key]
    # Back-compat: a plainly-set attribute (e.g. fn.version = "2") also counts as declared.
    return getattr(fn, key, None)


def scorer_name(fn: Callable, fallback: str = "scorer") -> str:
    """The ONE place a scorer's reported name is resolved: declared name -> ``__name__`` ->
    the callable's class name -> ``fallback``.

    Every site that names a scorer (the emitted Score, the scorer->metric ownership map, the span,
    the registration/completion manifest) must route through this. When they disagree -- e.g.
    ``@scorer(name="quality")`` over ``def grade`` -- the platform cannot attribute the metric to
    its definition and drops both ``emitted_metrics`` and the metric's pass/fail policy.
    """
    return _declared(fn, "name") or getattr(fn, "__name__", None) or type(fn).__name__ or fallback


def declared_version(fn: Callable) -> str | None:
    """The scorer's explicitly declared version, or None (never fabricated)."""
    return _declared(fn, "version")


def _capture_source(fn: Callable) -> str | None:
    """The scorer's source, verbatim, with any leading decorator lines stripped. None when
    the source cannot be introspected (C callables, REPL-defined, etc.)."""
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return None
    lines = src.splitlines()
    # Drop the decorator block so the definition shows just the function. A decorator can span
    # many physical lines (e.g. a multi-line @scorer(...)), so slice from the def/async-def
    # header the AST reports rather than only dropping lines that start with "@".
    try:
        node = ast.parse(src).body[0]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.decorator_list:
            lines = lines[node.lineno - 1 :]
    except (SyntaxError, IndexError, ValueError):
        while lines and lines[0].lstrip().startswith("@"):  # best-effort fallback
            lines.pop(0)
    return "\n".join(lines).strip() or None


def scorer_metadata(fn: Callable, *, value_type: str | None = None) -> dict[str, Any]:
    """Build a scorer descriptor: identity + comparison metadata + the read-only definition.

    Shared: name, version, scorer_type, value_type, direction, threshold, output_type,
    description, metadata. Code scorers add language + source; LLM judges add model +
    messages. ``value_type`` is a runtime hint used only when the scorer did not declare
    one. Absent fields are ``None`` here and omitted at the reporting boundary.
    """
    name = scorer_name(fn)
    # Stable SEMANTIC identity, independent of function spelling/language. Defaults to the definition
    # name; set an explicit `key` (identically in Python and TypeScript) to make the SAME logical
    # scorer resolve across languages. Never derived from source; code/language/version are provenance.
    key = _declared(fn, "key") or name
    vtype = _declared(fn, "value_type") or value_type
    direction = _declared(fn, "direction")
    if direction is None and vtype is not None:
        direction = "none" if vtype == "categorical" else "higher_is_better"
    scorer_type = _declared(fn, "scorer_type") or "code"
    output_type = _declared(fn, "output_type")
    if output_type is None and vtype is not None:
        output_type = "classification" if vtype == "categorical" else "score"

    # Declared requirements win; an llm_judge otherwise derives them from its template
    # placeholders. A bare/undeclared code scorer stays unknown (None -> omitted): we never
    # claim ``expected`` is required just because it exists on the context.
    required_inputs = _declared(fn, "required_inputs")
    if required_inputs is None and scorer_type == "llm_judge":
        required_inputs = _derive_required_inputs(_declared(fn, "messages"))

    desc: dict[str, Any] = {
        "key": key,
        "name": name,
        "version": declared_version(fn),
        "scorer_type": scorer_type,
        "value_type": vtype,
        "direction": direction,
        "threshold": _declared(fn, "threshold"),
        "output_type": output_type,
        "description": _declared(fn, "description"),
        "metadata": _declared(fn, "metadata"),
        "required_inputs": required_inputs,
    }
    if scorer_type == "llm_judge":
        desc["model"] = _declared(fn, "model")
        desc["messages"] = _declared(fn, "messages")
        # A DYNAMIC judge also carries its builder-callback provenance (captured at construction);
        # a static judge has none (its declarative config IS its source, via version=config-hash).
        builder_source = _declared(fn, "builder_source")
        if builder_source is not None:
            desc["language"] = "python"
            desc["source"] = builder_source
    else:
        src = _capture_source(fn)
        if src is not None:
            desc["language"] = "python"
            desc["source"] = src
    return desc


def describe_scorers(
    scorers: Sequence[Callable], *, value_types: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """Descriptor list for a set of scorers. ``value_types`` maps scorer name -> a runtime
    value-type hint for scorers that did not declare one."""
    hints = value_types or {}
    out: list[dict[str, Any]] = []
    for s in scorers:
        # Key the hint lookup by the DECLARED name (what the rendered descriptor uses), not the raw
        # implementation function name -- otherwise a @scorer(name=...) never receives its hint.
        # Honor the omitted-fields contract for direct consumers too: drop keys the scorer did not
        # declare (None) instead of exposing fabricated null schema fields.
        desc = {
            k: v
            for k, v in scorer_metadata(s, value_type=hints.get(scorer_name(s))).items()
            if v is not None
        }
        out.append(desc)
    return out


# --- LLM-judge scorer -----------------------------------------------------------------

_PLACEHOLDER = re.compile(r"\{\{\s*(input|output|expected)\s*\}\}")


def _derive_required_inputs(messages: Any) -> list[str] | None:
    """The ScorerContext fields an llm_judge template actually references, derived from its
    ``{{input}}``/``{{output}}``/``{{expected}}`` placeholders (canonical order). None when
    no messages are available; ``[]`` when the prompt references no case fields."""
    if not messages:
        return None
    found: set[str] = set()
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        found.update(m.group(1) for m in _PLACEHOLDER.finditer(content or ""))
    return [x for x in REQUIRED_INPUTS if x in found]


# The llm_judge scorer lives in its own module; re-exported here so existing imports
# (``from traceroot.eval.scorers import llm_judge``) keep working unchanged.
from traceroot.eval.judge import _parse_judge_output, llm_judge  # noqa: E402,F401


class Scorer:
    """Unified scorer-definition namespace with NAMED constructors (stronger types + autocomplete
    than one ``Scorer(type=...)`` constructor whose parameters mostly wouldn't apply):

      - ``Scorer.code(...)``       -- a code scorer. Bare/decorator (``@Scorer.code(...)``) or an
        adapter over an existing callable (``Scorer.code(fn, threshold=1.0)``). An ordinary function
        also works with NO wrapper at all (passed straight to ``evaluate(scorers=[...])``).
      - ``Scorer.llm_judge(...)``  -- an LLM judge. Static (declarative config only -- no function
        needed) or dynamic (``@Scorer.llm_judge(...)`` over a builder that returns the judge's
        template variables / messages per case).

    ``kind`` (``code`` | ``llm_judge``) stays INTERNAL on the normalized definition; there is no
    public ``Scorer(type=...)`` constructor. ``Scorer`` is the executable definition + policy +
    provenance; ``Score`` (separate) is one emitted result for one case."""

    code = staticmethod(scorer)
    llm_judge = staticmethod(llm_judge)

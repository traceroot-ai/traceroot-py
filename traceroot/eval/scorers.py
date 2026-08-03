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
        for key, val in (
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
                meta[key] = val
        setattr(f, _META_ATTR, meta)
        return f

    return apply(fn) if fn is not None else apply


def _declared(fn: Callable, key: str) -> Any:
    meta = getattr(fn, _META_ATTR, None)
    if isinstance(meta, dict) and meta.get(key) is not None:
        return meta[key]
    # Back-compat: a plainly-set attribute (e.g. fn.version = "2") also counts as declared.
    return getattr(fn, key, None)


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
    name = _declared(fn, "name") or getattr(fn, "__name__", None) or fn.__class__.__name__
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
        base_name = getattr(s, "__name__", None) or s.__class__.__name__
        out.append(scorer_metadata(s, value_type=hints.get(base_name)))
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


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    import json

    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _render_messages(messages: list[dict[str, str]], ctx: Any) -> list[dict[str, str]]:
    """Substitute {{input}}/{{output}}/{{expected}} into each message's content, per case.
    The AUTHORED template (with placeholders) is what gets reported; this rendering is for
    execution only."""
    values = {
        "input": _as_text(getattr(ctx, "input", None)),
        "output": _as_text(getattr(ctx, "output", None)),
        "expected": _as_text(getattr(ctx, "expected", None)),
    }
    rendered = []
    for m in messages:
        content = _PLACEHOLDER.sub(lambda mo: values[mo.group(1)], m.get("content", ""))
        rendered.append({"role": m.get("role", "user"), "content": content})
    return rendered


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_judge_output(text: str, output_type: str) -> float | str:
    """Parse a judge's response into a score.

    The judge contract is "reply with a single number and nothing else", so the response
    itself is the number. To avoid the "first number wins" footgun (``Step 3: the score is
    0.8`` must NOT become ``3``), we accept an exact numeric response, or a single unambiguous
    number in prose, and otherwise raise -- a malformed/ambiguous response is an isolated
    scorer error with the raw text preserved for diagnosis, never a wrong silent score.
    """
    if output_type == "classification":
        return (text or "").strip()
    stripped = (text or "").strip()
    candidate = stripped.rstrip(".")  # tolerate a trailing period on an exact answer
    if _NUMBER.fullmatch(candidate):
        return float(candidate)
    numbers = _NUMBER.findall(stripped)
    if len(numbers) == 1:
        return float(numbers[0])
    raise ValueError(
        f"llm_judge: expected a single numeric score, found {len(numbers)} in model "
        f"output: {stripped[:200]!r}"
    )


def _provider_integration_traces(model: str) -> bool:
    """True when an active traceroot integration already traces this model's provider calls.

    In that case the judge must NOT add its own LLM span (the integration emits one for the
    underlying request, and an LLM span nested inside an LLM span is redundant). The provider is
    inferred from the model id the same way ``_default_complete`` dispatches: an anthropic model
    checks the anthropic integration, otherwise openai.
    """
    try:
        import traceroot
        from traceroot.instrumentation import Integration

        # Read the already-initialized client directly (do NOT call get_client(), which would
        # auto-create one as a side effect). None -> not initialized -> no integration active.
        active = set(getattr(getattr(traceroot, "_client", None), "_instrumented", None) or [])
        if not active:
            return False
        m = (model or "").lower()
        if m.startswith("claude") or m.startswith("anthropic"):
            return Integration.ANTHROPIC in active
        return Integration.OPENAI in active
    except Exception:
        return False


def _default_complete(model: str, messages: list[dict[str, str]]) -> str:
    """Best-effort provider dispatch used when no `complete` is injected. Lazily imports the
    provider so the SDK never hard-depends on it; raises a clear error when unavailable."""
    if model.startswith(("claude", "anthropic")):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - execution path, not unit-tested
            raise RuntimeError(
                "llm_judge needs the 'anthropic' package to call this model, or pass complete=..."
            ) from e
        system = "\n".join(m["content"] for m in messages if m["role"] == "system") or None
        turns = [
            {"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"
        ]
        resp = anthropic.Anthropic().messages.create(
            model=model, max_tokens=512, system=system, messages=turns
        )
        return "".join(getattr(b, "text", "") for b in resp.content)
    try:
        import openai
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "llm_judge needs the 'openai' package to call this model, or pass complete=..."
        ) from e
    resp = openai.OpenAI().chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content or ""


def llm_judge(
    *,
    name: str,
    model: str,
    messages: list[dict[str, str]],
    version: str | None = None,
    output_type: str = "score",
    threshold: float | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
    direction: str | None = None,
    value_type: str | None = None,
    required_inputs: list[str] | None = None,
    complete: Callable[[str, list[dict[str, str]]], str] | None = None,
) -> Callable:
    """A first-class LLM-judge scorer: its ``model`` + ``messages`` (the judge prompt) are
    carried as the reported definition, and calling it runs the judge over a case.

    ``messages`` are the AUTHORED template (``{{output}}``/``{{input}}``/``{{expected}}``
    placeholders sent verbatim in the manifest); at run time they are rendered per case and
    sent to the model. ``complete(model, messages) -> str`` overrides the model call (used in
    tests / custom providers); the default lazily dispatches to anthropic/openai.
    """
    if output_type not in OUTPUT_TYPES:
        raise ValueError(f"output_type must be one of {OUTPUT_TYPES}, got {output_type!r}")
    if required_inputs is not None:
        required_inputs = _validate_required_inputs(required_inputs)

    from traceroot.constants import SpanKind
    from traceroot.decorators import observe
    from traceroot.eval.types import Score

    def _call(rendered_messages: list[dict[str, str]]) -> str:
        return (complete or _default_complete)(model, rendered_messages)

    # Self-instrument the model call as an LLM span (nested under the scorer span) so the
    # judge's LLM interaction shows in the trace without the caller wiring up provider
    # auto-instrumentation: the rendered messages are the input, the model response the output.
    @observe(name=f"llm_judge:{name}", type=SpanKind.LLM, metadata={"model": model})
    def _call_instrumented(rendered_messages: list[dict[str, str]]) -> str:
        return _call(rendered_messages)

    def judge(ctx: Any) -> Any:
        rendered = _render_messages(messages, ctx)
        # If a provider integration is already tracing this model's calls, let IT own the LLM
        # span (richer: tokens, native semantics) instead of adding our own — otherwise we'd
        # nest an LLM span inside an LLM span. Self-instrument only when nothing else will.
        invoke = _call if _provider_integration_traces(model) else _call_instrumented
        text = invoke(rendered)
        return Score(name, _parse_judge_output(text, output_type), comment=(text or "")[:2000])

    judge.__name__ = name
    meta = {
        "name": name,
        "version": version,
        "scorer_type": "llm_judge",
        "model": model,
        "messages": messages,
        "output_type": output_type,
        "threshold": threshold,
        "description": description,
        "metadata": metadata,
        "direction": direction,
        "value_type": value_type,
        "required_inputs": required_inputs,
    }
    setattr(judge, _META_ATTR, {k: v for k, v in meta.items() if v is not None})
    return judge

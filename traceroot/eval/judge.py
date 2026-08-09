"""The LLM-judge scorer: a first-class scorer whose ``model`` + ``messages`` (the judge
prompt) are the reported definition, and calling it runs the judge over a case.

Split out of ``scorers.py`` so it owns its own module. Imports from ``scorers`` are done
lazily inside functions so importing this module never triggers a circular import.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any


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
    from traceroot.eval.scorers import _PLACEHOLDER

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


# Float-shaped: optional sign, leading-dot decimals (.8), and exponents (8e-1) — otherwise
# `.8` / `-.8` / `8e-1` mis-parse (e.g. `.8` matched as `8`).
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


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


# A general {{VAR}} placeholder (any identifier), used to render a dynamic judge's builder variables.
_ANY_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def _config_revision(**config: Any) -> str:
    """Deterministic revision of a judge's DECLARATIVE configuration (model/messages/policy/...): its
    versioned 'source'. Canonical JSON (sorted keys) -> sha256, so the same config always hashes the
    same across processes and languages that canonicalize identically."""
    import hashlib
    import json

    canon = json.dumps(config, sort_keys=True, default=str, ensure_ascii=False, separators=(",", ":"))
    return "cfg_" + hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def _render_vars(messages: list[dict[str, str]], variables: dict[str, str]) -> list[dict[str, str]]:
    """Render {{VAR}} placeholders from an explicit variable map; unknown placeholders are left."""
    def sub(mo: "re.Match[str]") -> str:
        return variables.get(mo.group(1), mo.group(0))

    return [
        {"role": m.get("role", "user"), "content": _ANY_PLACEHOLDER.sub(sub, m.get("content", ""))}
        for m in messages
    ]


def _render_built(messages: list[dict[str, str]], built: Any, ctx: Any, name: str) -> list[dict[str, str]]:
    """Render a dynamic judge from its builder's return value. A DICT is a template-variable map
    ({{VAR}} filled from it, with input/output/expected as fallbacks); a LIST is dynamic messages
    (used as-is, with {{input/output/expected}} still rendered). Anything else is an actionable
    error -- the builder supplies prompt inputs, never the final score."""
    if isinstance(built, dict):
        base = {
            "input": _as_text(getattr(ctx, "input", None)),
            "output": _as_text(getattr(ctx, "output", None)),
            "expected": _as_text(getattr(ctx, "expected", None)),
        }
        return _render_vars(messages, {**base, **{k: _as_text(v) for k, v in built.items()}})
    if isinstance(built, list):
        return _render_messages(built, ctx)
    raise TypeError(
        f"llm_judge {name!r}: a dynamic builder must return a dict (template variables) or a list "
        f"(messages), got {type(built).__name__}."
    )


class _JudgeScorer:
    """An LLM-judge scorer callable. Used directly it runs the judge for a case; applied to a builder
    function (``@Scorer.llm_judge(...)``) it returns a DYNAMIC judge that builds its prompt per case.
    Its metadata (config + version=config-hash) is the reported, hashable definition."""

    def __init__(self, *, run: Callable[[Any], Any], meta: dict, name: str,
                 make_dynamic: Callable[[Callable], "_JudgeScorer"] | None) -> None:
        from traceroot.eval.scorers import _META_ATTR

        self._run = run
        self._make_dynamic = make_dynamic
        self.__name__ = name
        setattr(self, _META_ATTR, meta)

    def __call__(self, arg: Any) -> Any:
        import inspect

        # Decorator application: `@Scorer.llm_judge(...)` over a builder function -> a dynamic judge.
        # Any other arg is a ScorerContext the engine passed to run the judge.
        if self._make_dynamic is not None and (inspect.isfunction(arg) or inspect.ismethod(arg)):
            return self._make_dynamic(arg)
        return self._run(arg)


def _build_judge(*, name: str, model: str, messages: list[dict[str, str]], output_type: str,
                 complete: Callable | None, meta: dict, builder: Callable | None) -> _JudgeScorer:
    """Construct a (static or dynamic) judge callable that renders the prompt, invokes the model,
    parses the score, and traces the call. `builder` (dynamic) produces template variables or
    messages per case; None (static) renders the authored messages from input/output/expected."""
    from traceroot.constants import SpanKind
    from traceroot.decorators import observe
    from traceroot.eval.types import Score

    def _call(rendered_messages: list[dict[str, str]]) -> str:
        return (complete or _default_complete)(model, rendered_messages)

    @observe(name=f"llm_judge:{name}", type=SpanKind.LLM, metadata={"model": model})
    def _call_instrumented(rendered_messages: list[dict[str, str]]) -> str:
        return _call(rendered_messages)

    def run(ctx: Any) -> Any:
        if builder is not None:
            from traceroot.eval.engine import _bind_scorer_args

            b_args, b_kwargs = _bind_scorer_args(builder, ctx)
            built = builder(*b_args, **b_kwargs)
            import inspect

            if inspect.isawaitable(built):
                raise TypeError(f"llm_judge {name!r}: the dynamic builder must be synchronous.")
            rendered = _render_built(messages, built, ctx, name)
        else:
            rendered = _render_messages(messages, ctx)
        provider_traced = complete is None and _provider_integration_traces(model)
        invoke = _call if provider_traced else _call_instrumented
        text = invoke(rendered)
        return Score(name, _parse_judge_output(text, output_type), comment=(text or "")[:2000])

    def make_dynamic(b: Callable) -> _JudgeScorer:
        return _build_judge(name=name, model=model, messages=messages, output_type=output_type,
                            complete=complete, meta=meta, builder=b)

    return _JudgeScorer(run=run, meta=meta, name=name,
                        make_dynamic=None if builder is not None else make_dynamic)


def llm_judge(
    *,
    name: str,
    key: str | None = None,
    model: str,
    messages: list[dict[str, str]] | None = None,
    rubric: str | None = None,
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
    from traceroot.eval.scorers import (
        DIRECTIONS,
        OUTPUT_TYPES,
        VALUE_TYPES,
        _validate_required_inputs,
    )

    if output_type not in OUTPUT_TYPES:
        raise ValueError(f"output_type must be one of {OUTPUT_TYPES}, got {output_type!r}")
    # Validate the comparison metadata up front, like scorer() does, so an invalid direction or
    # value_type can't reach the reported manifest.
    if value_type is not None and value_type not in VALUE_TYPES:
        raise ValueError(f"value_type must be one of {VALUE_TYPES}, got {value_type!r}")
    if direction is not None and direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if required_inputs is not None:
        required_inputs = _validate_required_inputs(required_inputs)
    # `rubric` is a shorthand for a system instruction that grades the output; `messages` is the full
    # authored template. Exactly one source of the prompt is required (no empty function needed).
    if messages is None and rubric is not None:
        messages = [
            {"role": "system", "content": rubric},
            {"role": "user", "content": "{{output}}"},
        ]
    if messages is None:
        raise ValueError("llm_judge requires `messages` or `rubric`")
    # The judge's declarative config IS its versioned source: hash it deterministically. An explicit
    # `version` still wins.
    resolved_version = version or _config_revision(
        model=model, messages=messages, output_type=output_type, threshold=threshold,
        direction=direction, value_type=value_type, metadata=metadata,
    )
    meta = {
        k: v
        for k, v in {
            "key": key,  # stable cross-language identity (defaults to name via scorer_metadata)
            "name": name,
            "version": resolved_version,
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
        }.items()
        if v is not None
    }
    return _build_judge(name=name, model=model, messages=messages, output_type=output_type,
                        complete=complete, meta=meta, builder=None)

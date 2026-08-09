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


def llm_judge(
    *,
    name: str,
    key: str | None = None,
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
    from traceroot.constants import SpanKind
    from traceroot.decorators import observe
    from traceroot.eval.scorers import (
        _META_ATTR,
        DIRECTIONS,
        OUTPUT_TYPES,
        VALUE_TYPES,
        _validate_required_inputs,
    )
    from traceroot.eval.types import Score

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
        # nest an LLM span inside an LLM span. This only holds for the DEFAULT dispatch: a
        # user-supplied `complete` is not provider-instrumented, so it must be self-instrumented
        # (else its call would have no span at all), regardless of the model id.
        provider_traced = complete is None and _provider_integration_traces(model)
        invoke = _call if provider_traced else _call_instrumented
        text = invoke(rendered)
        return Score(name, _parse_judge_output(text, output_type), comment=(text or "")[:2000])

    judge.__name__ = name
    meta = {
        "key": key,  # stable cross-language identity (defaults to name via scorer_metadata)
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

"""LangChain integration: OpenInference's instrumentor, plus the ChatAnthropic calls it records.

OpenInference creates LangChain spans from callbacks and never makes them current, so the code that
sends a request cannot find its LLM run through the OTel context. Tracer callbacks cannot tell it
either: on async paths they run in a copied context. And a chat model's stream methods are not
handed the run's manager. Every path does go through ``on_chat_model_start``, in the context the
model then runs in, so the runs are noted there. ``ChatAnthropic._create`` and ``_acreate``, which
send the request on every path, look them up.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any
from uuid import UUID

from openinference.instrumentation.langchain import LangChainInstrumentor as _Inner
from wrapt import register_post_import_hook, wrap_function_wrapper

from traceroot.transport.span_processor import mark_recorded_by

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import Span

logger = logging.getLogger(__name__)

# Never reset: the callback manager gives no scope to do it in. Ids, not spans, so that nothing
# outlives its run, and a finished run is simply no longer found.
_llm_run_ids: ContextVar[tuple[UUID, ...]] = ContextVar("traceroot_llm_run_ids", default=())


def _on_chat_model_start(
    wrapped: Callable[..., Any], _instance: Any, args: Any, kwargs: Any
) -> Any:
    run_managers = wrapped(*args, **kwargs)
    _llm_run_ids.set(tuple(manager.run_id for manager in run_managers))
    return run_managers


async def _aon_chat_model_start(
    wrapped: Callable[..., Any], _instance: Any, args: Any, kwargs: Any
) -> Any:
    run_managers = await wrapped(*args, **kwargs)
    _llm_run_ids.set(tuple(manager.run_id for manager in run_managers))
    return run_managers


def _llm_span() -> Span | None:
    """The span of the LLM run the current model call belongs to, while it is open."""
    try:
        instrumentor = _Inner()
        if not instrumentor.is_instrumented_by_opentelemetry:
            return None
        spans = (instrumentor.get_span(run_id) for run_id in _llm_run_ids.get())
        return next((span for span in spans if span is not None), None)
    except Exception:
        logger.debug("Could not find the span of a ChatAnthropic call", exc_info=True)
        return None


def _create(wrapped: Callable[..., Any], _instance: Any, args: Any, kwargs: Any) -> Any:
    with mark_recorded_by(_llm_span()):
        return wrapped(*args, **kwargs)


async def _acreate(wrapped: Callable[..., Any], _instance: Any, args: Any, kwargs: Any) -> Any:
    with mark_recorded_by(_llm_span()):
        return await wrapped(*args, **kwargs)


def _wrap_chat_anthropic(_module: Any) -> None:
    # Runs inside the application's import of langchain_anthropic, which must not fail because of it.
    try:
        module = "langchain_anthropic.chat_models"
        wrap_function_wrapper(module, "ChatAnthropic._create", _create)
        wrap_function_wrapper(module, "ChatAnthropic._acreate", _acreate)
    except Exception:
        logger.warning(
            "traceroot: could not wrap ChatAnthropic; its calls may be counted twice. "
            "Set ANTHROPIC_OPEN_TELEMETRY=false before creating Anthropic clients to avoid that.",
            exc_info=True,
        )


class LangChainInstrumentor:
    """Installs OpenInference's LangChainInstrumentor and wraps ChatAnthropic, now or once imported."""

    _hooked: bool = False

    def instrument(self, tracer_provider: TracerProvider | None = None, **kwargs: Any) -> None:
        _Inner().instrument(tracer_provider=tracer_provider, **kwargs)
        if LangChainInstrumentor._hooked:
            return
        module = "langchain_core.callbacks.manager"
        wrap_function_wrapper(module, "CallbackManager.on_chat_model_start", _on_chat_model_start)
        wrap_function_wrapper(
            module, "AsyncCallbackManager.on_chat_model_start", _aon_chat_model_start
        )
        register_post_import_hook(_wrap_chat_anthropic, "langchain_anthropic")
        LangChainInstrumentor._hooked = True

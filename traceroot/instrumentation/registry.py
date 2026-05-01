"""Instrumentor registry and initialization logic."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

logger = logging.getLogger(__name__)


class Integration(StrEnum):
    """Supported auto-instrumentation targets."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LANGCHAIN = "langchain"
    GOOGLE_GENAI = "google_genai"
    CREWAI = "crewai"
    OPENAI_AGENTS = "openai_agents"
    CLAUDE_AGENT_SDK = "claude_agent_sdk"
    LLAMA_INDEX = "llama_index"
    AUTOGEN = "autogen"
    AGNO = "agno"
    GROQ = "groq"
    DSPY = "dspy"
    GOOGLE_ADK = "google_adk"
    MISTRAL = "mistral"


# Maps Integration enum ->
# (library to detect, instrumentor module path, instrumentor class name)
_BUILTIN_REGISTRY: dict[Integration, tuple[str, str, str]] = {
    Integration.OPENAI: (
        "openai",
        "openinference.instrumentation.openai",
        "OpenAIInstrumentor",
    ),
    Integration.ANTHROPIC: (
        "anthropic",
        "openinference.instrumentation.anthropic",
        "AnthropicInstrumentor",
    ),
    Integration.LANGCHAIN: (
        "langchain",
        "openinference.instrumentation.langchain",
        "LangChainInstrumentor",
    ),
    Integration.GOOGLE_GENAI: (
        "google-genai",
        "openinference.instrumentation.google_genai",
        "GoogleGenAIInstrumentor",
    ),
    Integration.CREWAI: (
        "crewai",
        "openinference.instrumentation.crewai",
        "CrewAIInstrumentor",
    ),
    Integration.OPENAI_AGENTS: (
        "openai-agents",
        "openinference.instrumentation.openai_agents",
        "OpenAIAgentsInstrumentor",
    ),
    Integration.CLAUDE_AGENT_SDK: (
        "claude-agent-sdk",
        "openinference.instrumentation.claude_agent_sdk",
        "ClaudeAgentSDKInstrumentor",
    ),
    Integration.LLAMA_INDEX: (
        "llama-index-core",
        "openinference.instrumentation.llama_index",
        "LlamaIndexInstrumentor",
    ),
    Integration.AUTOGEN: (
        "ag2",
        "openinference.instrumentation.autogen",
        "AutogenInstrumentor",
    ),
    Integration.AGNO: (
        "agno",
        "openinference.instrumentation.agno",
        "AgnoInstrumentor",
    ),
    Integration.GROQ: (
        "groq",
        "openinference.instrumentation.groq",
        "GroqInstrumentor",
    ),
    Integration.DSPY: (
        "dspy",
        "openinference.instrumentation.dspy",
        "DSPyInstrumentor",
    ),
    Integration.GOOGLE_ADK: (
        "google-adk",
        "openinference.instrumentation.google_adk",
        "GoogleADKInstrumentor",
    ),
    Integration.MISTRAL: (
        "mistralai",
        "openinference.instrumentation.mistralai",
        "MistralAIInstrumentor",
    ),
}


def _is_package_installed(package_name: str) -> bool:
    """Check if a Python package is installed."""
    try:
        importlib.metadata.version(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def initialize_integrations(
    tracer_provider: TracerProvider,
    integrations: Sequence[Integration],
) -> list[Integration]:
    """Initialize instrumentation for the specified libraries.

    Args:
        tracer_provider: The OTel TracerProvider to pass to instrumentors.
        integrations: List of Integration enum values to instrument.

    Returns:
        List of Integration values that were successfully instrumented.

    Raises:
        ImportError: If a requested library is not installed.
    """
    instrumented: list[Integration] = []

    for instrument in integrations:
        library, module_path, class_name = _BUILTIN_REGISTRY[instrument]

        if not _is_package_installed(library):
            logger.warning(
                "traceroot: skipping %s integration — '%s' is not installed. "
                "Install it with: pip install %s",
                instrument.value,
                library,
                library,
            )
            continue

        try:
            module = importlib.import_module(module_path)
            instrumentor_cls = getattr(module, class_name)
            instrumentor = instrumentor_cls()

            # AutoGen instrumentor (OpenInference) currently does not accept tracer_provider
            # https://github.com/Arize-ai/openinference/blob/main/python/instrumentation/openinference-instrumentation-autogen/src/openinference/instrumentation/autogen/__init__.py
            if instrument is Integration.AUTOGEN:
                instrumentor.instrument()
            else:
                instrumentor.instrument(tracer_provider=tracer_provider)

            logger.info(
                "Instrumented %s via %s.%s",
                library,
                module_path,
                class_name,
            )
            instrumented.append(instrument)
        except Exception:
            logger.warning("Failed to instrument %s", library, exc_info=True)

    return instrumented

"""Instrumentor registry and initialization logic."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
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
    PYDANTIC_AI = "pydantic_ai"
    BEDROCK = "bedrock"
    AGENT_FRAMEWORK = "agent_framework"
    LIVEKIT = "livekit"


@dataclass
class InstrumentorEntry:
    """Registry entry: import module_path, get class_name, call .instrument()."""

    package: str
    module_path: str
    class_name: str
    # Alternative package names that also provide this integration (e.g. slim variants).
    # The integration is considered available if *any* of package or alt_packages is installed.
    alt_packages: tuple[str, ...] = field(default_factory=tuple)

    @property
    def all_packages(self) -> tuple[str, ...]:
        return (self.package,) + self.alt_packages


_BUILTIN_REGISTRY: dict[Integration, InstrumentorEntry] = {
    Integration.OPENAI: InstrumentorEntry(
        package="openai",
        module_path="openinference.instrumentation.openai",
        class_name="OpenAIInstrumentor",
    ),
    Integration.ANTHROPIC: InstrumentorEntry(
        package="anthropic",
        module_path="openinference.instrumentation.anthropic",
        class_name="AnthropicInstrumentor",
    ),
    Integration.LANGCHAIN: InstrumentorEntry(
        # The instrumentor patches `langchain_core.callbacks.BaseCallbackManager`
        # (its `_MODULE` is "langchain_core"), so langchain-core is the real
        # dependency. The umbrella `langchain` distribution is optional: a LangGraph
        # app typically installs only langchain-core + langgraph + provider packages,
        # and gating on `langchain` skipped instrumentation for those apps entirely.
        package="langchain-core",
        module_path="openinference.instrumentation.langchain",
        class_name="LangChainInstrumentor",
        alt_packages=("langchain",),
    ),
    Integration.GOOGLE_GENAI: InstrumentorEntry(
        package="google-genai",
        module_path="openinference.instrumentation.google_genai",
        class_name="GoogleGenAIInstrumentor",
    ),
    Integration.CREWAI: InstrumentorEntry(
        package="crewai",
        module_path="openinference.instrumentation.crewai",
        class_name="CrewAIInstrumentor",
    ),
    Integration.OPENAI_AGENTS: InstrumentorEntry(
        package="openai-agents",
        module_path="openinference.instrumentation.openai_agents",
        class_name="OpenAIAgentsInstrumentor",
    ),
    Integration.CLAUDE_AGENT_SDK: InstrumentorEntry(
        package="claude-agent-sdk",
        module_path="openinference.instrumentation.claude_agent_sdk",
        class_name="ClaudeAgentSDKInstrumentor",
    ),
    Integration.LLAMA_INDEX: InstrumentorEntry(
        package="llama-index-core",
        module_path="openinference.instrumentation.llama_index",
        class_name="LlamaIndexInstrumentor",
    ),
    Integration.AUTOGEN: InstrumentorEntry(
        package="ag2",
        module_path="traceroot.instrumentation._instrumentors",
        class_name="AutogenInstrumentor",
    ),
    Integration.AGNO: InstrumentorEntry(
        package="agno",
        module_path="openinference.instrumentation.agno",
        class_name="AgnoInstrumentor",
    ),
    Integration.GROQ: InstrumentorEntry(
        package="groq",
        module_path="openinference.instrumentation.groq",
        class_name="GroqInstrumentor",
    ),
    Integration.DSPY: InstrumentorEntry(
        package="dspy",
        module_path="openinference.instrumentation.dspy",
        class_name="DSPyInstrumentor",
        # DSPy publishes the same release under both names (`dspy-ai` is the legacy
        # distribution), so accept either.
        alt_packages=("dspy-ai",),
    ),
    Integration.GOOGLE_ADK: InstrumentorEntry(
        package="google-adk",
        module_path="openinference.instrumentation.google_adk",
        class_name="GoogleADKInstrumentor",
    ),
    Integration.MISTRAL: InstrumentorEntry(
        package="mistralai",
        module_path="openinference.instrumentation.mistralai",
        class_name="MistralAIInstrumentor",
    ),
    Integration.PYDANTIC_AI: InstrumentorEntry(
        package="pydantic-ai",
        alt_packages=("pydantic-ai-slim",),
        module_path="traceroot.instrumentation._instrumentors",
        class_name="PydanticAIInstrumentor",
    ),
    Integration.BEDROCK: InstrumentorEntry(
        package="boto3",
        module_path="openinference.instrumentation.bedrock",
        class_name="BedrockInstrumentor",
    ),
    Integration.AGENT_FRAMEWORK: InstrumentorEntry(
        package="agent-framework-core",
        module_path="traceroot.instrumentation._instrumentors",
        class_name="AgentFrameworkInstrumentor",
    ),
    Integration.LIVEKIT: InstrumentorEntry(
        package="livekit-agents",
        module_path="traceroot.instrumentation.livekit",
        class_name="LiveKitInstrumentor",
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
    """
    instrumented: list[Integration] = []

    for instrument in integrations:
        entry = _BUILTIN_REGISTRY[instrument]

        if not any(_is_package_installed(p) for p in entry.all_packages):
            all_pkgs = entry.all_packages
            pkg_label = (
                f"'{all_pkgs[0]}'"
                if len(all_pkgs) == 1
                else " or ".join(f"'{p}'" for p in all_pkgs)
            )
            logger.warning(
                "traceroot: skipping %s integration — %s is not installed. "
                "Install it with: pip install %s",
                instrument.value,
                pkg_label,
                all_pkgs[0],
            )
            continue

        # In-house instrumentation for Claude Agent SDK (replaces OpenInference)
        if instrument is Integration.CLAUDE_AGENT_SDK:
            try:
                from traceroot.instrumentation.claude_agent_sdk import instrument_claude_agent_sdk

                instrument_claude_agent_sdk(tracer_provider)
                logger.info("Instrumented %s via in-house wrapper", instrument.value)
                instrumented.append(instrument)
            except Exception:
                logger.warning("Failed to instrument %s", instrument.value, exc_info=True)
            continue

        try:
            module = importlib.import_module(entry.module_path)
            instrumentor_cls = getattr(module, entry.class_name)
            instrumentor = instrumentor_cls()
            instrumentor.instrument(tracer_provider=tracer_provider)
            logger.info("Instrumented %s via %s", instrument.value, entry.module_path)
            instrumented.append(instrument)
        except Exception:
            logger.warning("Failed to instrument %s", instrument.value, exc_info=True)

    return instrumented

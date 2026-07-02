"""Tests for auto-instrumentation registry and initialization."""

from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider

import traceroot
from tests.utils import reset_traceroot
from traceroot.instrumentation._instrumentors import (
    AgentFrameworkInstrumentor,
    AutogenInstrumentor,
    PydanticAIInstrumentor,
)
from traceroot.instrumentation.livekit import LiveKitInstrumentor, LiveKitToOpenInferenceProcessor
from traceroot.instrumentation.registry import (
    Integration,
    _is_package_installed,
    initialize_integrations,
)

# =============================================================================
# Integration enum
# =============================================================================


def test_integration_enum_values():
    assert Integration.OPENAI == "openai"
    assert Integration.ANTHROPIC == "anthropic"
    assert Integration.LANGCHAIN == "langchain"
    assert Integration.GOOGLE_GENAI == "google_genai"
    assert Integration.OPENAI_AGENTS == "openai_agents"
    assert Integration.CREWAI == "crewai"
    assert Integration.LLAMA_INDEX == "llama_index"
    assert Integration.DSPY == "dspy"
    assert Integration.PYDANTIC_AI == "pydantic_ai"
    assert Integration.BEDROCK == "bedrock"
    assert Integration.LIVEKIT == "livekit"


def test_integration_exported_from_traceroot():
    assert traceroot.Integration is Integration


# =============================================================================
# _is_package_installed
# =============================================================================


def test_is_package_installed_for_installed_package():
    assert _is_package_installed("opentelemetry-api") is True


def test_is_package_installed_for_missing_package():
    assert _is_package_installed("nonexistent-package-xyz-12345") is False


# =============================================================================
# initialize_integrations
# =============================================================================


def test_empty_integrations_returns_empty():
    provider = TracerProvider()
    result = initialize_integrations(tracer_provider=provider, integrations=[])
    assert result == []


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_warns_and_skips_if_library_not_installed(mock_installed, caplog):
    import logging

    mock_installed.return_value = False

    provider = TracerProvider()
    with caplog.at_level(logging.WARNING, logger="traceroot.instrumentation.registry"):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.OPENAI],
        )

    assert result == []
    assert "skipping" in caplog.text
    assert "openai" in caplog.text


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_integrations_with_enum_values(mock_installed):
    mock_installed.return_value = True
    mock_instrumentor = MagicMock()
    mock_cls = MagicMock(return_value=mock_instrumentor)
    mock_module = MagicMock()
    mock_module.OpenAIInstrumentor = mock_cls

    provider = TracerProvider()

    with patch("importlib.import_module", return_value=mock_module):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.OPENAI],
        )

    assert result == [Integration.OPENAI]
    mock_instrumentor.instrument.assert_called_once_with(tracer_provider=provider)


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_integrations_multiple_enums(mock_installed):
    mock_installed.return_value = True
    mock_instrumentor = MagicMock()
    mock_cls = MagicMock(return_value=mock_instrumentor)
    mock_module = MagicMock()
    mock_module.OpenAIInstrumentor = mock_cls
    mock_module.AnthropicInstrumentor = mock_cls

    provider = TracerProvider()

    with patch("importlib.import_module", return_value=mock_module):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.OPENAI, Integration.ANTHROPIC],
        )

    assert Integration.OPENAI in result
    assert Integration.ANTHROPIC in result


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_crewai_integration_uses_crewai_instrumentor(mock_installed):
    mock_installed.return_value = True
    mock_instrumentor = MagicMock()
    mock_cls = MagicMock(return_value=mock_instrumentor)
    mock_module = MagicMock()
    mock_module.CrewAIInstrumentor = mock_cls

    provider = TracerProvider()

    with patch("importlib.import_module", return_value=mock_module):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.CREWAI],
        )

    assert result == [Integration.CREWAI]
    mock_instrumentor.instrument.assert_called_once_with(tracer_provider=provider)


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_crewai_missing_warns_and_skips(mock_installed, caplog):
    import logging

    mock_installed.return_value = False

    provider = TracerProvider()
    with caplog.at_level(logging.WARNING, logger="traceroot.instrumentation.registry"):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.CREWAI],
        )

    assert result == []
    assert "skipping" in caplog.text
    assert "crewai" in caplog.text


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_crewai_can_be_requested_with_other_integrations(mock_installed):
    mock_installed.return_value = True

    def import_module(name):
        module = MagicMock()
        if name == "openinference.instrumentation.crewai":
            module.CrewAIInstrumentor = MagicMock(return_value=MagicMock())
        elif name == "openinference.instrumentation.openai":
            module.OpenAIInstrumentor = MagicMock(return_value=MagicMock())
        else:
            raise AssertionError(f"unexpected module import: {name}")
        return module

    provider = TracerProvider()

    with patch("importlib.import_module", side_effect=import_module):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.OPENAI, Integration.CREWAI],
        )

    assert result == [Integration.OPENAI, Integration.CREWAI]


# =============================================================================
# LlamaIndex integration
# =============================================================================


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_llamaindex_integration_uses_llamaindex_instrumentor(mock_installed):
    mock_installed.return_value = True
    mock_instrumentor = MagicMock()
    mock_cls = MagicMock(return_value=mock_instrumentor)
    mock_module = MagicMock()
    mock_module.LlamaIndexInstrumentor = mock_cls

    provider = TracerProvider()

    with patch("importlib.import_module", return_value=mock_module):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.LLAMA_INDEX],
        )

    assert result == [Integration.LLAMA_INDEX]
    mock_cls.assert_called_once()
    mock_instrumentor.instrument.assert_called_once_with(tracer_provider=provider)


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_llamaindex_missing_warns_and_skips(mock_installed, caplog):
    import logging

    mock_installed.return_value = False

    provider = TracerProvider()
    with caplog.at_level(logging.WARNING, logger="traceroot.instrumentation.registry"):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.LLAMA_INDEX],
        )

    assert result == []
    assert "skipping" in caplog.text
    assert "llama-index-core" in caplog.text


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_llamaindex_can_be_requested_with_other_integrations(mock_installed):
    mock_installed.return_value = True

    def import_module(name):
        module = MagicMock()
        if name == "openinference.instrumentation.llama_index":
            module.LlamaIndexInstrumentor = MagicMock(return_value=MagicMock())
        elif name == "openinference.instrumentation.openai":
            module.OpenAIInstrumentor = MagicMock(return_value=MagicMock())
        else:
            raise AssertionError(f"unexpected module import: {name}")
        return module

    provider = TracerProvider()

    with patch("importlib.import_module", side_effect=import_module):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.OPENAI, Integration.LLAMA_INDEX],
        )

    assert result == [Integration.OPENAI, Integration.LLAMA_INDEX]


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_failed_instrumentation_continues(mock_installed):
    """If one instrumentor fails, others still get instrumented."""
    mock_installed.return_value = True

    provider = TracerProvider()

    with patch("importlib.import_module", side_effect=ImportError("no module")):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.OPENAI],
        )

    # Failed but didn't raise — just not in results
    assert result == []


# =============================================================================
# TracerootClient integration
# =============================================================================


@patch("traceroot.instrumentation.registry.initialize_integrations")
@patch("opentelemetry.trace.set_tracer_provider")
def test_client_calls_initialize_integrations(mock_set_provider, mock_init):
    mock_init.return_value = []
    reset_traceroot()

    traceroot.initialize(
        api_key="test-key",
        integrations=[Integration.OPENAI, Integration.LANGCHAIN],
    )

    mock_init.assert_called_once()
    _, kwargs = mock_init.call_args
    assert kwargs["integrations"] == [Integration.OPENAI, Integration.LANGCHAIN]


@patch("opentelemetry.trace.set_tracer_provider")
def test_client_skips_instrumentation_when_not_requested(mock_set_provider):
    reset_traceroot()

    with patch("traceroot.instrumentation.registry.initialize_integrations") as mock_init:
        traceroot.initialize(api_key="test-key")
        mock_init.assert_not_called()


def test_client_skips_instrumentation_when_disabled():
    reset_traceroot()

    with patch("traceroot.instrumentation.registry.initialize_integrations") as mock_init:
        traceroot.initialize(enabled=False, integrations=[Integration.OPENAI])
        mock_init.assert_not_called()


# =============================================================================
# Agno integration
# =============================================================================


def test_agno_integration_enum_value():
    assert Integration.AGNO == "agno"


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_agno_integration_uses_agno_instrumentor(mock_installed):
    mock_installed.return_value = True
    mock_instrumentor = MagicMock()
    mock_cls = MagicMock(return_value=mock_instrumentor)
    mock_module = MagicMock()
    mock_module.AgnoInstrumentor = mock_cls

    provider = TracerProvider()

    with patch("importlib.import_module", return_value=mock_module):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.AGNO],
        )

    assert result == [Integration.AGNO]
    mock_instrumentor.instrument.assert_called_once_with(tracer_provider=provider)


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_agno_missing_warns_and_skips(mock_installed, caplog):
    import logging

    mock_installed.return_value = False

    provider = TracerProvider()
    with caplog.at_level(logging.WARNING, logger="traceroot.instrumentation.registry"):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.AGNO],
        )

    assert result == []
    assert "skipping" in caplog.text
    assert "agno" in caplog.text


# =============================================================================
# Groq integration
# =============================================================================


def test_groq_integration_enum_value():
    assert Integration.GROQ == "groq"


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_groq_integration_uses_groq_instrumentor(mock_installed):
    mock_installed.return_value = True
    mock_instrumentor = MagicMock()
    mock_cls = MagicMock(return_value=mock_instrumentor)
    mock_module = MagicMock()
    mock_module.GroqInstrumentor = mock_cls

    provider = TracerProvider()

    with patch("importlib.import_module", return_value=mock_module):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.GROQ],
        )

    assert result == [Integration.GROQ]
    mock_instrumentor.instrument.assert_called_once_with(tracer_provider=provider)


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_groq_missing_warns_and_skips(mock_installed, caplog):
    import logging

    mock_installed.return_value = False

    provider = TracerProvider()
    with caplog.at_level(logging.WARNING, logger="traceroot.instrumentation.registry"):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.GROQ],
        )

    assert result == []
    assert "skipping" in caplog.text
    assert "groq" in caplog.text


# =============================================================================
# DSPy integration
# =============================================================================


def test_dspy_integration_enum_value():
    assert Integration.DSPY == "dspy"


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_dspy_integration_uses_dspy_instrumentor(mock_installed):
    mock_installed.return_value = True
    mock_instrumentor = MagicMock()
    mock_cls = MagicMock(return_value=mock_instrumentor)
    mock_module = MagicMock()
    mock_module.DSPyInstrumentor = mock_cls

    provider = TracerProvider()

    with patch("importlib.import_module", return_value=mock_module):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.DSPY],
        )

    assert result == [Integration.DSPY]
    mock_instrumentor.instrument.assert_called_once_with(tracer_provider=provider)


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_dspy_missing_warns_and_skips(mock_installed, caplog):
    import logging

    mock_installed.return_value = False

    provider = TracerProvider()
    with caplog.at_level(logging.WARNING, logger="traceroot.instrumentation.registry"):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.DSPY],
        )

    assert result == []
    assert "skipping" in caplog.text
    assert "dspy" in caplog.text


# =============================================================================
# Mistral integration
# =============================================================================


def test_mistral_integration_enum_value():
    assert Integration.MISTRAL == "mistral"


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_mistral_integration_uses_mistralai_instrumentor(mock_installed):
    mock_installed.return_value = True
    mock_instrumentor = MagicMock()
    mock_cls = MagicMock(return_value=mock_instrumentor)
    mock_module = MagicMock()
    mock_module.MistralAIInstrumentor = mock_cls

    provider = TracerProvider()

    with patch("importlib.import_module", return_value=mock_module):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.MISTRAL],
        )

    assert result == [Integration.MISTRAL]
    mock_instrumentor.instrument.assert_called_once_with(tracer_provider=provider)


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_mistral_missing_warns_and_skips(mock_installed, caplog):
    import logging

    mock_installed.return_value = False

    provider = TracerProvider()
    with caplog.at_level(logging.WARNING, logger="traceroot.instrumentation.registry"):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.MISTRAL],
        )

    assert result == []
    assert "skipping" in caplog.text
    assert "mistralai" in caplog.text


# =============================================================================
# Pydantic AI integration
# =============================================================================


@pytest.fixture(autouse=True)
def reset_custom_instrumentors():
    """Reset adapter instrumentor flags before and after each test."""
    AutogenInstrumentor._instrumented = False
    PydanticAIInstrumentor._instrumented = False
    AgentFrameworkInstrumentor._instrumented = False
    yield
    AutogenInstrumentor._instrumented = False
    PydanticAIInstrumentor._instrumented = False
    AgentFrameworkInstrumentor._instrumented = False


def test_pydantic_ai_integration_enum_value():
    assert Integration.PYDANTIC_AI == "pydantic_ai"


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_pydantic_ai_skipped_if_not_installed(mock_installed, caplog):
    import logging

    mock_installed.return_value = False

    provider = TracerProvider()
    with caplog.at_level(logging.WARNING, logger="traceroot.instrumentation.registry"):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.PYDANTIC_AI],
        )

    assert result == []
    assert "skipping" in caplog.text
    assert "pydantic-ai" in caplog.text


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_pydantic_ai_installs_processor_and_calls_instrument_all(mock_installed):
    import sys

    mock_installed.return_value = True

    mock_processor = MagicMock()
    mock_processor_cls = MagicMock(return_value=mock_processor)
    mock_agent_cls = MagicMock()

    provider = MagicMock(spec=TracerProvider)

    # Pass sys.modules (the object) not "sys.modules" (a string) so patch.dict
    # doesn't try to resolve it via importlib and hit an unrelated mock.
    with patch.dict(
        sys.modules,
        {
            "openinference.instrumentation.pydantic_ai": MagicMock(
                OpenInferenceSpanProcessor=mock_processor_cls
            ),
            "pydantic_ai": MagicMock(Agent=mock_agent_cls),
        },
    ):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.PYDANTIC_AI],
        )

    assert result == [Integration.PYDANTIC_AI]
    mock_processor_cls.assert_called_once_with()
    provider.add_span_processor.assert_called_once_with(mock_processor)
    mock_agent_cls.instrument_all.assert_called_once_with()


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_pydantic_ai_idempotent(mock_installed):
    """Calling initialize_integrations twice must not add the processor twice."""
    import sys

    mock_installed.return_value = True

    mock_processor = MagicMock()
    mock_processor_cls = MagicMock(return_value=mock_processor)
    mock_agent_cls = MagicMock()

    provider = MagicMock(spec=TracerProvider)

    with patch.dict(
        sys.modules,
        {
            "openinference.instrumentation.pydantic_ai": MagicMock(
                OpenInferenceSpanProcessor=mock_processor_cls
            ),
            "pydantic_ai": MagicMock(Agent=mock_agent_cls),
        },
    ):
        initialize_integrations(provider, [Integration.PYDANTIC_AI])
        initialize_integrations(provider, [Integration.PYDANTIC_AI])

    assert provider.add_span_processor.call_count == 1
    assert mock_agent_cls.instrument_all.call_count == 1


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_pydantic_ai_can_be_requested_with_other_integrations(mock_installed):
    import sys

    mock_installed.return_value = True

    mock_oi_processor = MagicMock()
    mock_agent_cls = MagicMock()
    mock_openai_instrumentor = MagicMock()
    mock_openai_cls = MagicMock(return_value=mock_openai_instrumentor)

    provider = MagicMock(spec=TracerProvider)

    # Inject all third-party modules via sys.modules so importlib.import_module
    # finds them via the cache without importing or patching the function itself.
    # traceroot.instrumentation._instrumentors is our own module and imports normally.
    with patch.dict(
        sys.modules,
        {
            "openinference.instrumentation.openai": MagicMock(OpenAIInstrumentor=mock_openai_cls),
            "openinference.instrumentation.pydantic_ai": MagicMock(
                OpenInferenceSpanProcessor=MagicMock(return_value=mock_oi_processor)
            ),
            "pydantic_ai": MagicMock(Agent=mock_agent_cls),
        },
    ):
        result = initialize_integrations(
            provider,
            [Integration.OPENAI, Integration.PYDANTIC_AI],
        )

    assert Integration.OPENAI in result
    assert Integration.PYDANTIC_AI in result
    mock_openai_instrumentor.instrument.assert_called_once_with(tracer_provider=provider)
    provider.add_span_processor.assert_called_once_with(mock_oi_processor)
    mock_agent_cls.instrument_all.assert_called_once_with()


# =============================================================================
# Bedrock integration
# =============================================================================


def test_bedrock_integration_enum_value():
    assert Integration.BEDROCK == "bedrock"


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_bedrock_integration_uses_bedrock_instrumentor(mock_installed):
    mock_installed.return_value = True
    mock_instrumentor = MagicMock()
    mock_cls = MagicMock(return_value=mock_instrumentor)
    mock_module = MagicMock()
    mock_module.BedrockInstrumentor = mock_cls

    provider = TracerProvider()

    with patch("importlib.import_module", return_value=mock_module):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.BEDROCK],
        )

    assert result == [Integration.BEDROCK]
    mock_instrumentor.instrument.assert_called_once_with(tracer_provider=provider)


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_bedrock_missing_warns_and_skips(mock_installed, caplog):
    import logging

    mock_installed.return_value = False

    provider = TracerProvider()
    with caplog.at_level(logging.WARNING, logger="traceroot.instrumentation.registry"):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.BEDROCK],
        )

    assert result == []
    assert "skipping" in caplog.text
    assert "boto3" in caplog.text


# =============================================================================
# Microsoft Agent Framework integration
# =============================================================================


def test_agent_framework_integration_enum_value():
    assert Integration.AGENT_FRAMEWORK == "agent_framework"


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_agent_framework_skipped_if_not_installed(mock_installed, caplog):
    import logging

    mock_installed.return_value = False

    provider = TracerProvider()
    with caplog.at_level(logging.WARNING, logger="traceroot.instrumentation.registry"):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.AGENT_FRAMEWORK],
        )

    assert result == []
    assert "skipping" in caplog.text
    assert "agent-framework-core" in caplog.text


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_agent_framework_installs_processor_and_enables_instrumentation(mock_installed):
    import sys

    mock_installed.return_value = True

    mock_processor = MagicMock()
    mock_processor_cls = MagicMock(return_value=mock_processor)
    mock_enable = MagicMock()

    provider = MagicMock(spec=TracerProvider)

    with patch.dict(
        sys.modules,
        {
            "openinference.instrumentation.agent_framework": MagicMock(
                AgentFrameworkToOpenInferenceProcessor=mock_processor_cls
            ),
            "agent_framework.observability": MagicMock(enable_instrumentation=mock_enable),
        },
    ):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.AGENT_FRAMEWORK],
        )

    assert result == [Integration.AGENT_FRAMEWORK]
    mock_processor_cls.assert_called_once_with()
    provider.add_span_processor.assert_called_once_with(mock_processor)
    mock_enable.assert_called_once_with(enable_sensitive_data=True)


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_agent_framework_idempotent(mock_installed):
    """Calling initialize_integrations twice must not re-register the processor."""
    import sys

    mock_installed.return_value = True

    mock_processor = MagicMock()
    mock_processor_cls = MagicMock(return_value=mock_processor)
    mock_enable = MagicMock()

    provider = MagicMock(spec=TracerProvider)

    with patch.dict(
        sys.modules,
        {
            "openinference.instrumentation.agent_framework": MagicMock(
                AgentFrameworkToOpenInferenceProcessor=mock_processor_cls
            ),
            "agent_framework.observability": MagicMock(enable_instrumentation=mock_enable),
        },
    ):
        initialize_integrations(provider, [Integration.AGENT_FRAMEWORK])
        initialize_integrations(provider, [Integration.AGENT_FRAMEWORK])

    assert provider.add_span_processor.call_count == 1
    assert mock_enable.call_count == 1


# =============================================================================
# LiveKit integration
# =============================================================================


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_livekit_missing_warns_and_skips(mock_installed, caplog):
    import logging

    mock_installed.return_value = False

    provider = TracerProvider()
    with caplog.at_level(logging.WARNING, logger="traceroot.instrumentation.registry"):
        result = initialize_integrations(
            tracer_provider=provider,
            integrations=[Integration.LIVEKIT],
        )

    assert result == []
    assert "skipping" in caplog.text
    assert "livekit-agents" in caplog.text


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_livekit_integration_sets_livekit_tracer_provider(mock_installed, monkeypatch):
    import sys
    import types

    LiveKitInstrumentor._instrumented = False
    mock_installed.return_value = True
    calls = []

    telemetry_module = types.ModuleType("livekit.agents.telemetry")

    def set_tracer_provider(tracer_provider):
        calls.append(tracer_provider)

    telemetry_module.set_tracer_provider = set_tracer_provider

    agents_module = types.ModuleType("livekit.agents")
    agents_module.telemetry = telemetry_module

    livekit_module = types.ModuleType("livekit")
    livekit_module.agents = agents_module

    monkeypatch.setitem(sys.modules, "livekit", livekit_module)
    monkeypatch.setitem(sys.modules, "livekit.agents", agents_module)
    monkeypatch.setitem(sys.modules, "livekit.agents.telemetry", telemetry_module)

    provider = TracerProvider()
    result = initialize_integrations(
        tracer_provider=provider,
        integrations=[Integration.LIVEKIT],
    )

    assert result == [Integration.LIVEKIT]
    assert calls == [provider]


def test_livekit_instrumentor_imports_livekit_only_during_instrument(monkeypatch):
    import builtins

    LiveKitInstrumentor._instrumented = False
    imported_names = []
    original_import = builtins.__import__

    def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        imported_names.append(name)
        if name == "livekit.agents.telemetry":
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", tracking_import)

    instrumentor = LiveKitInstrumentor()
    assert "livekit.agents.telemetry" not in imported_names

    with pytest.raises(ModuleNotFoundError):
        instrumentor.instrument(tracer_provider=TracerProvider())

    assert "livekit.agents.telemetry" in imported_names


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_livekit_contract_uses_existing_trace_context(mock_installed, monkeypatch):
    import sys
    import types

    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from traceroot import using_attributes
    from traceroot.transport.span_processor import TracerootSpanProcessor

    LiveKitInstrumentor._instrumented = False
    mock_installed.return_value = True
    provider_calls = []

    telemetry_module = types.ModuleType("livekit.agents.telemetry")

    def set_tracer_provider(tracer_provider):
        provider_calls.append(tracer_provider)

    telemetry_module.set_tracer_provider = set_tracer_provider

    agents_module = types.ModuleType("livekit.agents")
    agents_module.telemetry = telemetry_module

    livekit_module = types.ModuleType("livekit")
    livekit_module.agents = agents_module

    monkeypatch.setitem(sys.modules, "livekit", livekit_module)
    monkeypatch.setitem(sys.modules, "livekit.agents", agents_module)
    monkeypatch.setitem(sys.modules, "livekit.agents.telemetry", telemetry_module)
    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter.export",
        lambda _self, _spans: SpanExportResult.SUCCESS,
    )
    monkeypatch.setattr(
        "traceroot.transport.span_processor.OTLPSpanExporter.export",
        lambda _self, _spans: SpanExportResult.SUCCESS,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        TracerootSpanProcessor(
            api_key="test-key",
            host_url="http://127.0.0.1",
            flush_interval=9999,
        )
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    result = initialize_integrations(
        tracer_provider=provider,
        integrations=[Integration.LIVEKIT],
    )

    assert result == [Integration.LIVEKIT]
    assert provider_calls == [provider]
    assert isinstance(
        provider._active_span_processor._span_processors[0],
        LiveKitToOpenInferenceProcessor,
    )

    tracer = provider.get_tracer("fake-livekit")
    with (
        using_attributes(session_id="room-contract"),
        tracer.start_as_current_span("agent_session") as span,
    ):
        span.set_attribute("lk.response.text", "ready")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get("openinference.span.kind") == "AGENT"
    assert spans[0].attributes.get("output.value") == "ready"
    assert spans[0].attributes.get("session.id") == "room-contract"
    provider.shutdown()

"""Local smoke test for TraceRoot's LiveKit integration.

This does not start a real LiveKit agent. It installs a tiny fake
``livekit.agents.telemetry`` module so the example can verify the same provider
handoff path without requiring LiveKit credentials or a room.
"""

from __future__ import annotations

import json
import sys
import types

from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import traceroot
from traceroot import Integration, using_attributes


def install_fake_livekit() -> list[object]:
    """Install enough of LiveKit's telemetry module shape for a local smoke."""
    provider_calls: list[object] = []

    telemetry_module = types.ModuleType("livekit.agents.telemetry")
    telemetry_module.tracer = types.SimpleNamespace(_tracer_provider=None)

    def set_tracer_provider(tracer_provider: object) -> None:
        provider_calls.append(tracer_provider)
        telemetry_module.tracer._tracer_provider = tracer_provider

    telemetry_module.set_tracer_provider = set_tracer_provider

    agents_module = types.ModuleType("livekit.agents")
    agents_module.telemetry = telemetry_module

    livekit_module = types.ModuleType("livekit")
    livekit_module.agents = agents_module

    sys.modules["livekit"] = livekit_module
    sys.modules["livekit.agents"] = agents_module
    sys.modules["livekit.agents.telemetry"] = telemetry_module

    return provider_calls


def allow_fake_livekit_package_check() -> None:
    """Make the registry treat the fake LiveKit module as an installed package."""
    from traceroot.instrumentation import registry

    original_check = registry._is_package_installed

    def check(package_name: str) -> bool:
        if package_name == "livekit-agents":
            return True
        return original_check(package_name)

    registry._is_package_installed = check


def disable_network_export() -> None:
    """Keep this smoke local by replacing the OTLP HTTP export with success."""
    from traceroot.transport import span_processor

    span_processor.OTLPSpanExporter.export = lambda _self, _spans: SpanExportResult.SUCCESS


def main() -> None:
    provider_calls = install_fake_livekit()
    allow_fake_livekit_package_check()
    disable_network_export()

    client = traceroot.initialize(
        api_key="smoke-test-key",
        host_url="http://127.0.0.1",
        integrations=[Integration.LIVEKIT],
        flush_interval=9999,
        git_repo="traceroot-ai/traceroot-py",
        git_ref="livekit-smoke",
    )

    exporter = InMemorySpanExporter()
    assert client._provider is not None
    client._provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = client._provider.get_tracer("livekit-agents")
    with (
        using_attributes(session_id="room-smoke", user_id="user-smoke"),
        tracer.start_as_current_span("agent_session") as span,
    ):
        span.set_attribute("lk.response.text", "hello from livekit")

    traceroot.flush()
    spans = exporter.get_finished_spans()
    [span] = spans

    result = {
        "provider_handoff": provider_calls == [client._provider],
        "span_name": span.name,
        "session_id": span.attributes.get("session.id"),
        "user_id": span.attributes.get("user.id"),
        "openinference_span_kind": span.attributes.get("openinference.span.kind"),
        "output_value": span.attributes.get("output.value"),
    }

    assert result == {
        "provider_handoff": True,
        "span_name": "agent_session",
        "session_id": "room-smoke",
        "user_id": "user-smoke",
        "openinference_span_kind": "AGENT",
        "output_value": "hello from livekit",
    }

    print("INSTRUMENTATION_OK")
    print(json.dumps(result, indent=2, sort_keys=True))
    traceroot.shutdown()


if __name__ == "__main__":
    main()

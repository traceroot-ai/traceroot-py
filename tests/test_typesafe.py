import os
from unittest import mock

import httpx2
import pytest
from openinference.instrumentation.typesafe import TypeSafeAIInstrumentor
from opentelemetry.trace import SpanKind, StatusCode
from typesafe_sdk import TypeSafeClient
from typesafe_sdk._core.errors import TypeSafeBadRequestError

from traceroot import using_attributes
from traceroot.instrumentation.registry import Integration, initialize_integrations


class MockTransport(httpx2.BaseTransport):
    def __init__(self, status_code=200, json_body=None):
        self.status_code = status_code
        self.json_body = json_body

    def handle_request(self, request):
        if self.status_code >= 400:
            return httpx2.Response(self.status_code, request=request, json={"error": "bad request"})
        return httpx2.Response(self.status_code, request=request, json=self.json_body)


@pytest.fixture(autouse=True)
def setup_instrumentation(memory_exporter):
    from opentelemetry import trace

    assert initialize_integrations(trace.get_tracer_provider(), [Integration.TYPESAFE]) == [
        Integration.TYPESAFE
    ]
    yield
    TypeSafeAIInstrumentor().uninstrument()


def create_client(status_code=200, json_body=None):
    transport = MockTransport(status_code, json_body)
    return TypeSafeClient(api_key="mock", http_client=httpx2.Client(transport=transport))


def test_records_one_llm_span_per_system_one_call(memory_exporter):
    client = create_client(
        json_body={
            "model": "jev-1.13.0",
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "answers": {"test": {"type": "noul", "noul": 0.5}},
        }
    )
    with using_attributes(session_id="sess-1", user_id="u-1"):
        client.system_one(
            state="test state",
            questions={"test": {"type": "noul", "instructions": "test instructions"}},
        )

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.kind == SpanKind.INTERNAL
    assert span.attributes["llm.model_name"] == "jev-1.13.0"
    assert span.attributes["llm.token_count.prompt"] == 100
    assert span.attributes["llm.token_count.completion"] == 20
    assert span.attributes["llm.token_count.total"] == 120
    assert span.attributes["session.id"] == "sess-1"
    assert span.attributes["user.id"] == "u-1"
    assert "test state" in span.attributes["input.value"]
    assert span.status.status_code == StatusCode.OK


def test_alias_request_records_resolved_model(memory_exporter):
    client = create_client(
        json_body={
            "model": "jev-1.13.0",
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "answers": {"test": {"type": "noul", "noul": 0.5}},
        }
    )
    client.system_one(
        model="jev-latest",
        state="test state",
        questions={"test": {"type": "noul", "instructions": "test instructions"}},
    )
    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes["llm.request.model_name"] == "jev-latest"
    assert span.attributes["llm.model_name"] == "jev-1.13.0"


def test_contract_kind_model_tokens(memory_exporter):
    # TraceRoot ingest classifies and prices a span from these attributes. If upstream
    # moves Jev to another span kind or renames them, this fails instead of costs going silent.
    client = create_client(
        json_body={
            "model": "jev-1.13.0",
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "answers": {"test": {"type": "noul", "noul": 0.5}},
        }
    )
    client.system_one(
        state="test state",
        questions={"test": {"type": "noul", "instructions": "test instructions"}},
    )
    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes.get("openinference.span.kind") == "LLM"
    assert span.attributes["llm.model_name"] == "jev-1.13.0"
    assert span.attributes["llm.token_count.prompt"] == 100
    assert span.attributes["llm.token_count.completion"] == 20


def test_error_status_on_400(memory_exporter):
    client = create_client(status_code=400)
    with pytest.raises(TypeSafeBadRequestError):
        client.system_one(
            state="test state",
            questions={"test": {"type": "noul", "instructions": "test instructions"}},
        )
    span = memory_exporter.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    assert "TypeSafeBadRequestError" in span.status.description


@mock.patch.dict(os.environ, {"OPENINFERENCE_HIDE_INPUTS": "true"})
def test_hide_inputs(memory_exporter):
    TypeSafeAIInstrumentor().uninstrument()
    TypeSafeAIInstrumentor().instrument()
    client = create_client(
        json_body={
            "model": "jev-1.13.0",
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "answers": {"test": {"type": "noul", "noul": 0.5}},
        }
    )
    client.system_one(
        state="test state secret",
        questions={"test": {"type": "noul", "instructions": "test instructions"}},
    )
    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes.get("input.value") == "__REDACTED__"
    assert span.attributes["llm.model_name"] == "jev-1.13.0"
    assert span.attributes["llm.token_count.prompt"] == 100

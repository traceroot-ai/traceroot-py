"""Opt-in live check of a fresh TypeSafe install through TraceRoot ingestion.

Run with the fresh environment's Python, outside the source checkout:
    python verify_typesafe_e2e.py --env-file /private/keys.env \
        --git-ref <tested-commit> --output /tmp/typesafe-proof.json

Makes one real Jev call. Reads the resulting trace back with the project API key.
The report contains versions, trace IDs, model, tokens and cost, never credentials.
"""

import argparse
import importlib.metadata
import json
import math
import os
import platform
import shlex
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4


def read_config(path):
    keys = ("TYPESAFE_API_KEY", "TRACEROOT_API_KEY", "TRACEROOT_HOST_URL")
    config = {key: os.environ[key] for key in keys if os.environ.get(key)}
    if path:
        for line in Path(path).read_text().splitlines():
            key, separator, value = line.strip().removeprefix("export ").partition("=")
            key = key.strip()
            if separator and key in keys and key not in config:
                parts = shlex.split(value, comments=True)
                if len(parts) == 1:
                    config[key] = parts[0]
    for key in keys[:2]:
        if not config.get(key):
            raise RuntimeError(f"Missing {key}")
    return config


def get_json(host, api_key, path):
    request = Request(host + path, headers={"Authorization": f"Bearer {api_key}"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def verify(args):
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from typesafe_sdk import TypeSafeClient

    import traceroot
    from traceroot.instrumentation.registry import Integration

    config = read_config(args.env_file)
    host = config.get("TRACEROOT_HOST_URL", "https://app.traceroot.ai").rstrip("/")
    api_key = config["TRACEROOT_API_KEY"]
    # Validate the TraceRoot key before spending a Jev API call.
    identity = get_json(host, api_key, "/api/v1/public/whoami")
    session_id = f"pr192-e2e-{uuid4().hex}"
    exporter = InMemorySpanExporter()
    traceroot.initialize(
        api_key=api_key,
        host_url=host,
        integrations=[Integration.TYPESAFE],
        git_repo="traceroot-ai/traceroot-py",
        git_ref=args.git_ref,
    )
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
    try:
        with (
            TypeSafeClient(api_key=config["TYPESAFE_API_KEY"], timeout=30) as client,
            traceroot.using_attributes(session_id=session_id),
        ):
            result = client.system_one(
                model="jev-latest",
                state="The parcel arrived on time and its contents were intact.",
                questions={
                    "intact": {
                        "type": "noul",
                        "instructions": "How likely is it that the parcel arrived intact?",
                    }
                },
            )
        spans = exporter.get_finished_spans()
        assert len(spans) == 1, f"Expected one span, got {len(spans)}"
        span = spans[0]
        attributes = span.attributes
        assert attributes["openinference.span.kind"] == "LLM"
        assert attributes["llm.model_name"] == result.model
        assert attributes["llm.request.model_name"] == "jev-latest"
        assert attributes["llm.token_count.prompt"] == result.usage.input_tokens
        assert attributes["llm.token_count.completion"] == result.usage.output_tokens
        assert attributes["session.id"] == session_id
        trace_id = f"{span.context.trace_id:032x}"
        span_id = f"{span.context.span_id:016x}"
        traceroot.flush()
    finally:
        traceroot.shutdown()

    # Ingestion is asynchronous. Read only this new trace, never unrelated traces.
    deadline = time.monotonic() + 120
    while True:
        try:
            stored = get_json(host, api_key, f"/api/v1/public/traces/{trace_id}")
            break
        except HTTPError as error:
            if error.code != 404 or time.monotonic() >= deadline:
                raise
            time.sleep(3)
    matching = [item for item in stored["spans"] if item["span_id"] == span_id]
    assert len(matching) == 1, "Expected the emitted span in the stored trace"
    ingested = matching[0]
    assert ingested["span_kind"].upper() == "LLM"
    assert ingested["model_name"] == result.model
    assert ingested["input_tokens"] == result.usage.input_tokens
    assert ingested["output_tokens"] == result.usage.output_tokens
    assert ingested["total_tokens"] == result.usage.input_tokens + result.usage.output_tokens
    assert ingested["cost"] is not None and math.isfinite(ingested["cost"])
    assert ingested["cost"] > 0, "TraceRoot must store a nonzero calculated cost"
    assert stored["session_id"] == session_id
    assert stored["project_id"] == identity["project_id"]
    report = {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "tested_git_ref": args.git_ref,
        "python": platform.python_version(),
        "installed_sdk_path": traceroot.__file__,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("traceroot", "typesafe-sdk", "openinference-instrumentation-typesafe")
        },
        "project_id": stored["project_id"],
        "trace_url": stored["trace_url"],
        "trace_id": trace_id,
        "session_id": session_id,
        "requested_model": "jev-latest",
        "resolved_model": result.model,
        "api_usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
        },
        "ingested_span": {
            key: ingested[key]
            for key in (
                "span_id",
                "span_kind",
                "status",
                "model_name",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "usage_details",
                "cost",
                "cost_details",
            )
        },
        "checks": "SDK model/tokens match the real response and stored trace; cost is nonzero",
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file")
    parser.add_argument("--git-ref", required=True)
    parser.add_argument("--output", required=True)
    verify(parser.parse_args())

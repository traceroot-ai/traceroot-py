"""Real API + local TraceRoot integration suite; no mocked network responses."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import re
import runpy
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import urlopen
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_json, read_config

SCENARIOS = {
    "quickstart_sync": 1,
    "quickstart_async": 1,
    "search": 4,
    "structure": 2,
    "extraction": 9,
}
FIXTURE_HASHES = {
    "github-tos.txt": "8bd60793d355b8fc280c3197d48e8222db600d60360f0d0b49a7c948ee0fea0d",
    "build-memo.txt": "bd4fb88e60b326cb5713e425c6f08f8381bed03b4f486b52cfc27e9a625262ea",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def normalized_output(text):
    # The structure cookbook prints elapsed seconds; API answers remain exact.
    return re.sub(r"\b\d+(?:\.\d+)?s\b", "<elapsed>", text)


class Recorder:
    """Delegates to the real instrumented SDK, then records its unmodified response."""

    def __init__(self, client, exporter, records):
        self.client, self.exporter, self.records = client, exporter, records

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()

    def record(self, request, response, before):
        spans = self.exporter.get_finished_spans()[before:]
        require(len(spans) == 1, f"Expected exactly one span per call; got {len(spans)}")
        span = spans[0]
        attrs = dict(span.attributes)
        require(attrs.get("openinference.span.kind") == "DECISION", "Wrong span kind")
        require(attrs.get("decision.model_name") == response.model, "Resolved model mismatch")
        require(
            attrs.get("decision.token_count.input") == response.usage.input_tokens,
            "Input tokens mismatch",
        )
        require(
            attrs.get("decision.token_count.output") == response.usage.output_tokens,
            "Output tokens mismatch",
        )
        require(span.status.status_code.name == "OK", "Successful call has an error span")
        input_value, output_value = (
            json.loads(attrs["input.value"]),
            json.loads(attrs["output.value"]),
        )
        require(input_value["state"] == request["state"], "Emitted input changed or truncated")
        expected_questions = {
            name: question.model_dump(mode="json") if hasattr(question, "model_dump") else question
            for name, question in request["questions"].items()
        }
        require(input_value["questions"] == expected_questions, "Question content changed")
        require(
            output_value["answers"] == response.model_dump(mode="json")["answers"],
            "Emitted answers changed",
        )
        require(output_value["model"] == response.model, "Emitted response model changed")
        # Validate the SDK result against the actual HTTP response, independently of tracing.
        wire = response.raw_http_response.json()
        require(wire["model"] == response.model, "SDK changed the API model")
        require(
            wire["usage"]["input_tokens"] == response.usage.input_tokens, "SDK changed input usage"
        )
        require(
            wire["usage"]["output_tokens"] == response.usage.output_tokens,
            "SDK changed output usage",
        )
        # Score maps have integer keys in Python but string keys in JSON; JSON mode normalizes them.
        require(
            wire["answers"] == response.model_dump(mode="json")["answers"],
            "SDK changed API answers",
        )
        self.records.append(
            {
                "trace_id": f"{span.context.trace_id:032x}",
                "span_id": f"{span.context.span_id:016x}",
                "response": response,
                "input": input_value,
                "output": output_value,
                "attributes": attrs,
            }
        )
        return response

    def system_one(self, **kwargs):
        before = len(self.exporter.get_finished_spans())
        return self.record(kwargs, self.client.system_one(**kwargs), before)


class AsyncRecorder(Recorder):
    async def system_one(self, **kwargs):
        before = len(self.exporter.get_finished_spans())
        response = await self.client.system_one(**kwargs)
        return self.record(kwargs, response, before)


class Replay:
    """Logic-only replay of real results to check the example's output, without API calls."""

    def __init__(self, responses):
        self.responses = responses

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def system_one(self, **kwargs):
        return next(self.responses)


class AsyncReplay(Replay):
    async def system_one(self, **kwargs):
        return next(self.responses)


def run(args):
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient

    import traceroot
    from traceroot.instrumentation.registry import Integration

    config = read_config(args.env_file)
    host = args.host_url.rstrip("/")
    require(
        urlparse(host).hostname in {"localhost", "127.0.0.1", "::1"},
        "This suite targets local make dev only",
    )
    key = config["TRACEROOT_API_KEY"]
    identity = get_json(host, key, "/api/v1/public/whoami")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixtures = output / "fixtures"
    fixtures.mkdir(exist_ok=True)

    def load_fixture(url):
        filename = url.rsplit("/", 1)[-1]
        expected = FIXTURE_HASHES[filename]
        target = fixtures / filename
        if not target.exists():
            with urlopen(url, timeout=60) as response:
                target.write_bytes(response.read())
        data = target.read_bytes()
        require(hashlib.sha256(data).hexdigest() == expected, f"Changed fixture {filename}")
        return data.decode()

    if not args.verify_only:
        traceroot.initialize(
            api_key=key,
            host_url=host,
            integrations=[Integration.TYPESAFE],
            git_repo="traceroot-ai/traceroot-py",
            git_ref=args.git_ref,
        )
        exporter = InMemorySpanExporter()
        trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
        records, scenarios = [], []
        session_id = f"pr192-decision-{uuid4().hex}"
        try:
            for name, expected_calls in SCENARIOS.items():
                start = len(records)
                clients = []

                def make_client(clients=clients):
                    client = TypeSafeClient(
                        api_key=config["TYPESAFE_API_KEY"], model=args.model, timeout=120
                    )
                    clients.append(client)
                    return Recorder(client, exporter, records)

                def make_async_client():
                    return AsyncRecorder(
                        AsyncTypeSafeClient(
                            api_key=config["TYPESAFE_API_KEY"], model=args.model, timeout=120
                        ),
                        exporter,
                        records,
                    )

                scenario_path = Path(__file__).resolve().parent / "scenarios" / f"{name}.py"
                scenario = runpy.run_path(str(scenario_path))["run"]
                stdout = io.StringIO()
                print(f"Running {name}: expecting {expected_calls} real calls", flush=True)
                try:
                    with (
                        contextlib.redirect_stdout(stdout),
                        traceroot.using_attributes(
                            session_id=session_id, user_id="pr192-local-proof"
                        ),
                    ):
                        scenario(make_client, make_async_client, load_fixture)
                finally:
                    for client in clients:
                        client.close()
                case_records = records[start:]
                require(len(case_records) == expected_calls, f"Unexpected call count in {name}")
                replay_results = iter(item["response"] for item in case_records)
                replay_stdout = io.StringIO()
                with contextlib.redirect_stdout(replay_stdout):
                    scenario(
                        lambda responses=replay_results: Replay(responses),
                        lambda responses=replay_results: AsyncReplay(responses),
                        load_fixture,
                    )
                require(next(replay_results, None) is None, "Replay did not consume every response")
                require(
                    normalized_output(stdout.getvalue())
                    == normalized_output(replay_stdout.getvalue()),
                    "Tracing altered example output",
                )
                (output / f"{name}.txt").write_text(stdout.getvalue())
                for record in case_records:
                    record["scenario"] = name
                scenarios.append(
                    {
                        "name": name,
                        "calls": expected_calls,
                        "logic_replay_matches": True,
                        "output_file": f"{name}.txt",
                    }
                )
            checkpoint = {
                "sdk_commit": args.git_ref,
                "platform_commit": args.platform_ref,
                "session_id": session_id,
                "scenarios": scenarios,
                "records": [
                    {**item, "response": item["response"].model_dump(mode="json")}
                    for item in records
                ],
            }
            (output / "observations.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
            traceroot.flush()
        finally:
            traceroot.shutdown()
    else:
        from typesafe_sdk._core.response_types import SystemOneResponse

        checkpoint = json.loads((output / "observations.json").read_text())
        require(checkpoint["sdk_commit"] == args.git_ref, "Checkpoint SDK commit differs")
        require(checkpoint["platform_commit"] == args.platform_ref, "Checkpoint platform differs")
        records = [
            {**item, "response": SystemOneResponse.model_validate(item["response"])}
            for item in checkpoint["records"]
        ]
        session_id, scenarios = checkpoint["session_id"], checkpoint["scenarios"]

    results = []
    for record in records:
        deadline = time.monotonic() + 120
        while True:
            try:
                detail = get_json(
                    host, key, f"/api/v1/public/traces/{record['trace_id']}?fields=full"
                )
                break
            except HTTPError as error:
                if error.code != 404 or time.monotonic() >= deadline:
                    raise
                time.sleep(2)
        require(len(detail["spans"]) == 1, "Duplicate/missing ingested spans")
        stored = detail["spans"][0]
        response = record["response"]
        require(stored["span_id"] == record["span_id"], "Wrong stored span")
        require(stored["span_kind"] == "DECISION", "Stored kind is not DECISION")
        require(stored["status"] == "OK", "Wrong stored status")
        require(stored["model_name"] == response.model, "Stored resolved model mismatch")
        require(
            stored["input_tokens"] == response.usage.input_tokens, "Stored input usage mismatch"
        )
        require(
            stored["output_tokens"] == response.usage.output_tokens, "Stored output usage mismatch"
        )
        require(
            stored["total_tokens"] == response.usage.input_tokens + response.usage.output_tokens,
            "Stored total usage mismatch",
        )
        require(
            math.isclose(
                stored["cost"],
                response.usage.input_tokens * args.input_price_per_million / 1_000_000,
                rel_tol=1e-8,
                abs_tol=1e-12,
            ),
            "Stored cost disagrees with Jev input-only pricing",
        )
        require(stored["cost"] > 0, "Missing cost")
        require(json.loads(stored["input"]) == record["input"], "Stored input changed/truncated")
        require(json.loads(stored["output"]) == record["output"], "Stored output changed/truncated")
        require(detail["session_id"] == session_id, "Session propagation failed")
        require(detail["user_id"] == "pr192-local-proof", "User propagation failed")
        metadata = json.loads(stored["metadata"] or "{}")
        require(
            metadata.get("decision.request.model_name") == args.model,
            "Requested alias missing from metadata",
        )
        results.append(
            {
                "scenario": record["scenario"],
                "trace_id": record["trace_id"],
                "trace_url": detail["trace_url"],
                "span": {
                    k: stored[k]
                    for k in (
                        "span_id",
                        "span_kind",
                        "status",
                        "model_name",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                        "cost",
                    )
                },
                "input_characters": len(stored["input"]),
                "output_characters": len(stored["output"]),
                "answers": response.model_dump(mode="json")["answers"],
                "input_sha256": hashlib.sha256(stored["input"].encode()).hexdigest(),
                "output_sha256": hashlib.sha256(stored["output"].encode()).hexdigest(),
            }
        )
    report = {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "sdk_commit": args.git_ref,
        "platform_commit": args.platform_ref,
        "python": platform.python_version(),
        "sdk_import_path": traceroot.__file__,
        "host": host,
        "project_id": identity["project_id"],
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("traceroot", "typesafe-sdk", "openinference-instrumentation-typesafe")
        },
        "requested_model": args.model,
        "input_price_per_million": args.input_price_per_million,
        "scenarios": scenarios,
        "real_calls": len(records),
        "stored_decision_spans": len(results),
        "checks": [
            "one span per call",
            "real API response unchanged",
            "exact input and answers",
            "resolved model",
            "input/output/total tokens",
            "calculated cost",
            "alias metadata",
            "session/user context",
            "example output replay",
        ],
        "calls": results,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(scenarios)} scenarios, {len(results)} real calls and priced DECISION spans")
    print(f"Evidence: {output / 'report.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--host-url", default="http://localhost:8000")
    parser.add_argument("--git-ref", required=True)
    parser.add_argument("--platform-ref", required=True)
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--input-price-per-million", type=float, default=0.042)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Recheck stored traces from observations.json without another Jev call",
    )
    run(parser.parse_args())

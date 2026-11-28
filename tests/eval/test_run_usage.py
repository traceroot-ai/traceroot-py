"""LLM usage per case and per run: bucket math, leaf-only span counting, task/scorer attribution,
artifacts and events, client wiring, and the runner end to end (local and capture-off)."""

import asyncio
import contextlib
import contextvars
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import traceroot
from tests.utils import reset_traceroot
from traceroot.client import TracerootClient
from traceroot.eval import Dataset, EvalRunResult, Evaluation, FakeTransport, evaluate, usage
from traceroot.eval.results import EvalItemResult, RunDatasetRef, UploadState
from traceroot.eval.runner import (
    Emitter,
    _CancelledError,
    _case_metadata,
    _run_one,
    write_artifacts,
)
from traceroot.instrumentation import Integration
from traceroot.transport.span_processor import (
    LocalEvalSampler,
    TracerootSpanProcessor,
    allow_local_capture,
    mark_local_eval_run,
)

KIND = "openinference.span.kind"
PROMPT = "llm.token_count.prompt"
COMPLETION = "llm.token_count.completion"


def _bucket(**fields):
    return {**usage.empty_bucket(), **fields}


def _call(**attrs):
    return usage.read_call(attrs)


def _covered(calls, **sums):
    """Each given sum plus its coverage count of ``calls``."""
    return {**sums, **{f"llm_calls_with_{field}": calls for field in sums}}


def _assert_coverage_invariant(bucket):
    for field in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "cost",
    ):
        covered = bucket[f"llm_calls_with_{field}"]
        assert (covered == 0) == (bucket[field] is None), field
        assert 0 <= covered <= bucket["llm_calls"], field


@pytest.fixture
def provider():
    """A private provider feeding the usage collector (never the global one)."""
    p = TracerProvider()
    p.add_span_processor(usage.collector())
    yield p
    p.shutdown()


@pytest.fixture
def tracer(provider):
    return provider.get_tracer("test")


def _llm(tracer, name="llm", **attrs):
    return tracer.start_as_current_span(name, attributes={KIND: "LLM", **attrs})


# =============================================================================
# Reading a call and the bucket math
# =============================================================================
class TestReadCall:
    def test_first_convention_wins_and_is_never_summed(self):
        call = _call(
            **{PROMPT: 10, "gen_ai.usage.input_tokens": 99, "gen_ai.usage.output_tokens": 4}
        )
        assert call["input_tokens"] == 10
        assert call["output_tokens"] == 4

    def test_invalid_value_counts_as_absent_and_falls_through(self):
        for bad in (-1, float("nan"), float("inf"), "12", True, None):
            assert _call(**{PROMPT: bad, "gen_ai.usage.input_tokens": 5})["input_tokens"] == 5
            assert _call(**{PROMPT: bad})["input_tokens"] is None

    def test_float_token_count_is_rounded_half_up(self):
        assert _call(**{PROMPT: 2.5})["input_tokens"] == 3
        assert _call(**{PROMPT: 2.4})["input_tokens"] == 2

    def test_total_falls_back_to_input_plus_output_only_when_both_present(self):
        assert _call(**{PROMPT: 3, COMPLETION: 4})["total_tokens"] == 7
        assert _call(**{PROMPT: 3})["total_tokens"] is None
        assert _call(**{PROMPT: 3, COMPLETION: 4, "llm.token_count.total": 9})["total_tokens"] == 9

    def test_cache_and_reasoning_fields(self):
        call = _call(
            **{
                "gen_ai.usage.cache_read_input_tokens": 6,
                "llm.token_count.prompt_details.cache_creation": 2,
                "gen_ai.usage.cache_creation_input_tokens": 50,
                "llm.token_count.completion_details.reasoning": 8,
            }
        )
        assert (call["cache_read_tokens"], call["cache_write_tokens"]) == (6, 2)
        assert call["reasoning_tokens"] == 8
        write_first = _call(
            **{
                "llm.token_count.prompt_details.cache_write": 1,
                "llm.token_count.prompt_details.cache_creation": 2,
            }
        )
        assert write_first["cache_write_tokens"] == 1

    def test_cost_must_be_a_finite_non_negative_number(self):
        assert _call(**{"llm.cost.total": 0.25})["cost"] == 0.25
        assert _call(**{"llm.cost.total": 2})["cost"] == 2.0
        for bad in (-0.1, float("nan"), "0.2", True):
            assert _call(**{"llm.cost.total": bad})["cost"] is None


class TestBuckets:
    def test_empty_bucket_has_the_shared_shape(self):
        assert list(usage.empty_bucket()) == [
            "llm_calls",
            "input_tokens",
            "llm_calls_with_input_tokens",
            "output_tokens",
            "llm_calls_with_output_tokens",
            "total_tokens",
            "llm_calls_with_total_tokens",
            "cache_read_tokens",
            "llm_calls_with_cache_read_tokens",
            "cache_write_tokens",
            "llm_calls_with_cache_write_tokens",
            "reasoning_tokens",
            "llm_calls_with_reasoning_tokens",
            "cost",
            "llm_calls_with_cost",
            "currency",
        ]

    def test_token_sums_cover_only_reporting_calls_and_stay_null_when_none_did(self):
        bucket = usage.bucket_of(
            [
                _call(
                    **{PROMPT: 10, COMPLETION: 2, "llm.token_count.prompt_details.cache_read": 0}
                ),
                _call(),  # an LLM call that reported nothing
                _call(**{COMPLETION: 3}),
            ]
        )
        assert bucket == _bucket(
            llm_calls=3,
            input_tokens=10,
            llm_calls_with_input_tokens=1,
            output_tokens=5,
            llm_calls_with_output_tokens=2,
            total_tokens=12,  # only the first call reported both input and output
            llm_calls_with_total_tokens=1,
            cache_read_tokens=0,  # a reported 0 is a value, not an absence
            llm_calls_with_cache_read_tokens=1,
        )
        _assert_coverage_invariant(bucket)

    def test_each_sum_counts_the_calls_that_reported_it(self):
        # Call A reports only its input; call B reports input, output and total. The input sum
        # covers both calls, the output and total sums only B: lower bounds, not complete figures.
        bucket = usage.bucket_of(
            [
                _call(**{PROMPT: 100}),
                _call(**{PROMPT: 20, COMPLETION: 8, "llm.token_count.total": 28}),
            ]
        )
        assert bucket == {
            "llm_calls": 2,
            "input_tokens": 120,
            "llm_calls_with_input_tokens": 2,
            "output_tokens": 8,
            "llm_calls_with_output_tokens": 1,
            "total_tokens": 28,
            "llm_calls_with_total_tokens": 1,
            "cache_read_tokens": None,
            "llm_calls_with_cache_read_tokens": 0,
            "cache_write_tokens": None,
            "llm_calls_with_cache_write_tokens": 0,
            "reasoning_tokens": None,
            "llm_calls_with_reasoning_tokens": 0,
            "cost": None,
            "llm_calls_with_cost": 0,
            "currency": None,
        }

    def test_detail_fields_are_counted_per_field(self):
        bucket = usage.bucket_of(
            [
                _call(
                    **{
                        "llm.token_count.prompt_details.cache_read": 30,
                        "llm.token_count.completion_details.reasoning": 12,
                    }
                ),
                _call(**{"llm.token_count.prompt_details.cache_write": 5}),
                _call(**{"llm.token_count.prompt_details.cache_read": 6}),
            ]
        )
        assert bucket == _bucket(
            llm_calls=3,
            cache_read_tokens=36,
            llm_calls_with_cache_read_tokens=2,
            cache_write_tokens=5,
            llm_calls_with_cache_write_tokens=1,
            reasoning_tokens=12,
            llm_calls_with_reasoning_tokens=1,
        )
        _assert_coverage_invariant(bucket)

    def test_merging_adds_the_coverage_counts(self):
        a = usage.bucket_of([_call(**{PROMPT: 100}), _call(**{"llm.cost.total": 0.2})])
        b = usage.bucket_of([_call(**{PROMPT: 20, COMPLETION: 8})])
        merged = usage.merge_buckets(a, b)
        assert merged == _bucket(
            llm_calls=3,
            input_tokens=120,
            llm_calls_with_input_tokens=2,
            output_tokens=8,
            llm_calls_with_output_tokens=1,
            total_tokens=28,
            llm_calls_with_total_tokens=1,
            cost=0.2,
            llm_calls_with_cost=1,
            currency="USD",
        )
        _assert_coverage_invariant(merged)
        run = usage.sum_usage(
            [
                {"task": a, "scorers": usage.empty_bucket()},
                {"task": b, "scorers": b},
            ]
        )
        assert run["task"] == merged
        assert run["scorers"] == b

    def test_cost_is_unknown_not_zero_unless_reported(self):
        assert usage.bucket_of([_call(**{PROMPT: 1})])["cost"] is None
        bucket = usage.bucket_of(
            [_call(**{"llm.cost.total": 0.5}), _call(), _call(**{"llm.cost.total": 0.25})]
        )
        assert (bucket["cost"], bucket["llm_calls_with_cost"], bucket["currency"]) == (
            0.75,
            2,
            "USD",
        )

    def test_run_usage_sums_cases_and_keeps_null_only_when_null_everywhere(self):
        a = {
            "task": usage.bucket_of([_call(**{PROMPT: 4, COMPLETION: 1})]),
            "scorers": usage.empty_bucket(),
        }
        b = {
            "task": usage.bucket_of([_call(**{"llm.cost.total": 0.1})]),
            "scorers": usage.bucket_of([_call(**{PROMPT: 7})]),
        }
        total = usage.sum_usage([a, None, b])
        assert total["task"] == _bucket(
            llm_calls=2,
            input_tokens=4,
            llm_calls_with_input_tokens=1,
            output_tokens=1,
            llm_calls_with_output_tokens=1,
            total_tokens=5,
            llm_calls_with_total_tokens=1,
            cost=0.1,
            llm_calls_with_cost=1,
            currency="USD",
        )
        assert total["scorers"] == _bucket(
            llm_calls=1, input_tokens=7, llm_calls_with_input_tokens=1
        )
        assert usage.sum_usage([]) == {"task": _bucket(), "scorers": _bucket()}


# =============================================================================
# The collector: which spans count, and to whom
# =============================================================================
class TestCollector:
    def test_only_the_leaf_llm_span_counts(self, tracer):
        case = usage.CaseUsage("c")
        with usage.scope(case, usage.TASK):
            # An LLM wrapper whose provider call sits under a CHAIN span: one call, not two.
            with (
                _llm(tracer, "wrapper", **{PROMPT: 100, COMPLETION: 100}),
                tracer.start_as_current_span("chain", attributes={KIND: "CHAIN"}),
                _llm(tracer, "provider", **{PROMPT: 12, COMPLETION: 3}),
            ):
                pass
            with _llm(tracer, "second", **{PROMPT: 1, COMPLETION: 1}):
                pass
        assert case.take()["task"] == _bucket(
            llm_calls=2, **_covered(2, input_tokens=13, output_tokens=4, total_tokens=17)
        )

    def test_agent_span_with_aggregate_tokens_is_not_counted(self, tracer):
        case = usage.CaseUsage("c")
        with usage.scope(case, usage.TASK):
            with (
                tracer.start_as_current_span(
                    "agent", attributes={KIND: "AGENT", PROMPT: 500, COMPLETION: 50}
                ),
                _llm(tracer, **{PROMPT: 20, COMPLETION: 5}),
            ):
                pass
            with tracer.start_as_current_span("tool", attributes={KIND: "TOOL", PROMPT: 9}):
                pass
        assert case.take()["task"]["input_tokens"] == 20

    def test_judge_span_typed_llm_counts(self, tracer):
        case = usage.CaseUsage("c")
        with (
            usage.scope(case, usage.SCORER),
            tracer.start_as_current_span("llm_judge:j") as span,
        ):
            span.set_attribute("traceroot.span.type", "llm")
            span.set_attribute(PROMPT, 30)
        assert case.take()["scorers"]["llm_calls"] == 1

    def test_task_and_scorer_calls_land_in_their_own_buckets(self, tracer):
        case = usage.CaseUsage("c")
        with usage.scope(case, usage.TASK), _llm(tracer, **{PROMPT: 5}):
            pass
        with usage.scope(case, usage.SCORER), _llm(tracer, **{PROMPT: 7}):
            pass
        taken = case.take()
        assert taken["task"]["input_tokens"] == 5
        assert taken["scorers"]["input_tokens"] == 7

    def test_span_outside_a_scope_or_ending_after_take_is_not_counted(self, tracer):
        case = usage.CaseUsage("c")
        with _llm(tracer, **{PROMPT: 99}):  # no scope (module import, an uncopied thread)
            pass
        with usage.scope(case, usage.TASK):
            late = tracer.start_span("late", attributes={KIND: "LLM", PROMPT: 4})
        taken = case.take()
        late.end()  # the call outlived its case
        assert taken["task"]["llm_calls"] == 0
        assert case.take()["task"]["llm_calls"] == 0

    async def test_concurrent_cases_do_not_leak_into_each_other(self, tracer):
        cases = {n: usage.CaseUsage(f"c{n}") for n in range(1, 9)}

        async def one(n):
            with usage.scope(cases[n], usage.TASK):
                for _ in range(3):
                    with _llm(tracer, **{PROMPT: n, COMPLETION: 1}):
                        await asyncio.sleep(0.001 * (9 - n))

        def threaded(n):
            with usage.scope(cases[n], usage.SCORER):
                ctx = contextvars.copy_context()
                worker = threading.Thread(target=ctx.run, args=(_thread_call, tracer, n))
                worker.start()
                worker.join()

        await asyncio.gather(*(one(n) for n in cases))
        threads = [threading.Thread(target=threaded, args=(n,)) for n in cases]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for n, case in cases.items():
            taken = case.take()
            assert taken["task"]["llm_calls"] == 3
            assert taken["task"]["input_tokens"] == 3 * n
            assert taken["scorers"]["input_tokens"] == 100 + n


def _thread_call(tracer, n):
    with _llm(tracer, **{PROMPT: 100 + n}):
        time.sleep(0.001)


# =============================================================================
# The local-run guarantees still hold around a usage scope
# =============================================================================
class TestLocalRunGuards:
    def test_lifted_suppression_still_keeps_spans_from_traceroot_exporters(self, provider):
        from opentelemetry.instrumentation.utils import is_instrumentation_enabled

        exported = InMemorySpanExporter()
        exporting = TracerProvider()
        exporting.sampler = LocalEvalSampler(exporting.sampler)
        exporting.add_span_processor(SimpleSpanProcessor(exported))
        joined_host = TracerProvider()
        with patch("traceroot.transport.span_processor.OTLPSpanExporter", return_value=exported):
            traceroot_proc = TracerootSpanProcessor(api_key="k", host_url="http://offline.invalid")
        joined_host.add_span_processor(traceroot_proc)
        case = usage.CaseUsage("c", local=True)
        try:
            with mark_local_eval_run():
                assert not is_instrumentation_enabled()
                with usage.scope(case, usage.TASK):
                    assert is_instrumentation_enabled()  # instrumentors build spans again
                    with _llm(provider.get_tracer("capture"), **{PROMPT: 3}):
                        pass
                    exporting.get_tracer("app").start_span("owned").end()
                    joined_host.get_tracer("app").start_span("joined").end()
                assert not is_instrumentation_enabled()  # restored on the way out
            traceroot_proc.force_flush()
            assert exported.get_finished_spans() == ()
            assert case.take()["task"]["input_tokens"] == 3
        finally:
            traceroot_proc.shutdown()
            exporting.shutdown()
            joined_host.shutdown()

    def test_allow_local_capture_outside_a_local_run_changes_nothing_for_exporters(self):
        exported = InMemorySpanExporter()
        exporting = TracerProvider()
        exporting.sampler = LocalEvalSampler(exporting.sampler)
        exporting.add_span_processor(SimpleSpanProcessor(exported))
        with allow_local_capture():
            exporting.get_tracer("app").start_span("reported").end()
        assert [s.name for s in exported.get_finished_spans()] == ["reported"]
        exporting.shutdown()


# =============================================================================
# The engine records usage per case and per run
# =============================================================================
def _cases(*inputs):
    return [{"input": x, "id": f"c{x}"} for x in inputs]


@pytest.fixture
def capturing(monkeypatch):
    monkeypatch.setattr(usage, "capture_active", lambda local: True)


class TestEngine:
    def test_task_and_scorer_usage_per_case_and_summed_per_run(self, capturing, tracer):
        def task(x):  # sync: runs in the engine's thread pool with the case's context
            with _llm(tracer, **{PROMPT: x, COMPLETION: 1}):
                pass
            return x

        async def judge(ctx):
            with _llm(tracer, **{PROMPT: 100, COMPLETION: 2, "llm.cost.total": 0.5}):
                await asyncio.sleep(0)
            return 1.0

        result = evaluate(name="u", dataset=_cases(1, 2), task=task, scorers=[judge], local=True)
        first = result.item_results[0].usage
        assert first["task"] == _bucket(
            llm_calls=1, **_covered(1, input_tokens=1, output_tokens=1, total_tokens=2)
        )
        assert first["scorers"] == _bucket(
            llm_calls=1,
            **_covered(1, input_tokens=100, output_tokens=2, total_tokens=102, cost=0.5),
            currency="USD",
        )
        assert result.usage["task"]["input_tokens"] == 3
        assert result.usage["scorers"]["cost"] == 1.0
        assert result.usage["scorers"]["llm_calls"] == 2

    def test_a_task_that_raised_keeps_the_calls_it_made(self, capturing, tracer):
        def task(x):
            with _llm(tracer, **{PROMPT: 4}):
                pass
            with _llm(tracer, **{PROMPT: 6}):
                pass
            raise RuntimeError("failed after two calls")

        result = evaluate(
            name="u", dataset=_cases(1), task=task, scorers=[lambda ctx: 1.0], local=True
        )
        item = result.item_results[0]
        assert item.error is not None
        assert item.usage["task"]["llm_calls"] == 2
        assert item.usage["task"]["input_tokens"] == 10
        assert item.usage["scorers"] == usage.empty_bucket()  # scorers never ran

    async def test_concurrent_cases_keep_their_own_usage(self, capturing, tracer):
        from traceroot.eval import evaluate_async

        async def task(x):
            for _ in range(2):
                with _llm(tracer, **{PROMPT: x}):
                    await asyncio.sleep(0.001 * (10 - x))
            return x

        def scorer(ctx):
            with _llm(tracer, **{PROMPT: 1000 + ctx.output}):
                time.sleep(0.001)
            return 1.0

        result = await evaluate_async(
            name="u",
            dataset=_cases(*range(1, 9)),
            task=task,
            scorers=[scorer],
            transport=FakeTransport(),
        )
        for item in result.item_results:
            x = item.input
            assert item.usage["task"]["input_tokens"] == 2 * x
            assert item.usage["scorers"]["input_tokens"] == 1000 + x

    def test_usage_is_null_when_capture_is_not_active(self, tracer):
        def task(x):
            with _llm(tracer, **{PROMPT: 1}):
                pass
            return x

        result = evaluate(
            name="u", dataset=_cases(1), task=task, scorers=[lambda ctx: 1.0], local=True
        )
        assert result.item_results[0].usage is None
        assert result.usage is None


# =============================================================================
# When capture is active
# =============================================================================
class TestCaptureActive:
    @pytest.fixture
    def capture_provider(self, monkeypatch):
        p = TracerProvider()
        usage.attach(p)
        client = SimpleNamespace(
            _instrumented=[Integration.OPENAI], _provider=p, _capture_only=True
        )
        monkeypatch.setattr(traceroot, "_client", client)
        yield p, client
        p.shutdown()

    def test_needs_arming_and_an_instrumented_integration(self, capture_provider):
        _, client = capture_provider
        assert usage.capture_active(local=False) is False  # not armed
        with usage.armed():
            assert usage.capture_active(local=False) is True
            client._instrumented = []
            assert usage.capture_active(local=False) is False

    def test_local_needs_the_non_exporting_provider_to_be_global(self, capture_provider):
        p, client = capture_provider
        with usage.armed():
            with patch("opentelemetry.trace.get_tracer_provider", return_value=p):
                assert usage.capture_active(local=True) is True
                client._capture_only = False  # an exporting client: suppression stays
                assert usage.capture_active(local=True) is False
            client._capture_only = True
            other = TracerProvider()
            with patch("opentelemetry.trace.get_tracer_provider", return_value=other):
                assert usage.capture_active(local=True) is False  # someone else's provider


# =============================================================================
# The client wires the collector
# =============================================================================
class TestClientWiring:
    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_traceroot()
        yield
        reset_traceroot()

    def test_disabled_armed_client_runs_a_provider_with_only_the_collector(self, monkeypatch):
        monkeypatch.setattr(
            "traceroot.instrumentation.registry.initialize_integrations",
            lambda tracer_provider, integrations: list(integrations),
        )
        with (
            patch(
                "opentelemetry.trace.get_tracer_provider", return_value=trace.ProxyTracerProvider()
            ),
            patch("opentelemetry.trace.set_tracer_provider") as set_global,
            usage.armed(),
        ):
            client = TracerootClient(
                api_key="",
                enabled=False,
                integrations=[Integration.OPENAI],
                git_repo="o/r",
                git_ref="x",
            )
        assert client.enabled is False
        assert client._capture_only is True
        assert client._span_processor is None
        assert client._instrumented == [Integration.OPENAI]
        assert client._provider._active_span_processor._span_processors == (usage.collector(),)
        set_global.assert_called_once_with(client._provider)

    def test_disabled_client_is_unchanged_without_arming(self):
        client = TracerootClient(
            api_key="",
            enabled=False,
            integrations=[Integration.OPENAI],
            git_repo="o/r",
            git_ref="x",
        )
        assert client._provider is None
        assert client._capture_only is False
        assert client._instrumented == []

    def test_uploaded_run_adds_the_collector_and_exports_the_same_spans(self):
        def exported_attributes(armed):
            host = TracerProvider()
            exporter = InMemorySpanExporter()
            arming = usage.armed() if armed else contextlib.nullcontext()
            with (
                patch("opentelemetry.trace.get_tracer_provider", return_value=host),
                patch("traceroot.transport.span_processor.OTLPSpanExporter", return_value=exporter),
                arming,
            ):
                client = TracerootClient(
                    api_key="k", host_url="http://offline.invalid", git_repo="o/r", git_ref="x"
                )
                processors = host._active_span_processor._span_processors
                case = usage.CaseUsage("c")
                with usage.scope(case, usage.TASK), _llm(host.get_tracer("app"), **{PROMPT: 8}):
                    pass
                client.flush()
            client.shutdown()
            host.shutdown()
            spans = exporter.get_finished_spans()
            assert len(spans) == 1
            return processors, spans[0].attributes, case.take()

        processors, attrs, counted = exported_attributes(armed=True)
        assert isinstance(processors[0], TracerootSpanProcessor)
        assert processors[1:] == (usage.collector(),)
        assert counted["task"]["input_tokens"] == 8
        plain_processors, plain_attrs, not_counted = exported_attributes(armed=False)
        assert len(plain_processors) == 1
        assert not_counted["task"]["llm_calls"] == 0
        assert dict(attrs) == dict(plain_attrs)  # the collector changes nothing that is exported


# =============================================================================
# Events and artifacts
# =============================================================================
def _result(item_usage, run_usage):
    item = EvalItemResult(
        case_id="a",
        input=1,
        output=1,
        expected=1,
        scores=[],
        scorer_errors={},
        error=None,
        trace_id=None,
        duration_ms=1.0,
        usage=item_usage,
    )
    return EvalRunResult(
        name="u",
        item_results=[item],
        score_summary={},
        upload_state=UploadState(),
        local_run_id="run_1",
        dataset=RunDatasetRef(dataset_id="ds", revision="r", dataset_version_id=None, case_count=1),
        usage=run_usage,
    )


class TestCancelledRun:
    def test_run_usage_follows_the_local_setting_the_engine_ran_with(self, tmp_path, monkeypatch):
        # A reported runner run of an Evaluation declared local=True runs the engine locally. Here
        # capture is visible only to an exporting run, so the engine records each case's usage as
        # not captured, and the cancelled run's total must say the same rather than 0 calls.
        monkeypatch.setattr(usage, "capture_active", lambda local: not local)
        calls = {"n": 0}

        async def task(x):
            calls["n"] += 1
            if calls["n"] == 2:  # Ctrl+C arrives after the first case finished
                raise KeyboardInterrupt()
            return x

        ds = Dataset("u")
        for i in range(3):
            ds.add(input=i, id=f"c{i}", expected=i)
        evaluation = Evaluation(
            name="u", dataset=ds, task=task, scorers=[lambda ctx: 1.0], local=True
        )
        events = []
        with pytest.raises(_CancelledError):
            _run_one(
                evaluation,
                {"reporting": True, "out_dir": str(tmp_path), "max_concurrency": 1},
                Emitter(lambda line: events.append(json.loads(line))),
            )

        done = next(e for e in events if e["type"] == "evaluation_completed")
        assert done["status"] == "incomplete"
        assert [e["usage"] for e in events if e["type"] == "case_completed"] == [None]
        assert done["usage"] is None
        assert json.loads(Path(done["artifact"]["run"]).read_text())["usage"] is None


class TestArtifacts:
    def test_case_line_case_metadata_and_run_doc_carry_usage(self, tmp_path):
        case_usage = {"task": usage.bucket_of([_call(**{PROMPT: 2})]), "scorers": _bucket()}
        result = _result(case_usage, usage.sum_usage([case_usage]))
        run_path, cases_path = tmp_path / "run.json", tmp_path / "run.cases.jsonl"
        write_artifacts(
            result,
            run_path,
            cases_path,
            status="completed",
            run_mode="full",
            is_final=True,
            sample_count=None,
            sample_seed=None,
            candidate_version=None,
        )
        run_doc = json.loads(run_path.read_text())
        line = json.loads(cases_path.read_text().splitlines()[0])
        assert line["usage"] == case_usage
        assert run_doc["cases"][0]["usage"] == case_usage
        assert run_doc["usage"] == usage.sum_usage([case_usage])
        assert _case_metadata(result.item_results[0], _policy())["usage"] == case_usage

    def test_capture_off_writes_null_usage(self, tmp_path):
        run_path, cases_path = tmp_path / "run.json", tmp_path / "run.cases.jsonl"
        write_artifacts(
            _result(None, None),
            run_path,
            cases_path,
            status="completed",
            run_mode="full",
            is_final=True,
            sample_count=None,
            sample_seed=None,
            candidate_version=None,
        )
        assert json.loads(run_path.read_text())["usage"] is None
        assert json.loads(cases_path.read_text())["usage"] is None

    def test_an_artifact_written_before_usage_existed_reads_as_none(self):
        old = {
            "kind": "eval_run",
            "evaluation_name": "u",
            "local_run_id": "run_1",
            "cases": [{"case_id": "a", "scores": [], "scorer_errors": []}],
            "scores": {},
            "upload": {},
        }
        loaded = EvalRunResult.from_dict(old)
        assert loaded.usage is None
        assert loaded.item_results[0].usage is None
        saved_shape = _result(None, None).to_dict()
        del saved_shape["usage"]
        for item in saved_shape["item_results"]:
            del item["usage"]
        assert EvalRunResult.from_dict(saved_shape).item_results[0].usage is None

    def test_saved_result_round_trips_usage(self):
        case_usage = {"task": _bucket(llm_calls=1), "scorers": _bucket()}
        result = _result(case_usage, usage.sum_usage([case_usage]))
        loaded = EvalRunResult.from_dict(json.loads(json.dumps(result.to_dict())))
        assert loaded.usage == result.usage
        assert loaded.item_results[0].usage == case_usage


def _policy():
    from traceroot.eval.runner import ScorePolicy

    return ScorePolicy(None)


# =============================================================================
# The runner end to end, in a child process (instrumentation and the global tracer provider are
# process-wide), against in-process fake servers on 127.0.0.1
# =============================================================================
class _Server:
    """A loopback HTTP server that records every request and answers with ``respond``."""

    def __init__(self, respond):
        self.requests = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self):
                body = self.rfile.read(int(self.headers.get("content-length") or 0))
                server.requests.append((self.command, self.path))
                status, payload = respond(self.path, body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            # http.server dispatches on these names.
            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_CONNECT = _handle  # noqa: N815

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


JUDGE_MODEL = "claude-judge-test"
JUDGE_USAGE = (40, 1, 0, 0)


def _messages_api(path, body):
    """A fake Anthropic Messages API. A task's prompt "usage I O R W" picks the usage the reply
    reports (input, output, cache read, cache write); the judge model always reports JUDGE_USAGE."""
    request = json.loads(body)
    if request["model"] == JUDGE_MODEL:
        text, counts = "1", JUDGE_USAGE
    else:
        text = "ok"
        counts = tuple(int(n) for n in request["messages"][-1]["content"].split()[1:])
    time.sleep(0.05)  # keep concurrent cases' calls overlapping
    inp, out, cache_read, cache_write = counts
    return 200, {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": request["model"],
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": inp,
            "output_tokens": out,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_write,
        },
    }


EVAL_MODULE = """
import anthropic

import traceroot
from traceroot import Dataset, Evaluation
from traceroot.eval import llm_judge

traceroot.initialize(
    api_key="tr-test-key",
    host_url={traceroot_url!r},
    git_repo="test/repo",
    git_ref="test",
    integrations=[traceroot.Integration.ANTHROPIC],
    {extra}
)


def task(x):
    client = anthropic.Anthropic(api_key="test-key", base_url={llm_url!r})
    for prompt in x["calls"]:
        client.messages.create(
            model="claude-task-test",
            max_tokens=16,
            messages=[{{"role": "user", "content": prompt}}],
        )
    if x.get("fail"):
        raise RuntimeError("task failed after its calls")
    return "done"


judge = llm_judge(
    name="judge",
    model={judge_model!r},
    messages=[{{"role": "user", "content": "Score {{{{output}}}} from 0 to 1."}}],
)

ds = Dataset("usage")
ds.add(input={{"calls": ["usage 11 7 0 0"]}}, id="a")
ds.add(input={{"calls": ["usage 5 3 2 1", "usage 13 2 0 0"]}}, id="b")
ds.add(input={{"calls": ["usage 17 4 0 0"], "fail": True}}, id="c")

usage_eval = Evaluation(name="usage", dataset=ds, task=task, scorers=[judge])
"""


def _run_runner(tmp_path, extra=""):
    llm = _Server(_messages_api)
    sink = _Server(lambda path, body: (200, {}))  # traceroot, and any host but loopback
    try:
        evals = tmp_path / "evals"
        evals.mkdir()
        (evals / "usage_eval.py").write_text(
            EVAL_MODULE.format(
                traceroot_url=sink.url, llm_url=llm.url, judge_model=JUDGE_MODEL, extra=extra
            )
        )
        events_path = tmp_path / "events.ndjson"
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("TRACEROOT_", "ANTHROPIC_", "OPENAI_", "OTEL_", "OPENINFERENCE_"))
            and k.lower() not in ("http_proxy", "https_proxy", "all_proxy", "no_proxy")
        }
        env.update(
            {
                "TRACEROOT_EVAL_EVENT_FILE": str(events_path),
                "TRACEROOT_EVAL_OPTIONS": json.dumps({"out_dir": str(tmp_path / "runs")}),
                "TRACEROOT_API_KEY": "tr-test-key",
                "TRACEROOT_HOST_URL": sink.url,
                # The judge's default client reads these.
                "ANTHROPIC_API_KEY": "test-key",
                "ANTHROPIC_BASE_URL": llm.url,
                # Anything bound for a host other than loopback is sent to the sink instead.
                "HTTP_PROXY": sink.url,
                "HTTPS_PROXY": sink.url,
                "ALL_PROXY": sink.url,
                "NO_PROXY": "127.0.0.1,localhost",
            }
        )
        proc = subprocess.run(
            [sys.executable, "-m", "traceroot.eval.runner", str(evals)],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        assert not [e for e in events if e["type"] == "fatal"], events
        done = next(e for e in events if e["type"] == "evaluation_completed")
        run_doc = json.loads(Path(done["artifact"]["run"]).read_text())
        lines = [json.loads(x) for x in Path(done["artifact"]["cases"]).read_text().splitlines()]
        return SimpleNamespace(
            events=events,
            done=done,
            run_doc=run_doc,
            lines={line["case_id"]: line for line in lines},
            llm_requests=list(llm.requests),
            sink_requests=list(sink.requests),
        )
    finally:
        llm.close()
        sink.close()


def _anthropic_bucket(*calls):
    """The bucket for replies reporting (input, output, cache read, cache write). The Anthropic
    integration reports cache reads and writes as part of the prompt count, as the API bills them,
    and leaves a zero cache count off."""
    rows = []
    for inp, out, cache_read, cache_write in calls:
        prompt = inp + cache_read + cache_write
        attrs = {PROMPT: prompt, COMPLETION: out, "llm.token_count.total": prompt + out}
        if cache_read:
            attrs["llm.token_count.prompt_details.cache_read"] = cache_read
        if cache_write:
            attrs["llm.token_count.prompt_details.cache_write"] = cache_write
        rows.append(usage.read_call(attrs))
    return usage.bucket_of(rows)


@pytest.mark.skipif(sys.platform == "win32", reason="loopback server + child process")
class TestRunnerEndToEnd:
    def test_local_run_records_each_cases_usage_and_sends_nothing(self, tmp_path):
        out = _run_runner(tmp_path)

        expected = {
            "a": {
                "task": _anthropic_bucket((11, 7, 0, 0)),
                "scorers": _anthropic_bucket(JUDGE_USAGE),
            },
            "b": {
                "task": _anthropic_bucket((5, 3, 2, 1), (13, 2, 0, 0)),
                "scorers": _anthropic_bucket(JUDGE_USAGE),
            },
            # The task raised after its call: the call still counts; no scorer ran.
            "c": {"task": _anthropic_bucket((17, 4, 0, 0)), "scorers": usage.empty_bucket()},
        }
        assert expected["b"]["task"] == _bucket(
            llm_calls=2,
            **_covered(2, input_tokens=21, output_tokens=5, total_tokens=26),
            **_covered(1, cache_read_tokens=2, cache_write_tokens=1),
        )
        for case_id, case_usage in expected.items():
            assert out.lines[case_id]["usage"] == case_usage, case_id
        run_cases = {c["case_id"]: c for c in out.run_doc["cases"]}
        completed = {e["case_id"]: e for e in out.events if e["type"] == "case_completed"}
        for case_id, case_usage in expected.items():
            assert run_cases[case_id]["usage"] == case_usage
            assert completed[case_id]["usage"] == case_usage
        run_usage = usage.sum_usage(expected.values())
        assert run_usage["task"]["llm_calls"] == 4
        assert run_usage["scorers"]["input_tokens"] == 80
        assert out.run_doc["usage"] == run_usage
        assert out.done["usage"] == run_usage
        for written in [out.run_doc["usage"], *(line["usage"] for line in out.lines.values())]:
            _assert_coverage_invariant(written["task"])
            _assert_coverage_invariant(written["scorers"])

        # 4 task calls + 2 judge calls reached the model server; nothing reached traceroot or any
        # other host.
        assert len(out.llm_requests) == 6
        assert out.sink_requests == []

    def test_local_run_with_an_exporting_client_keeps_suppression_and_reports_null(self, tmp_path):
        # The module forces a reporting client on. The local run keeps tracing suppressed for it,
        # so its integration builds no span: usage is "not captured", and nothing is exported.
        out = _run_runner(tmp_path, extra="enabled=True,")

        assert out.run_doc["usage"] is None
        assert out.done["usage"] is None
        assert all(line["usage"] is None for line in out.lines.values())
        assert len(out.llm_requests) == 6
        assert out.sink_requests == []

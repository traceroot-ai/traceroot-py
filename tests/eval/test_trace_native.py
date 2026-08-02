"""OE-4: trace-native execution - span hierarchy, attributes, isolation."""

import traceroot
from traceroot.constants import SpanKind
from traceroot.eval import Dataset, EvalCase, FakeTransport, PlatformTransport, evaluate
from traceroot.span_attributes import SpanAttributes


def _reported():
    """A transport that reports traces (so eval spans export) without real network.

    Evaluation traces are exported only for REPORTED runs (privacy boundary); these
    structural span tests therefore run in reported mode. See TestLocalTracePrivacy
    for the local-only (no export) contract."""
    return FakeTransport()


def _ds(n, **kw):
    ds = Dataset(name="d")
    for i in range(n):
        ds.upsert(EvalCase(input=i, id=f"c{i}", expected=i, **kw))
    return ds


def echo(x):
    return x


def exact(ctx):
    return 1.0 if ctx.output == ctx.expected else 0.0


def _by_name(spans):
    return {s.name: s for s in spans}


class TestSpanKindExtension:
    def test_new_kinds_added(self):
        assert SpanKind.EVALUATION == "evaluation"
        assert SpanKind.TASK == "task"
        assert SpanKind.SCORER == "scorer"

    def test_existing_kinds_unchanged(self):
        assert SpanKind.SPAN == "span"
        assert SpanKind.AGENT == "agent"
        assert SpanKind.TOOL == "tool"
        assert SpanKind.LLM == "llm"


class TestSpanHierarchy:
    def test_three_span_kinds_per_case(self, memory_exporter):
        evaluate(name="r", data=_ds(1), task=echo, scorers=[exact], report_to=_reported())
        spans = memory_exporter.get_finished_spans()
        names = sorted(s.name for s in spans)
        assert names == ["evaluation-item", "exact", "task"]

    def test_task_parents_under_root_scorer_sibling_of_task(self, memory_exporter):
        evaluate(name="r", data=_ds(1), task=echo, scorers=[exact], report_to=_reported())
        by = _by_name(memory_exporter.get_finished_spans())
        root, task, scorer = by["evaluation-item"], by["task"], by["exact"]
        assert root.parent is None
        assert task.parent.span_id == root.context.span_id
        # scorer is a SIBLING of task (child of root), NOT a child of task
        assert scorer.parent.span_id == root.context.span_id

    def test_user_observe_span_nests_under_task(self, memory_exporter):
        @traceroot.observe(name="inner_llm", type="llm")
        def inner(x):
            return x

        def task_with_inner(x):
            return inner(x)

        evaluate(
            name="r", data=_ds(1), task=task_with_inner, scorers=[exact], report_to=_reported()
        )
        by = _by_name(memory_exporter.get_finished_spans())
        assert "inner_llm" in by
        assert by["inner_llm"].parent.span_id == by["task"].context.span_id

    async def test_user_observe_nests_for_async_task(self, memory_exporter):
        @traceroot.observe(name="inner_async", type="llm")
        async def inner(x):
            return x

        async def atask(x):
            return await inner(x)

        from traceroot.eval import evaluate_async

        await evaluate_async(
            name="r", data=_ds(1), task=atask, scorers=[exact], report_to=_reported()
        )
        by = _by_name(memory_exporter.get_finished_spans())
        assert by["inner_async"].parent.span_id == by["task"].context.span_id


class TestConcurrencyIsolation:
    def test_concurrent_cases_do_not_tangle(self, memory_exporter):
        evaluate(
            name="r",
            data=_ds(5),
            task=echo,
            scorers=[exact],
            max_concurrency=5,
            report_to=_reported(),
        )
        spans = memory_exporter.get_finished_spans()
        # group by trace id
        by_trace = {}
        for s in spans:
            by_trace.setdefault(s.context.trace_id, []).append(s)
        assert len(by_trace) == 5  # one trace per case
        for trace_spans in by_trace.values():
            names = sorted(s.name for s in trace_spans)
            assert names == ["evaluation-item", "exact", "task"]
            by = _by_name(trace_spans)
            root = by["evaluation-item"]
            # every non-root span belongs to this trace's root
            assert by["task"].parent.span_id == root.context.span_id
            assert by["exact"].parent.span_id == root.context.span_id


class TestEvalAttributes:
    def test_root_attributes(self, memory_exporter):
        ds = _ds(1, metadata={"cat": "x"}, source_trace_id="t1", source_span_id="s1")
        evaluate(name="routing-v2", data=ds, task=echo, scorers=[exact], report_to=_reported())
        root = _by_name(memory_exporter.get_finished_spans())["evaluation-item"]
        a = root.attributes
        assert a[SpanAttributes.SPAN_TYPE] == "evaluation"
        assert a["traceroot.eval.run_name"] == "routing-v2"
        assert a["traceroot.eval.dataset_name"] == "d"
        assert a["traceroot.eval.case_id"] == "c0"
        assert a["traceroot.eval.has_expected"] is True
        assert a["traceroot.eval.source_trace_id"] == "t1"
        assert a["traceroot.eval.source_span_id"] == "s1"

    def test_has_expected_false_when_absent(self, memory_exporter):
        ds = Dataset(name="d")
        ds.upsert(EvalCase(input=1, id="c0"))  # no expected
        evaluate(name="r", data=ds, task=echo, scorers=[exact], report_to=_reported())
        root = _by_name(memory_exporter.get_finished_spans())["evaluation-item"]
        assert root.attributes["traceroot.eval.has_expected"] is False

    def test_task_and_scorer_attributes(self, memory_exporter):
        evaluate(name="r", data=_ds(1), task=echo, scorers=[exact], report_to=_reported())
        by = _by_name(memory_exporter.get_finished_spans())
        assert by["task"].attributes[SpanAttributes.SPAN_TYPE] == "task"
        assert by["task"].attributes["traceroot.eval.task_name"] == "echo"
        scorer = by["exact"]
        assert scorer.attributes[SpanAttributes.SPAN_TYPE] == "scorer"
        assert scorer.attributes["traceroot.eval.scorer_name"] == "exact"
        assert scorer.attributes["traceroot.eval.score_value"] == 1.0

    def test_run_name_on_all_spans(self, memory_exporter):
        evaluate(name="routing-v2", data=_ds(1), task=echo, scorers=[exact], report_to=_reported())
        by = _by_name(memory_exporter.get_finished_spans())
        for span in (by["evaluation-item"], by["task"], by["exact"]):
            assert span.attributes["traceroot.eval.run_name"] == "routing-v2"

    def test_task_input_output_captured(self, memory_exporter):
        evaluate(name="r", data=_ds(1), task=echo, scorers=[exact], report_to=_reported())
        task = _by_name(memory_exporter.get_finished_spans())["task"]
        assert SpanAttributes.SPAN_INPUT in task.attributes
        assert SpanAttributes.SPAN_OUTPUT in task.attributes


class TestEvalSpanIO:
    """UI ask: the eval-structural spans (root + scorer) carry standard span.input/output
    so the trace viewer renders them like any other span (and diff mode has content)."""

    def test_root_carries_case_input_and_candidate_output(self, memory_exporter):
        evaluate(name="r", data=_ds(1), task=echo, scorers=[exact], report_to=_reported())
        root = _by_name(memory_exporter.get_finished_spans())["evaluation-item"]
        assert root.attributes[SpanAttributes.SPAN_INPUT] == 0  # the case input
        assert root.attributes[SpanAttributes.SPAN_OUTPUT] == 0  # the candidate output

    def test_root_output_is_task_error_on_failure(self, memory_exporter):
        def boom(x):
            raise ValueError("kaboom")

        evaluate(name="r", data=_ds(1), task=boom, scorers=[exact], report_to=_reported())
        root = _by_name(memory_exporter.get_finished_spans())["evaluation-item"]
        assert root.attributes[SpanAttributes.SPAN_INPUT] == 0
        assert "kaboom" in root.attributes[SpanAttributes.SPAN_OUTPUT]  # task error as output

    def test_scorer_input_has_candidate_and_expected(self, memory_exporter):
        evaluate(name="r", data=_ds(1), task=echo, scorers=[exact], report_to=_reported())
        scorer = _by_name(memory_exporter.get_finished_spans())["exact"]
        inp = scorer.attributes[SpanAttributes.SPAN_INPUT]
        assert "candidate" in inp and "expected" in inp  # what the scorer compared

    def test_scorer_output_has_score_value(self, memory_exporter):
        evaluate(name="r", data=_ds(1), task=echo, scorers=[exact], report_to=_reported())
        scorer = _by_name(memory_exporter.get_finished_spans())["exact"]
        out = scorer.attributes[SpanAttributes.SPAN_OUTPUT]
        assert "value" in out and "1.0" in out  # score value + explanation slot

    def test_scorer_output_is_error_on_scorer_failure(self, memory_exporter):
        def broken(ctx):
            raise RuntimeError("judge down")

        evaluate(name="r", data=_ds(1), task=echo, scorers=[broken], report_to=_reported())
        scorer = _by_name(memory_exporter.get_finished_spans())["broken"]
        assert "judge down" in scorer.attributes[SpanAttributes.SPAN_OUTPUT]

    def test_scorer_input_includes_target_span_id_when_present(self, memory_exporter):
        ds = Dataset(name="d")
        ds.upsert(EvalCase(input=0, id="c0", expected=0, score_target_span_id="span-xyz"))
        evaluate(name="r", data=ds, task=echo, scorers=[exact], report_to=_reported())
        scorer = _by_name(memory_exporter.get_finished_spans())["exact"]
        assert "span-xyz" in scorer.attributes[SpanAttributes.SPAN_INPUT]


class TestEvalTraceIdentityContract:
    """Phase 4: the versioned identity attribute set on each reported eval trace root
    (see offline-eval/contract-notes/eval-trace-attributes.md)."""

    def test_root_carries_full_identity(self, memory_exporter):
        ds = _ds(1, source_trace_id="src-t", source_span_id="src-s")
        ds.dataset_id = "ds_remote"
        ds.dataset_version_id = "dsv_9"
        transport = PlatformTransport(
            "ds_remote",
            scorer_names=["exact"],
            candidate_version="cand-42",
            dataset_version_id="dsv_9",
            api_key="tr-x",
            host_url="https://h",
        )
        # avoid real network; give the transport a run id as create_run would
        calls = []
        transport._request = lambda m, p, b=None: (
            calls.append(p)
            or ({"evaluation_run_id": "run_777"} if p.endswith("evaluation-runs") else {})
        )
        result = evaluate(
            name="billing-routing",
            data=ds,
            task=echo,
            scorers=[exact],
            candidate_version="cand-42",
            environment="evaluation",
            report_to=transport,
        )
        by = _by_name(memory_exporter.get_finished_spans())
        a = by["evaluation-item"].attributes
        assert a[SpanAttributes.SPAN_TYPE] == "evaluation"  # authoritative eval classifier
        assert a[SpanAttributes.EVAL_CONTRACT_VERSION] == "1"
        assert a[SpanAttributes.EVAL_ENVIRONMENT] == "evaluation"
        assert a[SpanAttributes.ENVIRONMENT] == "evaluation"
        assert a[SpanAttributes.EVAL_NAME] == "billing-routing"
        assert a[SpanAttributes.EVAL_RUN_ID] == "run_777"
        assert a[SpanAttributes.EVAL_LOCAL_RUN_ID] == result.local_run_id  # client run id
        assert a[SpanAttributes.EVAL_DATASET_ID] == "ds_remote"
        assert a[SpanAttributes.EVAL_DATASET_VERSION_ID] == "dsv_9"
        assert a[SpanAttributes.EVAL_DATASET_NAME] == "d"
        assert a[SpanAttributes.EVAL_CASE_ID] == "c0"
        assert a[SpanAttributes.EVAL_CANDIDATE_VERSION] == "cand-42"
        assert a[SpanAttributes.EVAL_SOURCE_TRACE_ID] == "src-t"
        assert a[SpanAttributes.EVAL_SOURCE_SPAN_ID] == "src-s"
        # case identity is also on the task span (not just the root)
        assert by["task"].attributes[SpanAttributes.EVAL_CASE_ID] == "c0"

    def test_optional_identity_absent_when_unknown(self, memory_exporter):
        # A local inline run has no run_id / dataset_version_id / candidate_version:
        # those keys are omitted (not stamped as empty), but core identity stays.
        evaluate(name="r", data=_ds(1), task=echo, scorers=[exact], report_to=_reported())
        a = _by_name(memory_exporter.get_finished_spans())["evaluation-item"].attributes
        assert SpanAttributes.EVAL_RUN_ID not in a
        assert SpanAttributes.EVAL_DATASET_VERSION_ID not in a
        assert SpanAttributes.EVAL_CANDIDATE_VERSION not in a
        assert a[SpanAttributes.EVAL_NAME] == "r"
        assert a[SpanAttributes.EVAL_CONTRACT_VERSION] == "1"


class TestTraceIdAndErrors:
    def test_trace_id_returned_with_provider(self, memory_exporter):
        result = evaluate(name="r", data=_ds(1), task=echo, scorers=[exact], report_to=_reported())
        item = result.item_results[0]
        assert item.trace_id is not None
        # matches the emitted evaluation-item root trace id
        root = _by_name(memory_exporter.get_finished_spans())["evaluation-item"]
        assert item.trace_id == format(root.context.trace_id, "032x")

    def test_task_error_marks_span_and_isolates(self, memory_exporter):
        def boom(x):
            if x == 1:
                raise ValueError("kaboom")
            return x

        result = evaluate(name="r", data=_ds(3), task=boom, scorers=[exact], report_to=_reported())
        by_id = {it.case_id: it for it in result.item_results}
        assert "kaboom" in by_id["c1"].error
        assert by_id["c0"].error is None and by_id["c2"].error is None
        # the failing case's task span carries the eval error attribute
        spans = memory_exporter.get_finished_spans()
        err_tasks = [
            s for s in spans if s.name == "task" and s.attributes.get("traceroot.eval.error")
        ]
        assert any("kaboom" in s.attributes["traceroot.eval.error"] for s in err_tasks)


class TestReportedTraceExport:
    """A reported (cloud) evaluation exports its per-case eval traces and links each item
    to its emitted trace id."""

    def test_reported_eval_exports_and_links_trace_id(self, memory_exporter):
        result = evaluate(name="r", data=_ds(1), task=echo, scorers=[exact], report_to=_reported())
        spans = memory_exporter.get_finished_spans()
        assert sorted(s.name for s in spans) == ["evaluation-item", "exact", "task"]
        item = result.item_results[0]
        root = _by_name(spans)["evaluation-item"]
        assert item.trace_id == format(root.context.trace_id, "032x")


class TestUsageAttributionHierarchy:
    """Phase 6 metrics boundary: the SDK only guarantees the span hierarchy for provider
    usage attribution (LLM under the right task/scorer span). It never fabricates token or
    cost totals on its wrapper spans -- the backend derives usage from the real trace."""

    def test_llm_span_nests_under_scorer_not_task(self, memory_exporter):
        @traceroot.observe(name="judge_llm", type="llm")
        def judge(x):
            return 1.0

        def llm_scorer(ctx):
            return judge(ctx.output)  # a judge LLM call inside a scorer

        evaluate(name="r", data=_ds(1), task=echo, scorers=[llm_scorer], report_to=_reported())
        by = _by_name(memory_exporter.get_finished_spans())
        # the judge LLM span is a child of the SCORER span (usage attributes to the judge),
        # NOT the task span.
        assert by["judge_llm"].parent.span_id == by["llm_scorer"].context.span_id
        assert by["llm_scorer"].parent.span_id == by["evaluation-item"].context.span_id

    def test_eval_wrapper_spans_fabricate_no_tokens_or_cost(self, memory_exporter):
        evaluate(name="r", data=_ds(1), task=echo, scorers=[exact], report_to=_reported())
        wrapper_names = {"evaluation-item", "task", "exact"}
        for span in memory_exporter.get_finished_spans():
            if span.name in wrapper_names:
                keys = set(span.attributes or {})
                assert SpanAttributes.LLM_USAGE not in keys
                assert not any(("token" in k) or ("cost" in k) for k in keys), keys

    def test_trace_id_links_result_to_the_emitted_trace(self, memory_exporter):
        result = evaluate(name="r", data=_ds(1), task=echo, scorers=[exact], report_to=_reported())
        root = _by_name(memory_exporter.get_finished_spans())["evaluation-item"]
        assert result.item_results[0].trace_id == format(root.context.trace_id, "032x")


class TestLlmJudgeTrace:
    def test_judge_emits_nested_llm_span(self, memory_exporter):
        from traceroot.eval import llm_judge

        judge = llm_judge(
            name="conciseness",
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "ANSWER:\n{{output}}"}],
            complete=lambda model, messages: "0.8",  # deterministic, no network
        )
        evaluate(name="r", data=_ds(1), task=echo, scorers=[judge], report_to=_reported())

        by = _by_name(memory_exporter.get_finished_spans())
        # the judge's model call is its own LLM span, nested under the scorer span
        assert "llm_judge:conciseness" in by
        llm = by["llm_judge:conciseness"]
        scorer = by["conciseness"]  # scorer span is named by the judge's name
        assert llm.attributes[SpanAttributes.SPAN_TYPE] == "llm"
        assert llm.parent.span_id == scorer.context.span_id  # nested under the scorer
        assert "ANSWER" in str(llm.attributes[SpanAttributes.SPAN_INPUT])  # rendered prompt in
        assert "0.8" in str(llm.attributes[SpanAttributes.SPAN_OUTPUT])  # model response out

    def test_judge_self_instruments_custom_complete_even_with_provider_integration(
        self, memory_exporter, monkeypatch
    ):
        # A user-supplied complete is NOT traced by the provider integration (it never calls the
        # provider SDK), so the judge must still emit its own LLM span even when the integration is
        # active — otherwise the judge call would have no span at all. Deferring to the integration
        # only applies to the default dispatch.
        import traceroot
        from traceroot.eval import llm_judge
        from traceroot.instrumentation import Integration

        class _FakeClient:
            _instrumented = (Integration.ANTHROPIC,)

            def shutdown(self):  # teardown (reset_traceroot) calls these
                pass

            def flush(self):
                pass

        monkeypatch.setattr(traceroot, "_client", _FakeClient(), raising=False)

        judge = llm_judge(
            name="conciseness",
            model="claude-sonnet-5",  # anthropic model -> checks the ANTHROPIC integration
            messages=[{"role": "user", "content": "ANSWER:\n{{output}}"}],
            complete=lambda model, messages: "0.8",
        )
        result = evaluate(name="r", data=_ds(1), task=echo, scorers=[judge], report_to=_reported())

        by = _by_name(memory_exporter.get_finished_spans())
        assert "llm_judge:conciseness" in by  # a custom complete is self-instrumented
        assert "conciseness" in by  # the scorer span still exists
        assert result.item_results[0].scores[0].value == 0.8  # judge still ran + scored

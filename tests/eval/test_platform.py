"""Platform reporting: PlatformTransport + pull_dataset + evaluate(dataset_id=...).

Network is stubbed via PlatformTransport._request so no real HTTP happens.
"""

import json

import pytest

import traceroot
from traceroot import Dataset, EvalCase, Score, evaluate
from traceroot.eval.platform import PlatformTransport, pull_dataset, pull_dataset_version
from traceroot.eval.results import UploadState
from traceroot.eval.transport import RunHandle


class RecordingTransport(PlatformTransport):
    """PlatformTransport with the HTTP seam replaced by an in-memory recorder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests: list[tuple] = []

    def _request(self, method, path, body=None):
        self.requests.append((method, path, body))
        if path == "/api/v1/public/evaluation-runs":
            return {"evaluation_id": "ev_1", "evaluation_run_id": "run_1", "run_number": 1}
        if path.endswith("/results"):
            return {"evaluation_result_id": "res_1"}
        if path.endswith("/complete"):
            return {"evaluation_run_id": "run_1", "status": body["status"]}
        return {}


def _ds(n):
    ds = Dataset(name="d")
    for i in range(n):
        ds.upsert(EvalCase(input={"m": i}, id=f"tc{i}", expected={"r": i}))
    return ds


def echo(x):
    return {"r": x["m"]}


def acc(ctx):
    return 1.0 if ctx.output == ctx.expected else 0.0


class _RunPathTransport(PlatformTransport):
    """Stubs the HTTP seam; create_run optionally returns a run_url and/or run_path."""

    def __init__(self, *args, run_path=None, run_url=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._resp_run_path = run_path
        self._resp_run_url = run_url

    def _request(self, method, path, body=None):
        if path == "/api/v1/public/evaluation-runs":
            resp = {"evaluation_id": "ev_1", "evaluation_run_id": "run_1", "run_number": 1}
            if self._resp_run_path is not None:
                resp["run_path"] = self._resp_run_path
            if self._resp_run_url is not None:
                resp["run_url"] = self._resp_run_url
            return resp
        return {}


class TestRunUrl:
    def test_run_path_becomes_dashboard_url(self):
        t = _RunPathTransport(
            "ds_1",
            api_key="tr-x",
            host_url="https://app.traceroot.ai",
            run_path="/projects/proj_9/evaluations/run_1",
        )
        t.create_run("e", "d", None)
        assert t.run_path == "/projects/proj_9/evaluations/run_1"
        state = t.finish_run(RunHandle(name="e", dataset_name="d", metadata=None))
        assert state.dashboard_url == "https://app.traceroot.ai/projects/proj_9/evaluations/run_1"

    def test_run_url_is_preferred_over_host_plus_path(self):
        # Split origins: API host on :8000, run_url resolved against the UI origin on :3000.
        t = _RunPathTransport(
            "ds_1",
            api_key="tr-x",
            host_url="http://localhost:8000",
            run_path="/projects/proj_9/evaluations/run_1",
            run_url="http://localhost:3000/projects/proj_9/evaluations/run_1",
        )
        t.create_run("e", "d", None)
        assert t.run_url == "http://localhost:3000/projects/proj_9/evaluations/run_1"
        state = t.finish_run(RunHandle(name="e", dataset_name="d", metadata=None))
        # run_url wins verbatim; the API origin is never used to build the link.
        assert state.dashboard_url == "http://localhost:3000/projects/proj_9/evaluations/run_1"

    def test_absent_run_url_falls_back_to_host_plus_path(self):
        # Older control plane: no run_url -> same-origin host_url + run_path (no regression).
        t = _RunPathTransport(
            "ds_1",
            api_key="tr-x",
            host_url="https://app.traceroot.ai",
            run_path="/projects/proj_9/evaluations/run_1",
        )
        t.create_run("e", "d", None)
        assert t.run_url is None
        state = t.finish_run(RunHandle(name="e", dataset_name="d", metadata=None))
        assert state.dashboard_url == "https://app.traceroot.ai/projects/proj_9/evaluations/run_1"

    def test_absent_run_path_leaves_url_none(self):
        t = _RunPathTransport("ds_1", api_key="tr-x", host_url="https://h", run_path=None)
        t.create_run("e", "d", None)
        assert t.run_path is None
        state = t.finish_run(RunHandle(name="e", dataset_name="d", metadata=None))
        assert state.dashboard_url is None
        assert state.status == "uploaded"


class TestPlatformTransport:
    def test_requires_api_key(self):
        traceroot.shutdown()
        traceroot._client = None
        with pytest.raises(ValueError):
            PlatformTransport("ds_1", api_key="", host_url="")

    def test_create_run_payload(self):
        t = RecordingTransport("ds_1", scorer_names=["acc"], api_key="tr-x", host_url="https://h")
        t.create_run("routing-v2", "d", None)
        method, path, body = t.requests[0]
        assert (method, path) == ("POST", "/api/v1/public/evaluation-runs")
        assert body["evaluation_name"] == "routing-v2"
        assert body["dataset_id"] == "ds_1"
        assert body["scorers"] == [{"name": "acc", "version": "unversioned"}]
        assert t.run_id == "run_1"

    def test_create_run_sends_main_score_and_never_a_baseline(self):
        t = RecordingTransport(
            "ds_1",
            scorer_names=["acc", "helpful"],
            main_score_name="acc",  # multiple scorers now require an explicit main
            candidate_version="sonnet",
            api_key="tr-x",
            host_url="https://h",
        )
        t.create_run("routing", "d", None)
        body = t.requests[0][2]
        assert body["main_score_name"] == "acc"  # the explicitly selected headline metric
        assert body["candidate_version"] == "sonnet"
        # Comparison is the backend's job; the SDK sends no baseline linkage.
        assert "baseline_run_id" not in body

    def test_record_item_result_payload(self):
        t = RecordingTransport("ds_1", scorer_names=["acc"], api_key="tr-x", host_url="https://h")
        t.create_run("r", "d", None)
        from traceroot.eval.results import EvalItemResult

        item = EvalItemResult(
            case_id="tc0",
            input={"m": 0},
            output={"r": 0},
            expected={"r": 0},
            scores=[Score("acc", 1.0, comment="ok"), Score("label", "billing")],
            scorer_errors={},
            error=None,
            trace_id="abc123",
        )
        t.record_item_result(None, item)
        _, path, body = t.requests[-1]
        assert path == "/api/v1/public/evaluation-runs/run_1/results"
        assert body["test_case_id"] == "tc0"
        assert body["trace_id"] == "abc123"
        assert body["status"] == "passed"
        # backend requires string input/output (z.string()); dicts -> JSON text
        assert body["input"] == '{"m": 0}'
        assert body["candidate_output"] == '{"r": 0}'
        assert body["expected_output"] == '{"r": 0}'
        assert isinstance(body["input"], str)
        assert {
            "scorer_name": "acc",
            "scorer_version": "unversioned",
            "numeric_value": 1.0,
            "explanation": "ok",
        } in body["scores"]
        assert {
            "scorer_name": "label",
            "scorer_version": "unversioned",
            "string_value": "billing",
        } in body["scores"]

    def test_result_reports_duration_ms_as_nonneg_int(self):
        from traceroot.eval.results import EvalItemResult

        t = RecordingTransport("ds_1", scorer_names=["acc"], api_key="tr-x", host_url="https://h")
        t.create_run("r", "d", None)
        # duration_ms is a float on the item; the wire wants a nonnegative INT (backend
        # z.number().int().nonnegative()).
        item = EvalItemResult(
            "tc0", {}, {}, {}, [Score("acc", 1.0)], {}, None, "t", duration_ms=12.7
        )
        t.record_item_result(None, item)
        body = t.requests[-1][2]
        assert body["duration_ms"] == 13
        assert isinstance(body["duration_ms"], int)

    def test_missing_duration_is_null_not_zero(self):
        from traceroot.eval.results import EvalItemResult

        t = RecordingTransport("ds_1", scorer_names=["acc"], api_key="tr-x", host_url="https://h")
        t.create_run("r", "d", None)
        item = EvalItemResult("tc0", {}, {}, {}, [Score("acc", 1.0)], {}, None, None)  # no duration
        t.record_item_result(None, item)
        assert t.requests[-1][2]["duration_ms"] is None  # null, never 0-when-unknown

    def test_string_input_passes_through_unquoted(self):
        from traceroot.eval.results import EvalItemResult

        t = RecordingTransport("ds_1", scorer_names=["acc"], api_key="tr-x", host_url="https://h")
        t.create_run("r", "d", None)
        item = EvalItemResult(
            "tc0", "raw text", "billing", "billing", [Score("acc", 1.0)], {}, None, None
        )
        t.record_item_result(None, item)
        body = t.requests[-1][2]
        assert body["input"] == "raw text"  # not JSON-quoted
        assert body["candidate_output"] == "billing"

    def test_none_input_coerced_to_empty_string(self):
        # Backend /results `input` is a REQUIRED string; a None input must not be sent as null.
        from traceroot.eval.results import EvalItemResult

        t = RecordingTransport("ds_1", scorer_names=["acc"], api_key="tr-x", host_url="https://h")
        t.create_run("r", "d", None)
        item = EvalItemResult("tc0", None, None, None, [], {}, None, None)
        t.record_item_result(None, item)
        assert t.requests[-1][2]["input"] == ""

    def test_errored_status(self):
        t = RecordingTransport("ds_1", scorer_names=["acc"], api_key="tr-x", host_url="https://h")
        t.create_run("r", "d", None)
        from traceroot.eval.results import EvalItemResult

        item = EvalItemResult("tc1", {}, None, {"r": 0}, [], {}, "ValueError: boom", None)
        t.record_item_result(None, item)
        assert t.requests[-1][2]["status"] == "errored"

    def test_finish_run_sends_aggregate_main_score(self):
        from traceroot.eval.results import EvalItemResult

        t = RecordingTransport("ds_1", scorer_names=["acc"], api_key="tr-x", host_url="https://h")
        t.create_run("r", "d", None)
        # main score (acc) = 1.0 and 0.0 -> run mainScore should be their mean 0.5
        t.record_item_result(
            None, EvalItemResult("a", {}, {}, {}, [Score("acc", 1.0)], {}, None, None)
        )
        t.record_item_result(
            None, EvalItemResult("b", {}, {}, {}, [Score("acc", 0.0)], {}, None, None)
        )
        t.finish_run(None)
        complete_body = t.requests[-1][2]
        assert complete_body["main_score"] == 0.5  # -> run.mainScore, powers the UI Change delta
        assert complete_body["scored_count"] == 2

    def test_finish_run_returns_uploaded(self):
        t = RecordingTransport("ds_1", scorer_names=["acc"], api_key="tr-x", host_url="https://h")
        t.create_run("r", "d", None)
        state = t.finish_run(None)
        assert isinstance(state, UploadState)
        assert state.status == "uploaded"
        assert t.requests[-1][1] == "/api/v1/public/evaluation-runs/run_1/complete"

    def test_publish_dataset_stays_local(self):
        t = RecordingTransport("ds_1", scorer_names=["acc"], api_key="tr-x", host_url="https://h")
        r = t.publish_dataset("d", 3)
        assert r.status == "local_only"


class TestEvalPaths:
    def test_eval_calls_use_api_v1_public_prefix(self):
        # Eval endpoints are proxied by the Python backend under /api/v1/public/*,
        # the same host as trace ingestion.
        t = RecordingTransport("ds_1", scorer_names=["acc"], api_key="tr-x", host_url="https://h")
        t.create_run("r", "d", None)
        assert t.requests[0][1] == "/api/v1/public/evaluation-runs"


class TestPullDataset:
    def test_builds_dataset_from_snapshot(self, monkeypatch):
        calls = []

        def fake_get(url, api_key):
            calls.append(url)
            if "/datasets/" in url:
                return {"name": "tickets", "current_dataset_version_id": "dsv_9"}
            # NATIVE backend shape: the pull route JSON-decodes input/expected before
            # returning them, so they arrive as real dicts/values; metadata is JSONB.
            return {
                "items": [
                    {
                        "test_case_id": "tc0",
                        "input": {"m": 1},
                        "expected": {"r": 1},
                        "metadata": {"k": "v"},
                        "source_trace_id": "t1",
                        "source_span_id": "s1",
                    },
                    {"test_case_id": "tc1", "input": {"m": 2}, "expected": None},
                ]
            }

        monkeypatch.setattr("traceroot.eval.platform._http_get_json", fake_get)
        ds = pull_dataset("ds_1", api_key="tr-x", host_url="https://h")
        assert isinstance(ds, Dataset)
        assert ds.dataset_id == "ds_1"
        assert ds.dataset_version_id == "dsv_9"
        assert len(ds) == 2
        assert ds.get("tc0").expected == {"r": 1}
        assert ds.get("tc0").metadata == {"k": "v"}
        assert ds.get("tc0").source_trace_id == "t1"

    def test_pull_takes_native_values_verbatim_no_redecode(self, monkeypatch):
        # The backend returns native JSON. The SDK must pass values through unchanged:
        # a genuine JSON-looking string stays a string (re-decoding would corrupt it),
        # and "42" stays the string "42", not int 42.
        def fake_get(url, api_key):
            if "/datasets/" in url:
                return {"name": "t", "current_dataset_version_id": "dsv_1"}
            return {
                "items": [
                    {"test_case_id": "d", "input": {"message": "hi"}, "expected": {"r": 1}},
                    {"test_case_id": "s", "input": '{"looks":"like json"}', "expected": "42"},
                    {"test_case_id": "n", "input": 7, "expected": True},
                ]
            }

        monkeypatch.setattr("traceroot.eval.platform._http_get_json", fake_get)
        ds = pull_dataset("ds_1", api_key="tr-x", host_url="https://h")
        assert ds.get("d").input == {"message": "hi"} and ds.get("d").expected == {"r": 1}
        # JSON-looking string preserved AS A STRING (not decoded to a dict)
        assert ds.get("s").input == '{"looks":"like json"}'
        assert ds.get("s").expected == "42" and type(ds.get("s").expected) is str
        assert ds.get("n").input == 7 and ds.get("n").expected is True


class TestDatasetLifecycleIntegration:
    """End-to-end push -> pull through a faithful in-memory double of the backend routes
    (encodeJsonValue on write, decodeJsonValue on read, version-by-id retrieval). Drives
    the REAL PlatformDatasetSync.push_dataset and pull_dataset code paths."""

    def _install_backend(self, monkeypatch):
        store: dict = {"versions": {}, "current": {}}

        def _encode(v):  # backend encodeJsonValue at the storage seam
            return json.dumps(v)

        def _decode(t):  # backend decodeJsonValue on read
            try:
                return json.loads(t)
            except (ValueError, TypeError):
                return t

        def http_json(method, url, api_key, body=None):
            if url.endswith("/datasets"):
                return {}
            if url.endswith("/versions") and method == "POST":
                ds_id = url.split("/datasets/")[1].split("/")[0]
                vid = f"dsv_{len(store['versions']) + 1}"
                items = [
                    {
                        "test_case_id": ch["test_case_id"],
                        "input": _encode(ch["input"]),
                        "expected": _encode(ch["expected"]) if "expected" in ch else None,
                        "metadata": ch.get("metadata"),
                    }
                    for ch in body["changes"]
                ]
                store["versions"][vid] = {
                    "dataset_id": ds_id,
                    "dataset_version_id": vid,
                    "items": items,
                }
                store["current"][ds_id] = vid
                return {"dataset_version_id": vid, "version_number": len(store["versions"])}
            raise AssertionError(f"unexpected write {method} {url}")

        def http_get_json(url, api_key):
            if "/dataset-versions/" in url:
                v = store["versions"][url.rsplit("/", 1)[-1]]
                return {
                    "dataset_id": v["dataset_id"],
                    "dataset_version_id": v["dataset_version_id"],
                    "items": [
                        {
                            **it,
                            "input": _decode(it["input"]),
                            "expected": _decode(it["expected"])
                            if it["expected"] is not None
                            else None,
                        }
                        for it in v["items"]
                    ],
                }
            ds_id = url.rsplit("/", 1)[-1]
            return {"name": ds_id, "current_dataset_version_id": store["current"][ds_id]}

        monkeypatch.setattr("traceroot.eval.platform._http_json", http_json)
        monkeypatch.setattr("traceroot.eval.platform._http_get_json", http_get_json)
        return store

    def test_push_then_pull_preserves_types_end_to_end(self, monkeypatch):
        from traceroot.eval.dataset_sync import PlatformDatasetSync

        self._install_backend(monkeypatch)
        sync = PlatformDatasetSync.__new__(PlatformDatasetSync)
        sync.host_url = "https://h"
        sync.api_key = "tr-x"

        ds = Dataset("d")
        ds.dataset_id = "ds_1"
        ds.add(input={"m": "hi"}, id="tc0", expected={"r": "billing"}, metadata={"s": 1})
        ds.add(input='{"looks":"like json"}', id="tc1", expected="42")  # tricky strings
        ds.add(input=[1, 2, 3], id="tc2")
        result = sync.push_dataset(ds.snapshot(), None)
        assert result.status == "uploaded"

        pulled = pull_dataset("ds_1", api_key="tr-x", host_url="https://h")
        assert pulled.get("tc0").input == {"m": "hi"}  # dict survives
        assert pulled.get("tc0").expected == {"r": "billing"}
        assert pulled.get("tc0").metadata == {"s": 1}
        assert pulled.get("tc1").input == '{"looks":"like json"}'  # JSON-looking str stays str
        assert pulled.get("tc1").expected == "42" and type(pulled.get("tc1").expected) is str
        assert pulled.get("tc2").input == [1, 2, 3]  # list survives
        # and the exact version can be pulled back by id
        exact = pull_dataset_version(
            result.dataset_version_id, dataset_id="ds_1", api_key="tr-x", host_url="https://h"
        )
        assert exact.dataset_version_id == result.dataset_version_id
        assert exact.get("tc2").input == [1, 2, 3]


class TestExactVersionPull:
    """pull_dataset(version_id=...) / pull_dataset_version fetch a precise immutable
    version and never silently substitute the current one."""

    def _backend(self, monkeypatch, *, current="dsv_current", versions=None):
        versions = versions or {}
        seen = []

        def fake_get(url, api_key):
            seen.append(url)
            if "/dataset-versions/" in url:
                vid = url.rsplit("/", 1)[-1]
                if vid not in versions:
                    raise RuntimeError(f"GET {url} failed -- HTTP 404: not found")
                return versions[vid]
            return {"name": "tickets", "current_dataset_version_id": current}

        monkeypatch.setattr("traceroot.eval.platform._http_get_json", fake_get)
        return seen

    def _version(self, vid, dataset_id, n=1):
        return {
            "dataset_id": dataset_id,
            "dataset_version_id": vid,
            "items": [
                {"test_case_id": f"tc{i}", "input": {"i": i}, "expected": {"i": i}}
                for i in range(n)
            ],
        }

    def test_pull_current_version(self, monkeypatch):
        self._backend(
            monkeypatch,
            current="dsv_current",
            versions={"dsv_current": self._version("dsv_current", "ds_1", n=2)},
        )
        ds = pull_dataset("ds_1", api_key="tr-x", host_url="https://h")
        assert ds.dataset_version_id == "dsv_current" and len(ds) == 2

    def test_pull_exact_version_not_current(self, monkeypatch):
        seen = self._backend(
            monkeypatch,
            current="dsv_current",
            versions={"dsv_old": self._version("dsv_old", "ds_1", n=3)},
        )
        ds = pull_dataset("ds_1", version_id="dsv_old", api_key="tr-x", host_url="https://h")
        assert ds.dataset_version_id == "dsv_old"  # NOT dsv_current
        assert len(ds) == 3
        assert not any(u.endswith("/dataset-versions/dsv_current") for u in seen)

    def test_pull_dataset_version_directly(self, monkeypatch):
        self._backend(monkeypatch, versions={"dsv_x": self._version("dsv_x", "ds_9", n=1)})
        ds = pull_dataset_version("dsv_x", api_key="tr-x", host_url="https://h")
        assert ds.dataset_id == "ds_9" and ds.dataset_version_id == "dsv_x"

    def test_missing_version_raises_clear_error(self, monkeypatch):
        self._backend(monkeypatch, versions={})
        with pytest.raises(ValueError, match="not found"):
            pull_dataset_version("dsv_missing", api_key="tr-x", host_url="https://h")

    def test_mismatched_dataset_version_identity_raises(self, monkeypatch):
        self._backend(monkeypatch, versions={"dsv_x": self._version("dsv_x", "ds_OTHER", n=1)})
        with pytest.raises(ValueError, match="belongs to dataset"):
            pull_dataset("ds_1", version_id="dsv_x", api_key="tr-x", host_url="https://h")
        with pytest.raises(ValueError, match="belongs to dataset"):
            pull_dataset_version("dsv_x", dataset_id="ds_1", api_key="tr-x", host_url="https://h")


class TestDatasetNativeJsonContract:
    """The dataset HTTP boundary transports NATIVE JSON. The backend owns the single
    JSON encode/decode at its storage column; the SDK sends/receives values as-is.
    These tests exercise the ACTUAL backend request/response representation, not an
    SDK encode-then-decode that cancels out.
    """

    @staticmethod
    def _backend_encode(native):
        # Mirror of the backend encodeJsonValue = JSON.stringify(value ?? null),
        # applied at the Postgres TEXT storage seam.
        return json.dumps(native)

    @staticmethod
    def _backend_decode(text):
        # Mirror of the backend decodeJsonValue = JSON.parse with raw-string fallback,
        # applied on read before the value is returned to the SDK.
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return text

    @pytest.mark.parametrize(
        "original",
        [
            {"message": "charge me twice"},
            {"nested": {"a": [1, 2]}, "n": 3},
            [1, 2, 3],
            42,
            3.5,
            True,
            None,
            "hello world",
            '{"looks": "like json"}',  # genuine JSON-looking string -> stays a string
            "[1, 2, 3]",
            "42",  # numeric-looking string -> stays a string, not int 42
        ],
    )
    def test_type_survives_full_native_round_trip(self, original):
        # SDK sends the native value on push (serialize_value is identity for these);
        # the backend encodes to text at storage and decodes on read; the SDK takes the
        # pulled value verbatim. Every JSON type survives, incl. JSON-looking strings.
        from traceroot.eval.platform import serialize_value

        on_wire_push = serialize_value(original)  # what the SDK actually sends
        stored = self._backend_encode(on_wire_push)  # backend encodeJsonValue at storage
        returned = self._backend_decode(stored)  # backend decodeJsonValue on read
        pulled = returned  # SDK pull is pass-through (no re-decode)

        assert pulled == original
        assert type(pulled) is type(original)  # int 42 != str "42", bool True != int 1

    def test_push_sends_native_values_not_json_text(self, monkeypatch):
        # PlatformDatasetSync must send input/expected/metadata as native values so the
        # backend does the single encode. A dict stays a dict; a string stays a string.
        from traceroot import Dataset
        from traceroot.eval.dataset_sync import PlatformDatasetSync

        sync = PlatformDatasetSync.__new__(PlatformDatasetSync)
        sync.host_url = "https://h"
        sync.api_key = "tr-x"
        bodies = []
        sync._request = lambda method, path, body=None: (
            bodies.append((path, body))
            or {
                "dataset_version_id": "dsv_1",
                "version_number": 1,
            }
        )

        ds = Dataset("d")
        ds.add(input={"message": "hi"}, id="tc0", expected={"route": "billing"}, metadata={"s": 1})
        ds.add(input="plain string", id="tc1", expected="42")
        sync.push_dataset(ds.snapshot(), None)

        version_body = next(b for p, b in bodies if p.endswith("/versions"))
        by_id = {c["test_case_id"]: c for c in version_body["changes"]}
        assert by_id["tc0"]["input"] == {"message": "hi"}  # native dict, NOT '{"message": "hi"}'
        assert by_id["tc0"]["expected"] == {"route": "billing"}
        assert by_id["tc0"]["metadata"] == {"s": 1}
        # a genuine string is sent as the string itself, not JSON-quoted
        assert by_id["tc1"]["input"] == "plain string"
        assert by_id["tc1"]["expected"] == "42"


class TestReportingDefaults:
    """Cloud-only: a run always reports. With credentials + a synced dataset (pulled/pushed
    or an explicit dataset_id) it builds a PlatformTransport and uploads. Without credentials,
    or with an unsynced (inline) dataset, it raises. An explicit report_to always wins.

    These tests exercise the real _auto_transport, so they opt out of conftest's default
    FakeTransport with @pytest.mark.no_default_transport."""

    def _spy(self, monkeypatch):
        calls = []

        def fake_request(self, method, path, body=None):
            calls.append(path)
            if path == "/api/v1/public/evaluation-runs":
                return {"evaluation_run_id": "run_1"}
            if path.endswith("/complete"):
                return {"status": (body or {}).get("status")}
            return {"evaluation_result_id": "r"}

        monkeypatch.setattr(PlatformTransport, "_request", fake_request)
        return calls

    def _remote_ds(self, n=2):
        ds = _ds(n)
        ds.dataset_id = "ds_remote"
        ds.dataset_version_id = "dsv_7"  # as pull_dataset stamps it
        return ds

    def _with_creds(self, fn):
        # Reset first so a leftover client from another test doesn't make initialize a
        # no-op (which would drop our ambient credentials).
        traceroot.shutdown()
        traceroot._client = None
        traceroot.initialize(api_key="tr-test", enabled=False)  # ambient creds
        try:
            return fn()
        finally:
            traceroot.shutdown()
            traceroot._client = None

    @pytest.mark.no_default_transport
    def test_no_credentials_raises(self, monkeypatch):
        # A remote dataset but no credentials -> nothing to report to -> raise (cloud-only).
        calls = self._spy(monkeypatch)
        with pytest.raises(RuntimeError, match="reports to the TraceRoot platform"):
            evaluate(name="r", data=self._remote_ds(), task=echo, scorers=[acc])
        assert calls == []

    @pytest.mark.no_default_transport
    def test_inline_dataset_raises_even_with_creds(self, monkeypatch):
        # A purely local dataset (no dataset_id/version) can't be created server-side, so
        # there is nothing to report against -> raise even with credentials.
        calls = self._spy(monkeypatch)
        with pytest.raises(RuntimeError, match="reports to the TraceRoot platform"):
            self._with_creds(lambda: evaluate(name="r", data=_ds(1), task=echo, scorers=[acc]))
        assert calls == []

    @pytest.mark.no_default_transport
    def test_remote_dataset_with_creds_uploads(self, monkeypatch):
        calls = self._spy(monkeypatch)
        result = self._with_creds(
            lambda: evaluate(name="r", data=self._remote_ds(), task=echo, scorers=[acc])
        )
        assert result.upload_state.status == "uploaded"
        assert result.run_id == "run_1"
        assert "/api/v1/public/evaluation-runs" in calls

    @pytest.mark.no_default_transport
    def test_dataset_id_with_creds_uploads(self, monkeypatch):
        calls = self._spy(monkeypatch)
        result = self._with_creds(
            lambda: evaluate(name="r", data=_ds(2), task=echo, scorers=[acc], dataset_id="ds_1")
        )
        assert result.upload_state.status == "uploaded"
        assert result.dataset.dataset_id == "ds_1"
        assert "/api/v1/public/evaluation-runs" in calls

    def test_explicit_report_to_uploads(self, monkeypatch):
        # An explicit platform transport always uploads (and carries baseline linking).
        recorded = {}

        def fake_request(self, method, path, body=None):
            recorded.setdefault("paths", []).append(path)
            if path == "/api/v1/public/evaluation-runs":
                return {"evaluation_run_id": "run_1"}
            if path.endswith("/complete"):
                return {"status": body["status"]}
            return {"evaluation_result_id": "r"}

        monkeypatch.setattr(PlatformTransport, "_request", fake_request)
        transport = PlatformTransport(
            "ds_1", scorer_names=["acc"], api_key="tr-x", host_url="https://h"
        )
        result = evaluate(name="r", data=_ds(2), task=echo, scorers=[acc], report_to=transport)
        assert result.upload_state.status == "uploaded"
        paths = recorded["paths"]
        assert paths[0] == "/api/v1/public/evaluation-runs"
        assert paths[-1].endswith("/complete")
        assert sum(1 for p in paths if p.endswith("/results")) == 2

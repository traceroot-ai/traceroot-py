"""Platform reporting transport for offline evaluation.

Implements the ``EvalTransport`` seam against the TraceRoot backend's offline-eval
reporting endpoints (see the backend ``offline-eval-sdk-contract.md``):

    POST /api/v1/public/evaluation-runs                    (register/start a run)
    POST /api/v1/public/evaluation-runs/{id}/results       (upsert one result + scores)
    POST /api/v1/public/evaluation-runs/{id}/complete      (finish a run)
    GET  /api/v1/public/datasets/{id}                       (fetch dataset -> version id)
    GET  /api/v1/public/dataset-versions/{id}               (fetch immutable snapshot)

Auth is ``Authorization: Bearer <api_key>`` (an existing project Access Key) - the
same credential and host as trace ingestion. Uses only the standard library.

The SDK is reporting-only by contract: the server owns dataset_id / test_case_id /
dataset_version_id, so datasets are FETCHED (``pull_dataset``), not created here.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from traceroot.eval.results import EvalItemResult, UploadState
from traceroot.eval.transport import PublishResult, RunHandle
from traceroot.eval.types import Dataset, EvalCase, Score
from traceroot.utils import serialize_value

_DEFAULT_PASS_THRESHOLD = 1.0


def _as_text(value: Any) -> str | None:
    """Coerce a value to a string for the backend's z.string() fields.

    Strings pass through as-is; None stays None (for nullable fields); everything
    else is JSON-encoded (dicts/lists/numbers become their JSON text).
    """
    if value is None or isinstance(value, str):
        return value
    return json.dumps(serialize_value(value))


def _resolve_credentials(api_key: str | None, host_url: str | None) -> tuple[str, str]:
    """Fill api_key/host_url from the global client when not explicitly given.

    The eval endpoints are served by the Python backend under ``/api/v1/public/*``
    (a proxy to the app) - the SAME host as trace ingestion - so a single
    ``host_url`` reaches both traces and eval calls.
    """
    if api_key is None or host_url is None:
        from traceroot import get_client

        client = get_client()
        if client is not None:
            api_key = api_key if api_key is not None else client.api_key
            host_url = host_url if host_url is not None else client.host_url
    return api_key or "", host_url or ""


def _http_json(method: str, url: str, api_key: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        location = exc.headers.get("Location", "") if exc.headers else ""
        blob = f"{location} {detail}".lower()
        if exc.code in (301, 302, 303, 307, 308) and ("sign-in" in blob or "/auth/" in blob):
            raise RuntimeError(
                f"{method} {url} -> HTTP {exc.code} redirect to sign-in. The API key was not "
                "honored: the backend's /api/public route is behind the app's session gate. "
                "The app auth middleware must exempt /api/public (server-side fix; not the SDK "
                "or your key)."
            ) from exc
        raise RuntimeError(f"{method} {url} -> HTTP {exc.code}: {detail}") from exc


def _http_get_json(url: str, api_key: str) -> dict:
    return _http_json("GET", url, api_key)


class PlatformTransport:
    """Reports an evaluation run to the TraceRoot backend. One instance per run."""

    def __init__(
        self,
        dataset_id: str,
        *,
        scorer_names: list[str] | None = None,
        candidate_version: str | None = None,
        environment: str = "evaluation",
        main_score_name: str | None = None,
        dataset_version_id: str | None = None,
        client_run_id: str | None = None,
        pass_threshold: float = _DEFAULT_PASS_THRESHOLD,
        api_key: str | None = None,
        host_url: str | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.scorer_names = scorer_names or []
        self.candidate_version = candidate_version or "sdk"
        self.environment = environment
        self.main_score_name = main_score_name or (
            self.scorer_names[0] if self.scorer_names else None
        )
        self.dataset_version_id = dataset_version_id
        self.client_run_id = client_run_id
        self.pass_threshold = pass_threshold
        self.api_key, self.host_url = _resolve_credentials(api_key, host_url)
        if not self.api_key:
            raise ValueError(
                "PlatformTransport needs an API key. Call traceroot.initialize(api_key=...) "
                "or pass api_key=... (uploading requires credentials)."
            )
        self.host_url = self.host_url.rstrip("/")
        self.run_id: str | None = None
        self._scored = 0
        self._task_errors = 0
        self._scorer_errors = 0

    # --- HTTP seam (overridable in tests) ---
    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        return _http_json(method, f"{self.host_url}{path}", self.api_key, body)

    # --- EvalTransport protocol ---
    def create_run(
        self,
        name: str,
        dataset_name: str,
        metadata: dict | None,
        client_run_id: str | None = None,
    ) -> RunHandle:
        body: dict[str, Any] = {
            "evaluation_name": name,
            "dataset_id": self.dataset_id,
            "candidate_version": self.candidate_version,
            "environment": self.environment,
            "scorers": [{"name": n, "version": "1"} for n in self.scorer_names],
        }
        if self.dataset_version_id is not None:
            body["dataset_version_id"] = self.dataset_version_id
        if self.main_score_name is not None:
            body["main_score_name"] = self.main_score_name
        # Idempotency key: prefer the one the RunSession drives with, else our own.
        effective_crun = client_run_id or self.client_run_id
        if effective_crun is not None:
            body["client_run_id"] = effective_crun
        resp = self._request("POST", "/api/v1/public/evaluation-runs", body)
        self.run_id = resp["evaluation_run_id"]
        return RunHandle(name=name, dataset_name=dataset_name, metadata=metadata)

    def register_item(self, run: RunHandle, case: EvalCase) -> None:
        # The item->trace link is folded into the result upsert (contract), so no-op.
        return None

    def record_item_result(self, run: RunHandle, item_result: EvalItemResult) -> None:
        status, main_score = self._status_and_main(item_result)
        if status == "errored":
            self._task_errors += 1
        elif status in ("passed", "failed"):
            self._scored += 1
        self._scorer_errors += len(item_result.scorer_errors)
        self._request(
            "POST",
            f"/api/v1/public/evaluation-runs/{self.run_id}/results",
            {
                "test_case_id": item_result.case_id,
                "trace_id": item_result.trace_id,
                # The backend validates these as strings (z.string()); serialize
                # non-string values (dicts/numbers) to JSON text. `input` is a
                # REQUIRED string (not nullable) -> coerce a missing input to "".
                "input": _as_text(item_result.input) or "",
                "expected_output": _as_text(item_result.expected),
                "candidate_output": _as_text(item_result.output),
                "status": status,
                "main_score": main_score,
                "task_error": item_result.error,
                "scores": self._scores_payload(item_result),
            },
        )

    def record_scores(self, run: RunHandle, case_id: str, scores: list[Score]) -> None:
        # Already sent inside record_item_result (which carries the full item).
        return None

    def finish_run(self, run: RunHandle, status: str | None = None) -> UploadState:
        effective = status or (
            "completed_with_errors" if (self._task_errors or self._scorer_errors) else "completed"
        )
        self._request(
            "POST",
            f"/api/v1/public/evaluation-runs/{self.run_id}/complete",
            {
                "status": effective,
                "scored_count": self._scored,
                "task_error_count": self._task_errors,
                "scorer_error_count": self._scorer_errors,
            },
        )
        # The backend returns no dashboard URL; report uploaded with url unknown.
        return UploadState(status="uploaded", dashboard_url=None)

    def publish_dataset(self, dataset_name: str, item_count: int) -> PublishResult:
        # Datasets are server/UI-owned; the SDK cannot create them. Stay local-only.
        return PublishResult(status="local_only", dataset_name=dataset_name, item_count=item_count)

    # --- mapping helpers ---
    def _scores_payload(self, item_result: EvalItemResult) -> list[dict]:
        payload: list[dict] = []
        for s in item_result.scores:
            entry: dict[str, Any] = {"scorer_name": s.name, "scorer_version": "1"}
            v = s.value
            if isinstance(v, bool):
                entry["bool_value"] = v
            elif isinstance(v, (int, float)):
                entry["numeric_value"] = float(v)
            else:
                entry["string_value"] = str(v)
            if s.comment is not None:
                entry["explanation"] = s.comment
            payload.append(entry)
        # A failing scorer is a score with an error and null value (never 0).
        for name, msg in item_result.scorer_errors.items():
            payload.append({"scorer_name": name, "scorer_version": "1", "error": msg})
        return payload

    def _status_and_main(self, item_result: EvalItemResult) -> tuple[str, float | None]:
        if item_result.error is not None:
            return "errored", None
        main = None
        for s in item_result.scores:
            if isinstance(s.value, bool) or not isinstance(s.value, (int, float)):
                continue
            if self.main_score_name is None or s.name == self.main_score_name:
                main = float(s.value)
                break
        if main is None:
            return "not_scored", None
        return ("passed" if main >= self.pass_threshold else "failed"), main


def pull_dataset(
    dataset_id: str, *, api_key: str | None = None, host_url: str | None = None
) -> Dataset:
    """Fetch a platform dataset's current snapshot into a local :class:`Dataset`.

    The returned Dataset carries ``dataset_id`` / ``dataset_version_id`` so that
    ``evaluate(data=dataset, ...)`` reports back to the exact same dataset version.
    """
    key, host = _resolve_credentials(api_key, host_url)
    if not key:
        raise ValueError(
            "pull_dataset needs an API key (initialize traceroot or pass api_key=...)."
        )
    host = host.rstrip("/")

    meta = _http_get_json(f"{host}/api/v1/public/datasets/{dataset_id}", key)
    version_id = meta["current_dataset_version_id"]
    snapshot = _http_get_json(f"{host}/api/v1/public/dataset-versions/{version_id}", key)

    ds = Dataset(name=meta.get("name", dataset_id))
    ds.dataset_id = dataset_id
    ds.dataset_version_id = version_id
    for item in snapshot.get("items", []):
        ds.upsert(
            EvalCase(
                id=item["test_case_id"],
                input=item["input"],
                expected=item.get("expected"),
                metadata=item.get("metadata"),
                source_trace_id=item.get("source_trace_id"),
                source_span_id=item.get("source_span_id"),
            )
        )
    return ds

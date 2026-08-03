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
from urllib.parse import quote

from traceroot.eval.results import EvalItemResult, UploadState
from traceroot.eval.transport import PublishResult, RunHandle
from traceroot.eval.types import Dataset, EvalCase, Score
from traceroot.utils import serialize_value

_DEFAULT_PASS_THRESHOLD = 1.0

# The backend's public contract requires a NON-EMPTY scorer version string
# (ScorerRefSchema/ScoreInputSchema: z.string().min(1)). The SDK data model and the
# CLI event/artifact path stay honest with version=None for an unversioned scorer;
# only at this reporting boundary do we map None -> an explicit "unversioned" sentinel
# so the request validates. This is a clear marker, not an invented version number.
_UNVERSIONED_SCORER = "unversioned"


def _as_text(value: Any) -> str | None:
    """Coerce a value to a string for the backend's z.string() fields.

    Strings pass through as-is; None stays None (for nullable fields); everything
    else is JSON-encoded (dicts/lists/numbers become their JSON text).
    """
    if value is None or isinstance(value, str):
        return value
    return json.dumps(serialize_value(value))


def _duration_ms(value: float | None) -> int | None:
    """Per-case wall-clock duration for the wire: a nonnegative INTEGer of milliseconds
    (backend ``z.number().int().nonnegative()``). Unknown stays None -- never 0."""
    if value is None:
        return None
    return max(0, round(value))


# NOTE: dataset case ``input``/``expected``/``metadata`` cross the HTTP boundary as
# NATIVE JSON values. The backend owns the single JSON encode/decode at its storage
# column (dataset authoring schema is ``z.unknown()``; the pull route JSON-decodes
# before returning). The SDK must NOT add its own json.dumps/json.loads for these --
# that would double-encode/decode and corrupt genuine JSON-looking strings. Explicit
# JSON text is retained only where the wire genuinely requires a string: the
# evaluation-RESULT reporting fields (``_as_text``) and local file persistence
# (``Dataset.save``/``EvalRunResult.save``).


def _numeric_score(value: Any) -> float | None:
    """The numeric value of a score (bool -> 1.0/0.0), or None for categorical/None."""
    if isinstance(value, bool):  # bool before int: True->1.0, False->0.0
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


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
        pass_threshold: float | None = None,
        scorer_specs: list[dict[str, Any]] | None = None,
        api_key: str | None = None,
        host_url: str | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.scorer_names = scorer_names or []
        # Rich scorer descriptors (name/version/value_type/direction/threshold). The engine
        # fills this from the actual scorer callables when the caller leaves it None; an
        # explicit value wins. Falls back to scorer_names when unset.
        self.scorer_specs = scorer_specs
        self.candidate_version = candidate_version or "sdk"
        self.environment = environment
        # Deterministic main-metric resolution (never a silent "first scorer function name"):
        #  - an explicit main_score wins (validated at finish against what was actually emitted);
        #  - a single scorer resolves name-agnostically from its one score, so a scorer whose
        #    function name differs from its emitted Score name can no longer silently zero the run;
        #  - multiple scorers with no explicit main have no headline metric here (the engine
        #    requires an explicit main_score for a reported multi-scorer run).
        n_scorers = len(self.scorer_names) or (len(self.scorer_specs) if self.scorer_specs else 0)
        self._main_configured = main_score_name is not None
        # Registration reports main_score_name ONLY when the user configured it. A single
        # scorer's metric is late-bound (resolved from what it actually emits) and reported at
        # completion -- never fabricated from the scorer's function name here.
        self.main_score_name = main_score_name
        self._name_agnostic_main = (not self._main_configured) and n_scorers == 1
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
        self.run_url: str | None = None  # absolute UI run link, when the backend returns one
        self.run_path: str | None = None  # UI-relative run path (back-compat fallback)
        self._scored = 0
        self._task_errors = 0
        self._scorer_errors = 0
        self._main_sum = 0.0  # aggregate of the main-score values for run.mainScore
        self._main_count = 0

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
        provenance: dict | None = None,
    ) -> RunHandle:
        body: dict[str, Any] = {
            "evaluation_name": name,
            "dataset_id": self.dataset_id,
            "candidate_version": self.candidate_version,
            "environment": self.environment,
            "scorers": self._scorer_refs(),
        }
        if self.dataset_version_id is not None:
            body["dataset_version_id"] = self.dataset_version_id
        if self.main_score_name is not None:
            body["main_score_name"] = self.main_score_name
        # Idempotency key: prefer the one the caller drives with, else our own.
        effective_crun = client_run_id or self.client_run_id
        if effective_crun is not None:
            body["client_run_id"] = effective_crun
        # Typed execution provenance (git/CI/SDK identity) and free-form user metadata.
        # Both are optional on the backend; omit when there is nothing to report so we
        # match its absent-or-null rules rather than sending empty objects.
        if provenance:
            body["provenance"] = provenance
        if metadata:
            body["metadata"] = metadata
        resp = self._request("POST", "/api/v1/public/evaluation-runs", body)
        self.run_id = resp["evaluation_run_id"]
        # Optional, absent on older/self-hosted backends. Prefer the absolute run_url
        # (resolved against the UI origin) so the link is correct even when the API and
        # UI live on different origins; keep run_path as a same-origin fallback.
        self.run_url = resp.get("run_url")
        self.run_path = resp.get("run_path")
        return RunHandle(name=name, dataset_name=dataset_name, metadata=metadata)

    def _scorer_refs(self) -> list[dict[str, Any]]:
        """Scorer descriptors for run registration. Prefers rich specs (value_type/
        direction/threshold, forward-compatible) and falls back to name/version. The
        backend requires a non-empty version string, so unversioned scorers use the
        sentinel; optional metadata fields are omitted when unknown."""
        if self.scorer_specs:
            refs = []
            for spec in self.scorer_specs:
                ref: dict[str, Any] = {
                    "name": spec["name"],
                    "version": spec.get("version") or _UNVERSIONED_SCORER,
                }
                # Comparison metadata + the read-only definition (scorer_type + type-specific
                # fields). Absent fields are omitted, never null-filled.
                for k in (
                    "scorer_type",
                    "value_type",
                    "direction",
                    "threshold",
                    "output_type",
                    "description",
                    "metadata",
                    "required_inputs",
                    "language",
                    "source",
                    "model",
                    "messages",
                ):
                    if spec.get(k) is not None:
                        ref[k] = spec[k]
                refs.append(ref)
            return refs
        return [{"name": n, "version": _UNVERSIONED_SCORER} for n in self.scorer_names]

    def _effective_threshold(self) -> float:
        """The pass threshold for status: an explicit pass_threshold wins; else the main
        scorer's DECLARED threshold; else the default. Keeps cloud status in agreement
        with a scorer's declared threshold (Phase 3)."""
        if self.pass_threshold is not None:
            return self.pass_threshold
        for spec in self.scorer_specs or []:
            if spec.get("name") == self.main_score_name and spec.get("threshold") is not None:
                return float(spec["threshold"])
        return _DEFAULT_PASS_THRESHOLD

    def _effective_direction(self) -> str:
        """The main scorer's DECLARED comparison direction (higher_is_better by default).
        lower_is_better inverts the threshold comparison; none -> not_scored."""
        for spec in self.scorer_specs or []:
            if spec.get("name") == self.main_score_name and spec.get("direction") is not None:
                return str(spec["direction"])
        return "higher_is_better"

    def register_item(self, run: RunHandle, case: EvalCase) -> None:
        # The item->trace link is folded into the result upsert (contract), so no-op.
        return None

    def record_item_result(self, run: RunHandle, item_result: EvalItemResult) -> None:
        status, main_score = self._status_and_main(item_result)
        if status == "errored":
            self._task_errors += 1
        elif status in ("passed", "failed"):
            self._scored += 1
        if main_score is not None:
            self._main_sum += main_score
            self._main_count += 1
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
                # Total wall-clock for this case (task + its scorers); nonneg int ms, null
                # when unknown. Run duration is NOT summed from cases (they run concurrently).
                "duration_ms": _duration_ms(item_result.duration_ms),
                "scores": self._scores_payload(item_result),
            },
        )

    def record_scores(self, run: RunHandle, case_id: str, scores: list[Score]) -> None:
        # Already sent inside record_item_result (which carries the full item).
        return None

    def finish_run(
        self,
        run: RunHandle,
        status: str | None = None,
        main_score_name: str | None = None,
    ) -> UploadState:
        # Pure reporter: the engine owns the ONE main-score resolution and passes the terminal
        # ``status`` (e.g. "failed" on a misconfiguration) and the resolved ``main_score_name``.
        effective = status or (
            "completed_with_errors" if (self._task_errors or self._scorer_errors) else "completed"
        )
        body: dict[str, Any] = {
            "status": effective,
            "scored_count": self._scored,
            "task_error_count": self._task_errors,
            "scorer_error_count": self._scorer_errors,
        }
        # The run's aggregate main score -> run.mainScore. Without it the UI shows the
        # main metric and the baseline delta as "-" (change = run.mainScore - baseline).
        if self._main_count:
            body["main_score"] = self._main_sum / self._main_count
        # NOTE: the resolved headline metric NAME is intentionally NOT sent here. The current
        # CompleteRunRequest schema has no main_score_name field AND rejects unknown keys, so
        # the late-bound name stays on the local result until the backend adds the field (see
        # the handoff). ``main_score_name`` is accepted for the eventual wire once it lands.
        _ = main_score_name
        self._request(
            "POST",
            f"/api/v1/public/evaluation-runs/{self.run_id}/complete",
            body,
        )
        # Join the backend's UI-relative run path with our host to form a clickable
        # link; None when the backend did not return one (older/self-hosted).
        # Prefer the backend's absolute run_url; fall back to host_url + run_path for a
        # control plane that predates run_url (keeps the same-origin behavior).
        url = self.run_url or (f"{self.host_url}{self.run_path}" if self.run_path else None)
        return UploadState(status="uploaded", dashboard_url=url)

    def publish_dataset(self, dataset_name: str, item_count: int) -> PublishResult:
        # Datasets are server/UI-owned; the SDK cannot create them. Stay local-only.
        return PublishResult(status="local_only", dataset_name=dataset_name, item_count=item_count)

    # --- mapping helpers ---
    def _scores_payload(self, item_result: EvalItemResult) -> list[dict]:
        payload: list[dict] = []
        for s in item_result.scores:
            # Declared version when present; sentinel where the backend requires a string.
            entry: dict[str, Any] = {
                "scorer_name": s.name,
                "scorer_version": s.version or _UNVERSIONED_SCORER,
            }
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
            payload.append(
                {"scorer_name": name, "scorer_version": _UNVERSIONED_SCORER, "error": msg}
            )
        return payload

    def _main_value(self, scores: list[Score]) -> float | None:
        """The run's main-metric value for one case, or None when unresolved.

        Name-agnostic for a single unconfigured scorer (its one numeric/boolean score);
        matched by ``main_score_name`` when an explicit main is set; None when multiple
        scorers have no configured main (genuinely no headline metric)."""
        if self._name_agnostic_main:
            for s in scores:
                v = _numeric_score(s.value)
                if v is not None:
                    return v
            return None
        if self.main_score_name is None:
            return None
        for s in scores:
            if s.name == self.main_score_name:
                return _numeric_score(s.value)  # numeric, or None for a categorical main
        return None

    def _status_and_main(self, item_result: EvalItemResult) -> tuple[str, float | None]:
        if item_result.error is not None:
            return "errored", None
        main = self._main_value(item_result.scores)
        if main is None:
            return "not_scored", None
        threshold = self._effective_threshold()
        direction = self._effective_direction()
        if direction == "lower_is_better":
            passed = main <= threshold
        elif direction == "none":
            return "not_scored", main
        else:  # higher_is_better (the default)
            passed = main >= threshold
        return ("passed" if passed else "failed"), main


def _dataset_from_version(snapshot: dict, name: str) -> Dataset:
    """Build a local Dataset pinned to the exact version described by ``snapshot``."""
    ds = Dataset(name=name)
    ds.dataset_id = snapshot.get("dataset_id")  # type: ignore[assignment]
    ds.dataset_version_id = snapshot.get("dataset_version_id")
    for item in snapshot.get("items", []):
        # Native JSON at the HTTP boundary: the backend already JSON-decodes
        # input/expected before returning them, so the SDK takes the values as-is.
        # Re-decoding here would double-decode a genuine JSON-looking string.
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


def pull_dataset(
    dataset_id: str,
    *,
    version_id: str | None = None,
    api_key: str | None = None,
    host_url: str | None = None,
) -> Dataset:
    """Fetch a platform dataset into a local :class:`Dataset`.

    Pull data, not runs: reproduce a run by pulling its ``dataset_version_id`` (see
    :func:`pull_dataset_version`) -- there is no ``pull_run``.

    Without ``version_id`` the dataset's CURRENT version is pulled. With ``version_id``
    that EXACT immutable version is pulled and validated to belong to ``dataset_id`` --
    the current version is never silently substituted, so a run can reproduce the precise
    version it recorded. The returned Dataset carries ``dataset_id`` / ``dataset_version_id``.
    """
    key, host = _resolve_credentials(api_key, host_url)
    if not key:
        raise ValueError(
            "pull_dataset needs an API key (initialize traceroot or pass api_key=...)."
        )
    host = host.rstrip("/")

    meta = _http_get_json(f"{host}/api/v1/public/datasets/{quote(dataset_id, safe='')}", key)
    name = meta.get("name", dataset_id)
    if version_id is None:
        version_id = meta["current_dataset_version_id"]
    return pull_dataset_version(
        version_id, dataset_id=dataset_id, name=name, api_key=key, host_url=host
    )


def pull_dataset_version(
    version_id: str,
    *,
    dataset_id: str | None = None,
    name: str | None = None,
    api_key: str | None = None,
    host_url: str | None = None,
) -> Dataset:
    """Fetch one EXACT immutable dataset version by its id.

    Pull data, not runs: this is how you reproduce a run -- pass the run's
    ``dataset_version_id`` to get the exact cases it scored, then supply your own task +
    scorers. There is no ``pull_run`` (a run is task + scorers + data; only the data is
    on the platform).

    When ``dataset_id`` is supplied, the returned version is validated to belong to it
    (a mismatch raises ``ValueError`` rather than silently returning the wrong data).
    A missing version surfaces the backend's 404 as a clear error.
    """
    key, host = _resolve_credentials(api_key, host_url)
    if not key:
        raise ValueError(
            "pull_dataset_version needs an API key (initialize traceroot or pass api_key=...)."
        )
    host = host.rstrip("/")

    try:
        snapshot = _http_get_json(
            f"{host}/api/v1/public/dataset-versions/{quote(version_id, safe='')}", key
        )
    except RuntimeError as exc:
        if " HTTP 404:" in str(exc):
            raise ValueError(f"dataset version {version_id!r} not found") from exc
        raise

    returned_dataset_id = snapshot.get("dataset_id")
    if dataset_id is not None and returned_dataset_id not in (None, dataset_id):
        raise ValueError(
            f"dataset version {version_id!r} belongs to dataset {returned_dataset_id!r}, "
            f"not {dataset_id!r}"
        )
    ds = _dataset_from_version(snapshot, name or returned_dataset_id or dataset_id or version_id)
    # Pin ids even if the snapshot omitted them.
    ds.dataset_version_id = ds.dataset_version_id or version_id
    ds.dataset_id = ds.dataset_id or dataset_id  # type: ignore[assignment]
    return ds

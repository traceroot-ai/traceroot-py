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

from traceroot.eval.results import EvalItemResult, UploadState, case_status as _case_status
from traceroot.eval.transport import PublishResult, RunHandle
from traceroot.eval.types import Dataset, EvalCase, Score
from traceroot.utils import serialize_value

_DEFAULT_PASS_THRESHOLD = 1.0

# The backend's public contract requires a NON-EMPTY scorer version string
# (ScorerRefSchema/ScoreInputSchema: z.string().min(1)). The SDK data model and the
# event/artifact path stays honest with version=None for an unversioned scorer;
# only at this reporting boundary do we map None -> an explicit "unversioned" sentinel
# so the request validates. This is a clear marker, not an invented version number.
_UNVERSIONED_SCORER = "unversioned"

# --- Contract caps (frontend/packages/core/src/eval-contract.ts) -------------
# The backend REJECTS an over-cap field, and a 400 on one result aborts the whole run. The
# SDK clamps at this serialization boundary instead, with an explicit marker so a consumer
# never mistakes a cap for real data. Keep these values (and the marker) identical to the
# TypeScript SDK's src/eval/platform.ts.
_TRUNCATION_SUFFIX = "...[truncated]"
_STRING_VALUE_MAX = 2_000
_EXPLANATION_MAX = 5_000
_SCORE_ERROR_MAX = 5_000
_TASK_ERROR_MAX = 10_000
_PAYLOAD_TEXT_MAX = 1_000_000
_SCORER_SOURCE_MAX = 50_000
_METADATA_MAX = 64_000


def _clamp(text: str | None, limit: int) -> str | None:
    """Bound one string field to ``limit`` characters, marking any truncation. The result
    is exactly ``limit`` characters long when it had to be cut."""
    if text is None or len(text) <= limit:
        return text
    return text[: limit - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX


def _clamp_metadata(value: Any, limit: int) -> Any:
    """Bound a free-form metadata object by its SERIALIZED size (the contract's rule).
    Over-cap metadata collapses to an explicit truncation marker rather than 400ing."""
    if value is None:
        return None
    try:
        text = json.dumps(serialize_value(value))
    except Exception:
        return value
    if len(text) <= limit:
        return value
    return {"truncated": True, "preview": _clamp(text, limit - 32)}


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
    # Serialize the body consistently (like _json_dumps) so valid but non-JSON-native SDK metadata
    # doesn't fail locally with a TypeError before the request is even sent.
    data = json.dumps(serialize_value(body)).encode() if body is not None else None
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
        # Per-case contribution keyed by test_case_id, so a retried case (or a repeated
        # upload() with the same transport) REPLACES its contribution instead of
        # double-counting the completion totals.
        self._contrib: dict[str, tuple[bool, int, bool]] = {}

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
            "scorers": self._scorer_refs(),
        }
        if self.dataset_version_id is not None:
            body["dataset_version_id"] = self.dataset_version_id
        # Idempotency key: prefer the one the caller drives with, else our own.
        effective_crun = client_run_id or self.client_run_id
        if effective_crun is not None:
            body["client_run_id"] = effective_crun
        # Free-form user metadata only; optional on the backend, so omit when empty to
        # match its absent-or-null rules rather than sending an empty object.
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
                    "key",
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
                # Contract-capped fields on the definition: a long captured source or a fat
                # metadata blob must not 400 the whole registration.
                if "source" in ref:
                    ref["source"] = _clamp(ref["source"], _SCORER_SOURCE_MAX)
                if "metadata" in ref:
                    ref["metadata"] = _clamp_metadata(ref["metadata"], _METADATA_MAX)
                refs.append(ref)
            return refs
        return [{"name": n, "version": _UNVERSIONED_SCORER} for n in self.scorer_names]

    def register_item(self, run: RunHandle, case: EvalCase) -> None:
        # The item->trace link is folded into the result upsert (contract), so no-op.
        return None

    def record_item_result(self, run: RunHandle, item_result: EvalItemResult) -> None:
        status = _case_status(item_result)
        self._request(
            "POST",
            f"/api/v1/public/evaluation-runs/{self.run_id}/results",
            {
                "test_case_id": item_result.case_id,
                "trace_id": item_result.trace_id,
                # The backend validates these as strings (z.string()); serialize
                # non-string values (dicts/numbers) to JSON text. `input` is a
                # REQUIRED string (not nullable) -> coerce a missing input to "".
                "input": _clamp(_as_text(item_result.input), _PAYLOAD_TEXT_MAX) or "",
                "expected_output": _clamp(_as_text(item_result.expected), _PAYLOAD_TEXT_MAX),
                "candidate_output": _clamp(_as_text(item_result.output), _PAYLOAD_TEXT_MAX),
                "status": status,
                "task_error": _clamp(item_result.error, _TASK_ERROR_MAX),
                # Total wall-clock for this case (task + its scorers); nonneg int ms, null
                # when unknown. Run duration is NOT summed from cases (they run concurrently).
                "duration_ms": _duration_ms(item_result.duration_ms),
                "scores": self._scores_payload(item_result),
            },
        )
        # Record this case's contribution only AFTER the upsert persisted, keyed by id so a
        # retry or repeated upload replaces (not adds to) the completion aggregate. A case
        # that produced at least one score (and no task error) counts toward scored_count;
        # the SDK derives no case-level pass/fail (every score carries its own `passed`).
        self._contrib[item_result.case_id] = (
            item_result.error is not None,
            len(item_result.scorer_errors),
            item_result.error is None and bool(item_result.scores),
        )

    def record_scores(self, run: RunHandle, case_id: str, scores: list[Score]) -> None:
        # Already sent inside record_item_result (which carries the full item).
        return None

    def _resolved_scorer_manifest(self, emitted: dict[str, list[str]]) -> list[dict]:
        """The registration scorer refs augmented with the metrics each DEFINITION actually emitted
        during the run. The platform keys a metric's policy on the EMITTED-metric name, so a
        definition whose function name differs from its emitted metric (``grade`` -> ``quality``)
        must declare that ownership here or the platform can't resolve the metric's threshold/
        direction. Each emitted metric carries the definition's declared policy (a scorer's
        declaration governs whatever metric it emits). Rebuilt from the SAME refs registration sent,
        so completion merges by definition name without dropping the definition's detail."""
        out: list[dict] = []
        for ref in self._scorer_refs():
            metrics = emitted.get(ref["name"])
            if metrics:
                policy = {
                    k: ref[k]
                    for k in ("value_type", "direction", "threshold")
                    if ref.get(k) is not None
                }
                ref = {**ref, "emitted_metrics": [{"name": m, **policy} for m in metrics]}
            out.append(ref)
        return out

    def finish_run(
        self,
        run: RunHandle,
        status: str | None = None,
        emitted_metrics: dict[str, list[str]] | None = None,
    ) -> UploadState:
        # Pure reporter: the engine passes the terminal ``status`` (else derived from error counts).
        # Derive the aggregate from the per-case map so the totals always match the DISTINCT cases
        # actually persisted (no inflation from retries/replays).
        task_errors = sum(1 for task_error, _, _ in self._contrib.values() if task_error)
        scorer_errors = sum(n for _, n, _ in self._contrib.values())
        scored = sum(1 for _, _, was_scored in self._contrib.values() if was_scored)
        effective = status or (
            "completed_with_errors" if (task_errors or scorer_errors) else "completed"
        )
        body: dict[str, Any] = {
            "status": effective,
            "scored_count": scored,
            "task_error_count": task_errors,
            "scorer_error_count": scorer_errors,
        }
        # The RESOLVED scorer->emitted-metric manifest, discovered during execution. The platform
        # merges it (by definition name) into the stored manifest so each emitted metric's policy is
        # keyed on the metric name for reconciliation, read-back, and comparison. Additive.
        if emitted_metrics:
            manifest = self._resolved_scorer_manifest(emitted_metrics)
            if manifest:
                body["scorers"] = manifest
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
                entry["string_value"] = _clamp(str(v), _STRING_VALUE_MAX)
            if s.comment is not None:
                entry["explanation"] = _clamp(s.comment, _EXPLANATION_MAX)
            passed = self._score_passed(s, single_emission=len(item_result.scores) == 1)
            if passed is not None:
                entry["passed"] = passed
            payload.append(entry)
        # A failing scorer is a score with an error and null value (never 0). Use the scorer's
        # DECLARED version (from the manifest) so an errored versioned scorer isn't misattributed
        # to 'unversioned'.
        for name, msg in item_result.scorer_errors.items():
            payload.append(
                {
                    "scorer_name": name,
                    "scorer_version": self._declared_version(name),
                    "error": _clamp(msg, _SCORE_ERROR_MAX),
                }
            )
        return payload

    def _declared_version(self, name: str) -> str:
        """Declared manifest version for a scorer name, or the sentinel when none was declared."""
        for spec in self.scorer_specs or []:
            if spec.get("name") == name:
                return spec.get("version") or _UNVERSIONED_SCORER
        return _UNVERSIONED_SCORER

    def _score_policy(
        self, name: str | None, *, single_emission: bool
    ) -> tuple[float | None, str] | None:
        """(threshold, direction) for ONE emitted metric, or None when it can't be resolved
        without guessing. A single scorer that emitted a SINGLE metric owns it even when the
        function name differs from the emitted Score name (name-agnostic). The moment that
        scorer emits a metric MAP, its one declared threshold cannot own every metric, so the
        emitted name must match the declaration -- otherwise a `{threshold: 0.9}` scorer would
        stamp `passed` on an unrelated `latency_ms: 120`. With multiple scorers the same
        name-match rule applies. An unmatched metric returns None so the platform is told
        'unknown', never a fabricated pass/fail."""
        specs = self.scorer_specs or []
        owner = (
            specs[0]
            if len(specs) == 1 and single_emission
            else next((s for s in specs if s.get("name") == name), None)
        )
        if owner is None:
            return None
        # RAW declared threshold (may be None): a numeric metric with no declared threshold is
        # scored but gets no fabricated pass/fail. Direction still defaults for a declared metric.
        threshold = owner.get("threshold")
        direction = owner.get("direction")
        return (threshold, str(direction) if direction is not None else "higher_is_better")

    def _score_passed(self, score: Score, *, single_emission: bool = True) -> bool | None:
        """SDK-computed pass/fail for one emitted metric, derived at serialization time and never
        stored on the Score. Boolean: true = pass. Numeric: compared against the OWNING scorer's
        threshold+direction (the same policy that decides the case status, so a single scorer's
        main score and its per-score `passed` always agree). Categorical values, a 'none'-direction
        metric, or a numeric metric whose policy can't be resolved have no pass/fail -> None (the
        SDK does not guess; the platform stores null)."""
        v = score.value
        if isinstance(v, bool):
            return v
        n = _numeric_score(v)
        if n is None:
            return None
        policy = self._score_policy(score.name, single_emission=single_emission)
        if policy is None:
            return None
        threshold, direction = policy
        if threshold is None or direction == "none":
            return (
                None  # no declared threshold (or no ordering) -> scored, but no fabricated verdict
            )
        if direction == "lower_is_better":
            return n <= threshold
        return n >= threshold  # higher_is_better (the default)


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

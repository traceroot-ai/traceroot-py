"""Low-level evaluation run lifecycle (DS-3).

A ``RunSession`` is the explicit, retry-safe lifecycle for distributed, async, or
custom harnesses: start a run, pre-register cases, record results, attach traces
and scores out of order, then complete/fail/cancel. It is built on the
``EvalTransport`` seam (so ``LocalTransport`` / ``FakeTransport`` / ``PlatformTransport``
all work), and the high-level runner uses this same layer internally -- one code path.

Partial updates (``attach_trace`` / ``score``) merge into the session's last known
item and re-record the FULL item, so they never clobber previously reported output
or scores even against a replace-on-upsert backend.
"""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Sequence
from typing import Any

from traceroot.eval.results import EvalItemResult, UploadState
from traceroot.eval.transport import EvalTransport, RunHandle
from traceroot.eval.types import EvalCase, Score


class RunSession:
    def __init__(
        self,
        transport: EvalTransport,
        *,
        name: str,
        dataset_name: str = "<inline>",
        dataset_ref: Any = None,
        candidate_version: str | None = None,
        environment: str = "evaluation",
        client_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.transport = transport
        self.name = name
        self.dataset_name = dataset_name
        self.dataset_ref = dataset_ref
        self.candidate_version = candidate_version
        self.environment = environment
        # Stable idempotency key so retried start() calls resolve to the same run.
        self.client_run_id = client_run_id or f"crun_{uuid.uuid4().hex}"
        self.metadata = metadata
        self._run: RunHandle | None = None
        self._items: dict[str, EvalItemResult] = {}

    # --- lifecycle ---
    def start(self) -> RunSession:
        self._run = self.transport.create_run(
            name=self.name,
            dataset_name=self.dataset_name,
            metadata=self.metadata,
            client_run_id=self.client_run_id,
        )
        return self

    def _require_run(self) -> RunHandle:
        if self._run is None:
            raise RuntimeError("RunSession.start() must be called before recording")
        return self._run

    def register(self, case: EvalCase) -> None:
        """Pre-register a case before its output exists (status not_scored upstream)."""
        run = self._require_run()
        self._items[case.id] = EvalItemResult(  # type: ignore[index]
            case_id=case.id,  # type: ignore[arg-type]
            input=case.input,
            output=None,
            expected=case.expected,
            scores=[],
            scorer_errors={},
            error=None,
            trace_id=None,
        )
        self.transport.register_item(run, case)

    def record(self, item_result: EvalItemResult) -> None:
        """Record (upsert) a full item result and its scores."""
        run = self._require_run()
        self._items[item_result.case_id] = item_result
        self.transport.record_item_result(run, item_result)
        self.transport.record_scores(run, item_result.case_id, item_result.scores)

    def attach_trace(self, case_id: str, trace_id: str) -> None:
        """Attach a trace id to a (possibly already recorded) item, keeping its scores."""
        run = self._require_run()
        merged = dataclasses.replace(self._item_or_empty(case_id), trace_id=trace_id)
        self._items[case_id] = merged
        self.transport.record_item_result(run, merged)

    def score(self, case_id: str, scores: Sequence[Score]) -> None:
        """Add scores to an item (e.g. delayed/human), merging with existing scores."""
        run = self._require_run()
        cur = self._item_or_empty(case_id)
        merged = dataclasses.replace(cur, scores=[*cur.scores, *scores])
        self._items[case_id] = merged
        self.transport.record_item_result(run, merged)
        self.transport.record_scores(run, case_id, merged.scores)

    def complete(self) -> UploadState:
        return self.transport.finish_run(self._require_run(), status=None)

    def fail(self, reason: str | None = None) -> UploadState:
        return self.transport.finish_run(self._require_run(), status="failed")

    def cancel(self) -> UploadState:
        # 'cancelled' is not a backend run status yet (see contract delta B1);
        # map to 'incomplete' until the server adds it.
        return self.transport.finish_run(self._require_run(), status="incomplete")

    # --- access ---
    def item(self, case_id: str) -> EvalItemResult | None:
        return self._items.get(case_id)

    def _item_or_empty(self, case_id: str) -> EvalItemResult:
        return self._items.get(case_id) or EvalItemResult(
            case_id=case_id,
            input=None,
            output=None,
            expected=None,
            scores=[],
            scorer_errors={},
            error=None,
            trace_id=None,
        )

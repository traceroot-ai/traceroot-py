"""Transport seam for offline evaluation.

A narrow internal protocol for platform persistence. Evaluation is cloud-only: every
run reports to the platform through a ``PlatformTransport``. This module defines the
seam plus a recording ``FakeTransport`` for deterministic tests (a stand-in cloud
transport that records calls instead of making HTTP requests).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol, runtime_checkable

from traceroot.eval.results import EvalItemResult, UploadState
from traceroot.eval.types import EvalCase, Score


@dataclasses.dataclass
class RunHandle:
    """Opaque handle for one evaluation run, returned by ``create_run``."""

    name: str
    dataset_name: str
    metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class PublishResult:
    """Result of ``Dataset.publish``."""

    status: str
    dataset_name: str
    item_count: int


@runtime_checkable
class EvalTransport(Protocol):
    """Persistence seam. Implementations must never fabricate remote URLs."""

    def create_run(
        self,
        name: str,
        dataset_name: str,
        metadata: dict[str, Any] | None,
        client_run_id: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> RunHandle: ...

    def register_item(self, run: RunHandle, case: EvalCase) -> None: ...

    def record_item_result(self, run: RunHandle, item_result: EvalItemResult) -> None: ...

    def record_scores(self, run: RunHandle, case_id: str, scores: list[Score]) -> None: ...

    def finish_run(
        self,
        run: RunHandle,
        status: str | None = None,
        main_score_name: str | None = None,
        emitted_metrics: dict[str, list[str]] | None = None,
    ) -> UploadState: ...


class FakeTransport:
    """Records every call in order for deterministic tests -- a stand-in cloud transport
    (spans export as on a reported run; no real HTTP)."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def create_run(
        self,
        name: str,
        dataset_name: str,
        metadata: dict[str, Any] | None,
        client_run_id: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> RunHandle:
        self.calls.append(("create_run", name, dataset_name, client_run_id))
        self.last_run_metadata = metadata
        self.last_run_provenance = provenance
        return RunHandle(name=name, dataset_name=dataset_name, metadata=metadata)

    def register_item(self, run: RunHandle, case: EvalCase) -> None:
        self.calls.append(("register_item", case.id))

    def record_item_result(self, run: RunHandle, item_result: EvalItemResult) -> None:
        self.calls.append(("record_item_result", item_result.case_id))

    def record_scores(self, run: RunHandle, case_id: str, scores: list[Score]) -> None:
        self.calls.append(("record_scores", case_id))

    def finish_run(
        self,
        run: RunHandle,
        status: str | None = None,
        main_score_name: str | None = None,
        emitted_metrics: dict[str, list[str]] | None = None,
    ) -> UploadState:
        self.calls.append(("finish_run", status))
        self.last_main_score_name = main_score_name
        self.last_emitted_metrics = emitted_metrics
        return UploadState(status="uploaded", dashboard_url=None)

    def publish_dataset(self, dataset_name: str, item_count: int) -> PublishResult:
        self.calls.append(("publish_dataset", dataset_name))
        return PublishResult(status="uploaded", dataset_name=dataset_name, item_count=item_count)

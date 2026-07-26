"""Transport seam for offline evaluation (OE-5).

A narrow internal protocol for eventual platform persistence - STRUCTURE ONLY,
no HTTP. Ships a no-op ``LocalTransport`` (the default) and a recording
``FakeTransport`` for deterministic tests. No real network client exists until a
backend OpenAPI contract lands; URLs are never fabricated. See design spec
section 6.
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
    """Result of ``Dataset.publish`` - explicit about local-only state."""

    status: str
    dataset_name: str
    item_count: int


@runtime_checkable
class EvalTransport(Protocol):
    """Persistence seam. Implementations must never fabricate remote URLs."""

    def create_run(
        self, name: str, dataset_name: str, metadata: dict[str, Any] | None
    ) -> RunHandle: ...

    def register_item(self, run: RunHandle, case: EvalCase) -> None: ...

    def record_item_result(self, run: RunHandle, item_result: EvalItemResult) -> None: ...

    def record_scores(self, run: RunHandle, case_id: str, scores: list[Score]) -> None: ...

    def finish_run(self, run: RunHandle) -> UploadState: ...


class LocalTransport:
    """Default no-op transport. Everything stays local; nothing is uploaded."""

    def create_run(
        self, name: str, dataset_name: str, metadata: dict[str, Any] | None
    ) -> RunHandle:
        return RunHandle(name=name, dataset_name=dataset_name, metadata=metadata)

    def register_item(self, run: RunHandle, case: EvalCase) -> None:
        return None

    def record_item_result(self, run: RunHandle, item_result: EvalItemResult) -> None:
        return None

    def record_scores(self, run: RunHandle, case_id: str, scores: list[Score]) -> None:
        return None

    def finish_run(self, run: RunHandle) -> UploadState:
        return UploadState(status="local_only", dashboard_url=None)

    def publish_dataset(self, dataset_name: str, item_count: int) -> PublishResult:
        return PublishResult(status="local_only", dataset_name=dataset_name, item_count=item_count)


class FakeTransport:
    """Records every call in order for deterministic tests. Local-only."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def create_run(
        self, name: str, dataset_name: str, metadata: dict[str, Any] | None
    ) -> RunHandle:
        self.calls.append(("create_run", name, dataset_name))
        return RunHandle(name=name, dataset_name=dataset_name, metadata=metadata)

    def register_item(self, run: RunHandle, case: EvalCase) -> None:
        self.calls.append(("register_item", case.id))

    def record_item_result(self, run: RunHandle, item_result: EvalItemResult) -> None:
        self.calls.append(("record_item_result", item_result.case_id))

    def record_scores(self, run: RunHandle, case_id: str, scores: list[Score]) -> None:
        self.calls.append(("record_scores", case_id))

    def finish_run(self, run: RunHandle) -> UploadState:
        self.calls.append(("finish_run",))
        return UploadState(status="local_only", dashboard_url=None)

    def publish_dataset(self, dataset_name: str, item_count: int) -> PublishResult:
        self.calls.append(("publish_dataset", dataset_name))
        return PublishResult(status="local_only", dataset_name=dataset_name, item_count=item_count)

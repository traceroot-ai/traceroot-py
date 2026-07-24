"""Offline evaluation SDK for TraceRoot.

Local-first, trace-native evaluation: dataset / test cases -> execute a task ->
score each result -> structured results (and evaluation traces).

See ``offline-eval/design-spec-offline-eval-sdk.md``.
"""

import importlib
from typing import TYPE_CHECKING

# Public symbol -> submodule. Resolved lazily (PEP 562) so importing the package
# never eagerly pulls every submodule; a symbol loads from its module on first use.
_EXPORTS = {
    "DatasetConflictError": "dataset_sync",
    "FakeDatasetSync": "dataset_sync",
    "LocalDatasetSync": "dataset_sync",
    "PlatformDatasetSync": "dataset_sync",
    "PushResult": "dataset_sync",
    "Evaluation": "evaluation",
    "evaluate": "evaluation",
    "evaluate_async": "evaluation",
    "PlatformTransport": "platform",
    "pull_dataset": "platform",
    "pull_dataset_version": "platform",
    "collect_run_provenance": "provenance",
    "EvalItemResult": "results",
    "EvalRunResult": "results",
    "RunDatasetRef": "results",
    "ScoreSummary": "results",
    "UploadState": "results",
    "aggregate_scores": "results",
    "describe_scorers": "scorers",
    "llm_judge": "scorers",
    "scorer": "scorers",
    "dataset_latest_snippet": "snippets",
    "dataset_version_snippet": "snippets",
    "reproduce_run_snippet": "snippets",
    "EvalTransport": "transport",
    "FakeTransport": "transport",
    "PublishResult": "transport",
    "RunHandle": "transport",
    "Dataset": "types",
    "DatasetSnapshot": "types",
    "DeferredScore": "types",
    "EvalCase": "types",
    "Score": "types",
    "ScorerContext": "types",
}

if TYPE_CHECKING:
    from traceroot.eval.dataset_sync import (
        DatasetConflictError,
        FakeDatasetSync,
        LocalDatasetSync,
        PlatformDatasetSync,
        PushResult,
    )
    from traceroot.eval.evaluation import Evaluation, evaluate, evaluate_async
    from traceroot.eval.platform import PlatformTransport, pull_dataset, pull_dataset_version
    from traceroot.eval.provenance import collect_run_provenance
    from traceroot.eval.results import (
        EvalItemResult,
        EvalRunResult,
        RunDatasetRef,
        ScoreSummary,
        UploadState,
        aggregate_scores,
    )
    from traceroot.eval.scorers import describe_scorers, llm_judge, scorer
    from traceroot.eval.snippets import (
        dataset_latest_snippet,
        dataset_version_snippet,
        reproduce_run_snippet,
    )
    from traceroot.eval.transport import EvalTransport, FakeTransport, PublishResult, RunHandle
    from traceroot.eval.types import (
        Dataset,
        DatasetSnapshot,
        DeferredScore,
        EvalCase,
        Score,
        ScorerContext,
    )


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"traceroot.eval.{module}"), name)
    globals()[name] = value
    return value


# --- CLI runner compatibility handshake (see offline-eval/cli-architecture-2026-07-20.md) ---
# Bump __api_version__ on any breaking change to: the runner event protocol, the
# side-effect-free/network-free Evaluation construction guarantee, or the artifact schema.
__api_version__ = 1


def capabilities() -> dict[str, bool]:
    """Feature flags the CLI runner negotiates against. Stable keys only."""
    return {
        "snapshot": True,
        "run_session": True,
        "compare": True,
        "dataset_push": True,
        "sampling": True,
        "provenance": True,
        "cancellation": True,
    }


__all__ = [
    "Dataset",
    "DatasetSnapshot",
    "EvalCase",
    "Score",
    "ScorerContext",
    "DeferredScore",
    "__api_version__",
    "capabilities",
    "Evaluation",
    "evaluate",
    "evaluate_async",
    "EvalItemResult",
    "EvalRunResult",
    "RunDatasetRef",
    "ScoreSummary",
    "UploadState",
    "aggregate_scores",
    "EvalTransport",
    "FakeTransport",
    "RunHandle",
    "PublishResult",
    "PlatformTransport",
    "pull_dataset",
    "pull_dataset_version",
    "scorer",
    "llm_judge",
    "describe_scorers",
    "dataset_latest_snippet",
    "dataset_version_snippet",
    "reproduce_run_snippet",
    "collect_run_provenance",
    "PushResult",
    "DatasetConflictError",
    "LocalDatasetSync",
    "FakeDatasetSync",
    "PlatformDatasetSync",
]

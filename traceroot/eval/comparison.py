"""Baseline comparison for offline evaluation (DS-5).

Local convenience for offline/console use. Counts are SCORE-CELL counts: one cell per
(test_case_id, scorer), matched by stable test_case_id -- improvements / regressions /
unchanged cells, with unpaired cells preserved explicitly. These are NOT regressed
test-CASE counts (a case with two scorers contributes two cells). A clean aggregate delta
requires a compatible identity: the same scorer set and the same dataset revision.

Ownership note: this comparison is computed locally and is never uploaded. The SDK reports
only raw outcomes; the backend derives authoritative candidate-vs-baseline labels. The
only comparison linkage the SDK sends is ``baseline_run_id``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from traceroot.eval.results import EvalRunResult


@dataclasses.dataclass(frozen=True)
class CaseDelta:
    case_id: str
    score_name: str
    candidate: float | None
    baseline: float | None
    delta: float | None
    direction: str  # improved | regressed | unchanged

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Comparison:
    compatible: bool
    improvements: list[CaseDelta]
    regressions: list[CaseDelta]
    unchanged: list[CaseDelta]
    unpaired: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "improvements": [d.to_dict() for d in self.improvements],
            "regressions": [d.to_dict() for d in self.regressions],
            "unchanged": [d.to_dict() for d in self.unchanged],
            "unpaired": self.unpaired,
        }

    def summary(self) -> str:
        flag = "" if self.compatible else " (INCOMPATIBLE: differing scorers/dataset revision)"
        return (
            f"Comparison{flag} (score cells): {len(self.improvements)} improved, "
            f"{len(self.regressions)} regressed, {len(self.unchanged)} unchanged, "
            f"{len(self.unpaired)} unpaired"
        )


def _numeric_index(run: EvalRunResult) -> dict[tuple[str, str], float]:
    idx: dict[tuple[str, str], float] = {}
    for item in run.item_results:
        for s in item.scores:
            if isinstance(s.value, bool):
                idx[(item.case_id, s.name)] = 1.0 if s.value else 0.0
            elif isinstance(s.value, (int, float)):
                idx[(item.case_id, s.name)] = float(s.value)
    return idx


def compare_runs(candidate: EvalRunResult, baseline: EvalRunResult) -> Comparison:
    cand_idx = _numeric_index(candidate)
    base_idx = _numeric_index(baseline)

    same_scorers = set(candidate.score_summary) == set(baseline.score_summary)
    same_revision = bool(
        candidate.dataset
        and baseline.dataset
        and candidate.dataset.revision == baseline.dataset.revision
    )
    compatible = same_scorers and same_revision

    improvements: list[CaseDelta] = []
    regressions: list[CaseDelta] = []
    unchanged: list[CaseDelta] = []
    for key in sorted(set(cand_idx) & set(base_idx)):
        case_id, score_name = key
        c, b = cand_idx[key], base_idx[key]
        delta = c - b
        direction = "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged"
        row = CaseDelta(case_id, score_name, c, b, delta, direction)
        (improvements if delta > 0 else regressions if delta < 0 else unchanged).append(row)

    cand_cases = {it.case_id for it in candidate.item_results}
    base_cases = {it.case_id for it in baseline.item_results}
    unpaired = sorted(cand_cases ^ base_cases)

    return Comparison(compatible, improvements, regressions, unchanged, unpaired)

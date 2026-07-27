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


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _pct(value: float | None) -> str:
    # Zero-padded to 5 chars so columns align (e.g. "05.20%", "85.00%").
    return f"{value * 100:05.2f}%" if value is not None else "  n/a "


def _delta(diff: float | None) -> str:
    if diff is None:
        return "  —   "
    if diff == 0:
        return "  —   "
    sign = "+" if diff > 0 else "-"
    return f"{sign}{abs(diff) * 100:05.2f}%"


def format_comparison_report(
    candidate: EvalRunResult,
    baseline: EvalRunResult,
    *,
    comparison: Comparison | None = None,
    url: str | None = None,
) -> str:
    """Render a Braintrust-style per-scorer candidate-vs-baseline block.

    One line per numeric scorer: candidate mean, the delta of means vs the baseline,
    and the per-case improvement/regression cell counts. Local presentation only.
    """
    comp = comparison or compare_runs(candidate, baseline)

    imp_by_name: dict[str, int] = {}
    reg_by_name: dict[str, int] = {}
    for d in comp.improvements:
        imp_by_name[d.score_name] = imp_by_name.get(d.score_name, 0) + 1
    for d in comp.regressions:
        reg_by_name[d.score_name] = reg_by_name.get(d.score_name, 0) + 1

    names = [n for n, s in candidate.score_summary.items() if s.mean is not None]
    width = max((len(f"'{n}'") for n in names), default=0)

    lines: list[str] = []
    for name in names:
        c_mean = candidate.score_summary[name].mean
        b = baseline.score_summary.get(name)
        b_mean = b.mean if b is not None else None
        diff = (c_mean - b_mean) if (c_mean is not None and b_mean is not None) else None
        label = f"'{name}'".ljust(width)
        counts = f"({_plural(imp_by_name.get(name, 0), 'improvement')}, {_plural(reg_by_name.get(name, 0), 'regression')})"
        lines.append(f"  {_pct(c_mean)} ({_delta(diff)}) {label}  {counts}")

    header = "=" * 20 + " COMPARISON " + "=" * 20
    title = f"{candidate.name}  vs  {baseline.name} [baseline]:"
    flag = (
        ""
        if comp.compatible
        else "\n  (INCOMPATIBLE: differing scorers/dataset revision — deltas may be unaligned)"
    )
    body = "\n".join(lines) if lines else "  (no numeric scorers to compare)"
    tail = f"\n\n  → {url}" if url else ""
    return f"{header}\n{title}{flag}\n{body}{tail}"

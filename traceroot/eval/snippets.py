"""Canonical copy-paste starter snippets for the offline-eval SDK.

The SDK owns these so any surface that shows "how to bring this local" (the UI dataset
page, the run/evaluation page, docs) templates against the real SDK signatures instead of
hand-writing strings that drift.

Mental model (identical for datasets and runs): bringing something local always means
pulling DATA -- a dataset at a specific version.

    dataset, latest   -> pull_dataset(dataset_id)
    dataset, pinned   -> pull_dataset_version(version_id)
    reproduce a run   -> pull_dataset_version(run.dataset_version_id)   # same primitive

A run is task + scorers + data; task and scorers live in the user's code, so a run is not
re-runnable from an id -- the platform can only return the dataset the run used. There is
intentionally no pull_run / pull_evaluation.
"""

from __future__ import annotations

LANGUAGES = ("python", "typescript")

_DATASET_ID_PLACEHOLDER = "<dataset_id>"
_VERSION_ID_PLACEHOLDER = "<dataset_version_id>"


def _q(value: str | None, placeholder: str) -> str:
    """A double-quoted id literal, or the placeholder token (also quoted) when unknown."""
    return f'"{value if value is not None else placeholder}"'


def _check_lang(lang: str) -> None:
    if lang not in LANGUAGES:
        raise ValueError(f"lang must be one of {LANGUAGES}, got {lang!r}")


def dataset_latest_snippet(dataset_id: str | None = None, *, lang: str = "python") -> str:
    """Snippet that pulls a dataset's CURRENT published version."""
    _check_lang(lang)
    ds = _q(dataset_id, _DATASET_ID_PLACEHOLDER)
    if lang == "python":
        return (
            "from traceroot import pull_dataset\n\n"
            f"dataset = pull_dataset({ds})  # current published version\n"
        )
    return (
        'import { pullDataset } from "traceroot";\n\n'
        f"const dataset = await pullDataset({ds});  // current published version\n"
    )


def dataset_version_snippet(dataset_version_id: str | None = None, *, lang: str = "python") -> str:
    """Snippet that pulls one EXACT immutable dataset version."""
    _check_lang(lang)
    vid = _q(dataset_version_id, _VERSION_ID_PLACEHOLDER)
    if lang == "python":
        return (
            "from traceroot import pull_dataset_version\n\n"
            f"dataset = pull_dataset_version({vid})  # exact immutable version\n"
        )
    return (
        'import { pullDatasetVersion } from "traceroot";\n\n'
        f"const dataset = await pullDatasetVersion({vid});  // exact immutable version\n"
    )


def reproduce_run_snippet(dataset_version_id: str | None = None, *, lang: str = "python") -> str:
    """Snippet that reproduces a run: pull the exact dataset version it used, then supply
    your own task + scorers (a run is task + scorers + data; only the data is on the
    platform). The evaluate(...) call is commented so the user fills in their code."""
    _check_lang(lang)
    vid = _q(dataset_version_id, _VERSION_ID_PLACEHOLDER)
    if lang == "python":
        return (
            "from traceroot import evaluate, pull_dataset_version\n\n"
            "# Reproduce a run: pull the EXACT dataset version it scored, then supply your\n"
            "# own task + scorers. A run is task + scorers + data; only the data lives on\n"
            "# the platform -- you bring the code (there is no run-pull primitive).\n"
            f"dataset = pull_dataset_version({vid})\n\n"
            "# run = evaluate(\n"
            '#     name="<evaluation name>",\n'
            "#     dataset=dataset,\n"
            "#     task=your_task,            # your candidate function\n"
            "#     scorers=[your_scorer],     # your scorer callables\n"
            '#     candidate_version="<label>",\n'
            "# )\n"
        )
    return (
        'import { evaluate, pullDatasetVersion } from "traceroot";\n\n'
        "// Reproduce a run: pull the EXACT dataset version it scored, then supply your\n"
        "// own task + scorers. A run is task + scorers + data; only the data lives on\n"
        "// the platform -- you bring the code (there is no run-pull primitive).\n"
        f"const dataset = await pullDatasetVersion({vid});\n\n"
        "// const run = await evaluate({\n"
        '//   name: "<evaluation name>",\n'
        "//   dataset,\n"
        "//   task: yourTask,            // your candidate function\n"
        "//   scorers: [yourScorer],     // your scorer callables\n"
        '//   candidateVersion: "<label>",\n'
        "// });\n"
    )

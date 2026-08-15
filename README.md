# TraceRoot Python SDK

[![Y Combinator][y-combinator-image]][y-combinator-url]
[![License][license-image]][license-url]
[![X (Twitter)][twitter-image]][twitter-url]
[![Discord][discord-image]][discord-url]
[![Documentation][docs-image]][docs-url]
[![PyPI Version][pypi-version-image]][pypi-version-url]
[![PyPI SDK Downloads][pypi-sdk-downloads-image]][pypi-sdk-downloads-url]

# Documentation

Please see the [Python SDK Docs](https://traceroot.ai/docs/tracing/get-started) for details.

# Offline Evaluation

Run your AI app over a dataset, score each case with your own scorers, and (optionally) report the
results to TraceRoot to compare candidates over time. A runnable end-to-end example lives in
[`examples/offline_eval.py`](examples/offline_eval.py) — it runs offline via an in-memory transport.

`evaluate()` is **cloud-first**: by default it reports to TraceRoot and therefore needs
`TRACEROOT_API_KEY` and a **pulled** dataset (a purely local `Dataset` cannot start a reported run).
For a self-contained run with no backend, pass an in-memory transport (below).

The task, scorer, and reporting behavior are identical in both modes — only the dataset source and
the transport differ.

## Quickstart — local run (no backend)

```python
from traceroot import Dataset, ScorerContext, evaluate
from traceroot.eval import scorer
from traceroot.eval.transport import FakeTransport  # in-memory stand-in for the platform

# 1. A local dataset — add(input, *, expected=..., id=...). (No cloud dataset is created.)
dataset = Dataset("support-routing")
dataset.add(input={"message": "I was charged twice"}, expected={"route": "billing"})
dataset.add(input={"message": "the app keeps crashing"}, expected={"route": "technical"})

# 2. Your app (the "task"): one case input -> one candidate output.
def route_ticket(ticket: dict) -> dict:
    msg = ticket["message"].lower()
    return {"route": "billing" if "charge" in msg else "technical" if "crash" in msg else "general"}

# 3. A scorer. @scorer declares the metric's comparison policy; it receives a ScorerContext
#    (input / output / expected / metadata) and returns a value, bool, Score, or list of Scores.
@scorer(value_type="numeric", direction="higher_is_better", threshold=1.0)
def accuracy(ctx: ScorerContext) -> float:
    return 1.0 if ctx.output["route"] == ctx.expected["route"] else 0.0

# 4. Run it locally — FakeTransport keeps everything in-memory (nothing is uploaded).
run = evaluate(name="routing-v1", dataset=dataset, task=route_ticket, scorers=[accuracy],
               transport=FakeTransport())
print(run.summary())
```

## Quickstart — cloud run (reports to TraceRoot)

Create the dataset in the **TraceRoot UI**, then pull it and run — the run appears in the platform.

```python
import traceroot
from traceroot import ScorerContext, evaluate
from traceroot.eval import scorer

# Needs TRACEROOT_API_KEY. Pull the dataset you created in the UI (this is the synced dataset a
# reported run requires); it arrives with its dataset_id/dataset_version_id already set.
dataset = traceroot.pull_dataset("<dataset-id>")

@scorer(value_type="numeric", direction="higher_is_better", threshold=1.0)
def accuracy(ctx: ScorerContext) -> float:
    return 1.0 if ctx.output["route"] == ctx.expected["route"] else 0.0

def route_ticket(ticket: dict) -> dict:
    msg = ticket["message"].lower()
    return {"route": "billing" if "charge" in msg else "technical" if "crash" in msg else "general"}

run = evaluate(name="routing-v1", dataset=dataset, task=route_ticket, scorers=[accuracy],
               candidate_version="v1")           # -> a real run visible in the TraceRoot UI
print(run.upload.dashboard_url)                    # link to the run in the platform
```

## Datasets

Cases are added locally with `add(input, *, expected=..., id=...)` or `upsert(EvalCase(...))`:

```python
from traceroot import Dataset, EvalCase

ds = Dataset("support-routing")
ds.add(input={"message": "hi"}, id="c1", expected={"route": "general"})
ds.upsert(EvalCase(input={"message": "bye"}, id="c2", expected={"route": "general"}))
```

**Datasets are created and versioned in the TraceRoot UI, not from the SDK.** `ds.push()` returns
`status="local_only"` and does **not** create a platform dataset. To run a reported evaluation, create
the dataset in the UI and pull it:

```python
import traceroot
pulled = traceroot.pull_dataset("<dataset-id>")                              # current version
exact  = traceroot.pull_dataset("<dataset-id>", version_id="<version-id>")   # a pinned version
```

Pulling a dataset stamps its `dataset_id`/`dataset_version_id` onto the returned `Dataset`, so a run
against it is pinned to the exact cases it scored (reproducible).

## Scorers

A **deterministic** scorer is any function of the `ScorerContext`. `@scorer` declares its metric
policy; the emitted `Score` name may differ from the function name (a `grade` function may emit a
`quality` metric):

```python
from traceroot import Score

@scorer(value_type="numeric", direction="lower_is_better", threshold=0.2)
def grade(ctx: ScorerContext) -> Score:
    return Score("quality", latency_seconds(ctx.output))   # emits the metric "quality"
```

An **LLM judge** — `model` + `messages` are its reported definition; `{{output}}` etc. interpolate
the case. Provide `complete=...` to stub the model (deterministic/offline), or omit it to call the
real model:

```python
from traceroot.eval import llm_judge

tone = llm_judge(
    name="tone", model="claude-sonnet-5",
    messages=[{"role": "user", "content": "Rate politeness 0..1:\n{{output}}"}],
    output_type="score",
)
```

## Main score, threshold, direction

- Each scorer declares its own **`threshold`** and **`direction`** (`higher_is_better` /
  `lower_is_better`); that policy decides whether the metric passes for a case.
- The **main score** is the headline metric that sets each case's pass/fail. With a **single**
  scorer it is inferred from the one metric emitted — you don't name it, even if the function name
  differs from the emitted metric name. With **multiple** numeric metrics, name it explicitly:
  `evaluate(..., main_score="accuracy")`; otherwise the run fails loudly rather than guess.

## Reporting, transports, and failure behavior

`evaluate` is cloud-first: it reports to TraceRoot using `TRACEROOT_API_KEY` and a synced dataset.
The **default transport** is built for you from those. Pass an **explicit transport** to control
reporting directly — both behave identically for scoring:

```python
from traceroot.eval.platform import PlatformTransport
t = PlatformTransport("<dataset-id>", candidate_version="v2")
run = evaluate(name="routing-v2", dataset=pulled, task=route_ticket, scorers=[accuracy], transport=t)
```

Honest statuses, never faked: a **task** error → `errored`; a **scorer** error is isolated to that
scorer (other scorers on the case still record); a case with no numeric/boolean score →
`not_scored` (never a silent 0).

## Candidate vs. baseline

Tag each run with `candidate_version` (e.g. `"v1"`, `"gpt-4o"`); the SDK reports only the candidate
and never a baseline linkage — **comparison is done in the TraceRoot UI**, where you pick the
baseline run to diff against. Re-run the same dataset with a new `candidate_version` to compare:

```python
evaluate(name="routing", dataset=pulled, task=route_ticket, scorers=[accuracy], candidate_version="v1")
evaluate(name="routing", dataset=pulled, task=route_v2,     scorers=[accuracy], candidate_version="v2")
```

See the [platform evaluation docs](https://traceroot.ai/docs) for the UI: run tables, comparison,
and the scorer catalog.

<!-- Links -->

[discord-image]: https://img.shields.io/discord/1395844148568920114?logo=discord&labelColor=%235462eb&logoColor=%23f5f5f5&color=%235462eb
[discord-url]: https://discord.gg/tPyffEZvvJ
[docs-image]: https://img.shields.io/badge/docs-traceroot.ai-0dbf43
[docs-url]: https://traceroot.ai/docs/tracing/get-started
[license-image]: https://img.shields.io/badge/License-Apache%202.0-blue.svg
[license-url]: https://opensource.org/licenses/Apache-2.0
[pypi-version-image]: https://img.shields.io/pypi/v/traceroot?label=traceroot&labelColor=0dbf43&color=grey
[pypi-version-url]: https://pypi.org/project/traceroot/
[pypi-sdk-downloads-image]: https://static.pepy.tech/badge/traceroot
[pypi-sdk-downloads-url]: https://pypi.python.org/pypi/traceroot
[twitter-image]: https://img.shields.io/twitter/follow/TracerootAI
[twitter-url]: https://x.com/TracerootAI
[y-combinator-image]: https://img.shields.io/badge/Combinator-S25-orange?logo=ycombinator&labelColor=white
[y-combinator-url]: https://www.ycombinator.com/companies/traceroot-ai

"""Offline Evaluation — runnable end-to-end example (Python).

Evaluation is cloud-first: a real run pulls a dataset and reports results to the TraceRoot platform
(see "Reporting to the platform" in the README):

    import traceroot
    dataset = traceroot.pull_dataset("<dataset-id>")        # needs TRACEROOT_API_KEY
    run = evaluate(name="...", dataset=dataset, task=..., scorers=[...])

So it runs fully OFFLINE against the wheel with no backend, this example passes an in-memory
`FakeTransport` (the stand-in transport the SDK's own tests use) in place of the platform. Everything
else — datasets, scorers, the ScorerContext, main-score resolution, the four honest statuses — is
exactly what a reported run does.

    python examples/offline_eval.py
"""

from traceroot import Dataset, Score, ScorerContext, evaluate
from traceroot.eval import llm_judge, scorer
from traceroot.eval.transport import FakeTransport


# --- 1. The app under test (your "task"): input case -> candidate output --------------------
def route_ticket(ticket: dict) -> dict:
    msg = ticket["message"].lower()
    if "refund" in msg or "charge" in msg:
        return {"route": "billing"}
    if "crash" in msg or "error" in msg:
        return {"route": "technical"}
    return {"route": "general"}


# --- 2. A dataset: cases with input + (optional) expected reference --------------------------
dataset = Dataset("support-routing")
dataset.add(input={"message": "I was charged twice"}, expected={"route": "billing"})
dataset.add(input={"message": "the app keeps crashing"}, expected={"route": "technical"})
dataset.add(input={"message": "hello there"}, expected={"route": "general"})


# --- 3. Scorers ------------------------------------------------------------------------------
# A deterministic scorer. `@scorer` declares the metric's policy: how its value is compared
# (direction) and the pass threshold. It receives a ScorerContext (input / output / expected /
# metadata) and returns a value, a bool, a Score, or a list of Scores.
@scorer(value_type="numeric", direction="higher_is_better", threshold=1.0)
def accuracy(ctx: ScorerContext) -> float:
    return 1.0 if ctx.output["route"] == ctx.expected["route"] else 0.0


# A scorer may emit a metric whose NAME differs from the function name, and may attach a comment.
@scorer(value_type="boolean")
def is_confident(ctx: ScorerContext) -> Score:
    return Score("confident", ctx.output["route"] != "general", comment="non-general route")


# An LLM judge. `complete` is stubbed here so the example is deterministic and offline; drop it to
# call the real model named by `model`. The judge is reported by its definition (model + messages).
tone = llm_judge(
    name="tone",
    model="claude-sonnet-5",
    messages=[{"role": "user", "content": "Rate politeness 0..1 for:\n{{output}}"}],
    output_type="score",
    complete=lambda model, messages: "0.9",  # remove to hit the real model
)


# --- 4. Run the evaluation ------------------------------------------------------------------
# `main_score` names the headline metric when several numeric metrics exist; with a single scorer it
# is inferred from the one metric it emits, so you can omit it. Swap `transport=FakeTransport()` for a
# pulled dataset + TRACEROOT_API_KEY to report the identical run to the platform.
run = evaluate(
    name="support-routing-v1",
    dataset=dataset,
    task=route_ticket,
    scorers=[accuracy, is_confident, tone],
    main_score="accuracy",  # the metric that decides pass/fail per case
    candidate_version="v1",
    transport=FakeTransport(),  # in-memory stand-in so the example needs no backend
)

# --- 5. Inspect results ---------------------------------------------------------------------
print(run.summary())  # a formatted report: pass/fail/errored/not_scored counts + per-metric means
for item in run.item_results:
    scores = ", ".join(f"{s.name}={s.value!r}" for s in item.scores)
    print(f"  {item.case_id}: [{scores}]")

"""``evaluation_key``: the stable identity a run is grouped by, separate from its display name.

The backend groups runs by ``evaluation_key`` and falls back to ``evaluation_name`` when the SDK
sends none -- which makes the display name the identity, so renaming an evaluation forks its
history, and a Python and a TypeScript run of the SAME evaluation only group if their names happen
to match character for character. The SDK therefore always sends a key (defaulting to the name, so
nothing changes for a caller who never sets one) and lets it be set explicitly. Same design as a
scorer's ``key``. Parity with traceroot-ts/tests/eval-evaluation-key.test.ts.
"""

from __future__ import annotations

from traceroot.eval import Dataset, EvalCase, evaluate
from traceroot.eval.platform import PlatformTransport


class _Recording(PlatformTransport):
    """PlatformTransport with the HTTP seam replaced by a recorder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests: list[tuple] = []

    def _request(self, method, path, body=None):
        self.requests.append((method, path, body))
        if path == "/api/v1/public/evaluation-runs":
            return {"evaluation_run_id": "run_1"}
        return {}

    @property
    def register_body(self) -> dict:
        return next(b for _m, p, b in self.requests if p == "/api/v1/public/evaluation-runs")


def _transport(**kwargs) -> _Recording:
    return _Recording("ds_1", api_key="tr-x", host_url="https://h", **kwargs)


def _ds() -> Dataset:
    d = Dataset(name="d")
    d.upsert(EvalCase(input=1, id="c0", expected=1))
    d.dataset_id = "ds_1"
    d.dataset_version_id = "dsv_1"
    return d


class TestRegistrationAlwaysCarriesAKey:
    def test_it_defaults_to_the_evaluation_name(self):
        t = _transport()
        t.create_run("regression suite", "d", None)
        assert t.register_body["evaluation_key"] == "regression suite"
        # The display name is still sent; the key is identity, the name is presentation.
        assert t.register_body["evaluation_name"] == "regression suite"

    def test_an_explicit_key_overrides_the_name(self):
        t = _transport(evaluation_key="checkout-flow")
        t.create_run("Checkout Flow (nightly)", "d", None)
        assert t.register_body["evaluation_key"] == "checkout-flow"
        assert t.register_body["evaluation_name"] == "Checkout Flow (nightly)"

    def test_the_key_is_sent_verbatim_for_cross_language_grouping(self):
        """The value on the wire is the key as written -- no normalization, no language marker --
        so a Python and a TypeScript run under the same key land in the same group."""
        t = _transport(evaluation_key="checkout-flow")
        t.create_run("py runner", "d", None)
        assert t.register_body["evaluation_key"] == "checkout-flow"


class TestEvaluateThreadsTheKey:
    def test_evaluate_forwards_an_explicit_key_to_registration(self):
        t = _transport()
        evaluate(
            name="Nightly regression",
            data=_ds(),
            task=lambda x: x,
            scorers=[lambda ctx: 1.0],
            transport=t,
            evaluation_key="nightly-regression",
        )
        assert t.register_body["evaluation_key"] == "nightly-regression"

    def test_without_one_the_name_is_the_key(self):
        t = _transport()
        evaluate(
            name="Nightly regression",
            data=_ds(),
            task=lambda x: x,
            scorers=[lambda ctx: 1.0],
            transport=t,
        )
        assert t.register_body["evaluation_key"] == "Nightly regression"

    def test_a_key_set_on_the_transport_is_not_overwritten(self):
        t = _transport(evaluation_key="from-the-transport")
        evaluate(
            name="n",
            data=_ds(),
            task=lambda x: x,
            scorers=[lambda ctx: 1.0],
            transport=t,
            evaluation_key="from-the-call",
        )
        assert t.register_body["evaluation_key"] == "from-the-transport"

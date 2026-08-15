"""Shared eval-test fixtures.

Evaluation is cloud-only: a run always reports through a transport. Most tests exercise the
engine/scoring/dataset and don't care about the wire, so this defaults a bare ``evaluate()``
(no explicit ``transport=``) to a non-network ``FakeTransport`` stand-in. Tests that need the
real no-credentials -> raise path opt out with ``@pytest.mark.no_default_transport``.
"""

import pytest

try:
    from traceroot.eval import engine as _engine_mod
    from traceroot.eval import transport as _transport_mod
except ImportError:  # not every module is present on every rung of the stack
    _engine_mod = None
    _transport_mod = None


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "no_default_transport: run without the default FakeTransport (exercise cloud-only raise)",
    )


@pytest.fixture(autouse=True)
def _cloud_default_transport(request, monkeypatch):
    if _engine_mod is None or request.node.get_closest_marker("no_default_transport"):
        return
    monkeypatch.setattr(
        _engine_mod,
        "_auto_transport",
        lambda *a, **k: _transport_mod.FakeTransport(),
        raising=False,
    )

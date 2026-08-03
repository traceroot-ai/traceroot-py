"""Phase 4: truthful local/cloud boundary. Local-only enforcement drops credentials so the
engine can build no reporting transport -- a local run never silently uploads eval results
on ambient credentials. Parity with traceroot-ts/tests/eval-local-only.test.ts."""

import os

import pytest

import traceroot
from traceroot.eval.engine import _auto_transport
from traceroot.eval.platform import _resolve_credentials
from traceroot.eval.runner import _enforce_local_only


def _scorer(ctx):
    return 1.0


class _FakeClient:
    api_key = "tr-x"
    host_url = "https://h"
    git_repo = None
    git_ref = None


@pytest.mark.no_default_transport  # exercise the real _auto_transport, not the test fake
def test_local_only_drops_credentials_and_builds_no_transport(monkeypatch):
    monkeypatch.delenv("TRACEROOT_ENABLED", raising=False)
    # Ambient credentials present (monkeypatch reverts the client after the test).
    monkeypatch.setattr(traceroot, "_client", _FakeClient(), raising=False)
    try:
        assert _resolve_credentials(None, None)[0] == "tr-x"
        # With creds + a dataset id, the auto transport would report to the platform.
        assert _auto_transport([], [_scorer], "ds_1", None, "evaluation") is not None

        _enforce_local_only()

        # After local-only enforcement there are no credentials, so no reporting transport is
        # built (the engine then surfaces a clear cloud-only error, never a silent upload).
        assert _resolve_credentials(None, None)[0] == ""
        assert _auto_transport([], [_scorer], "ds_1", None, "evaluation") is None
    finally:
        os.environ.pop("TRACEROOT_ENABLED", None)

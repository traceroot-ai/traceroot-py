"""Cloud-only: evaluation with no credentials (and no explicit transport) raises.

Runs in a fresh subprocess so no ambient TraceRoot credentials/provider are present - the
true credential-free scenario. Evaluation reports to the platform, so without a way to
report it must fail loudly rather than silently run locally.
"""

import os
import subprocess
import sys

_SCRIPT = """
import traceroot
from traceroot.eval import Dataset, EvalCase, evaluate

ds = Dataset(name="d")
ds.upsert(EvalCase(input=2, id="c0", expected=2))


def task(x):
    return x


def exact(ctx):
    return 1.0 if ctx.output == ctx.expected else 0.0


try:
    evaluate(name="r", dataset=ds, task=task, scorers=[exact])
except RuntimeError as e:
    assert "reports to the TraceRoot platform" in str(e), str(e)
    print("CLOUD_ONLY_RAISED")
else:
    raise AssertionError("expected evaluate() to raise without credentials")
"""


def test_evaluate_without_credentials_raises():
    # Inherit the real env (so the venv resolves) but strip any TRACEROOT_* creds.
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRACEROOT_")}
    proc = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert "CLOUD_ONLY_RAISED" in proc.stdout

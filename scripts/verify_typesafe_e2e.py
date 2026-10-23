"""Run the opt-in local TypeSafe end-to-end suite. See examples/typesafe_e2e/README.md."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples/typesafe_e2e/run.py"),
        run_name="__main__",
    )

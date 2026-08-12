from __future__ import annotations

import sys

from main import run_pipeline_cli


if __name__ == "__main__":
    raise SystemExit(run_pipeline_cli(sys.argv[1:]))

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import run_chat_cli


if __name__ == "__main__":
    raise SystemExit(run_chat_cli(cwd=Path.cwd()))

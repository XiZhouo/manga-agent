from __future__ import annotations

import argparse
import json
import selectors
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, Any


DEFAULT_INPUT = Path("mangatest") / "Nightwing 124 (2025) (Pyrate-DCP).cbz"
DEFAULT_OUTPUT = Path("data/test_runs/nightwing_agent_stage2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live stage-2 agent/orchestrator validation on the Nightwing CBZ.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Path to the Nightwing CBZ.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT), help="Directory for logs and metadata.")
    parser.add_argument("--max-pages", type=int, default=1, help="Limit pages for the live end-to-end run.")
    parser.add_argument("--max-steps", type=int, default=0, help="Forwarded to main.py.")
    parser.add_argument("--pipeline-mode", type=str, default="vlm", choices=["vlm", "classic"], help="Forwarded to main.py.")
    parser.add_argument("--timeout-sec", type=int, default=900, help="Kill the run if it exceeds this timeout.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "main.py",
        "--input",
        str(Path(args.input)),
        "--pipeline-mode",
        args.pipeline_mode,
        "--target-language",
        "zh",
        "--max-pages",
        str(args.max_pages),
        "--output",
        "data/output",
    ]
    if args.max_steps > 0:
        cmd.extend(["--max-steps", str(args.max_steps)])

    (output_dir / "command.txt").write_text(" ".join(cmd), encoding="utf-8")
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    heartbeat_path = output_dir / "heartbeat.json"
    started_at = time.time()

    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stdout_tail = deque(maxlen=40)
    stderr_tail = deque(maxlen=40)
    timed_out = False

    with stdout_path.open("w", encoding="utf-8") as stdout_fh, stderr_path.open("w", encoding="utf-8") as stderr_fh:
        selector = selectors.DefaultSelector()
        assert proc.stdout is not None
        assert proc.stderr is not None
        selector.register(proc.stdout, selectors.EVENT_READ, ("stdout", stdout_fh, stdout_tail))
        selector.register(proc.stderr, selectors.EVENT_READ, ("stderr", stderr_fh, stderr_tail))

        while True:
            elapsed = time.time() - started_at
            heartbeat_path.write_text(
                json.dumps(
                    {
                        "running": proc.poll() is None,
                        "pid": proc.pid,
                        "elapsed_sec": round(elapsed, 3),
                        "timeout_sec": args.timeout_sec,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            if proc.poll() is not None and not selector.get_map():
                break

            if args.timeout_sec > 0 and elapsed > args.timeout_sec and proc.poll() is None:
                timed_out = True
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)

            events = selector.select(timeout=1.0)
            if not events:
                if proc.poll() is not None:
                    for fileobj in list(selector.get_map().values()):
                        selector.unregister(fileobj.fileobj)
                    break
                continue

            for key, _ in events:
                stream_name, handle, tail = key.data
                line = key.fileobj.readline()
                if line == "":
                    selector.unregister(key.fileobj)
                    continue
                handle.write(line)
                handle.flush()
                tail.append(line.rstrip("\n"))

    summary: Dict[str, Any] = {
        "returncode": proc.returncode,
        "input": str(args.input),
        "pipeline_mode": args.pipeline_mode,
        "max_pages": args.max_pages,
        "max_steps": args.max_steps,
        "timeout_sec": args.timeout_sec,
        "timed_out": timed_out,
        "elapsed_sec": round(time.time() - started_at, 3),
        "stdout_tail": list(stdout_tail)[-20:],
        "stderr_tail": list(stderr_tail)[-20:],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    raise SystemExit(proc.returncode if not timed_out else 124)


if __name__ == "__main__":
    main()

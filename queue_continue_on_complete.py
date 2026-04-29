#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time


ROOT = pathlib.Path(os.environ.get("RMT_OPTUNA_RUN", "./optuna_run"))


def load_state(queue_name: str) -> dict:
    path = ROOT / "model_queue_runs" / queue_name / "queue_state.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: queue_continue_on_complete.py PRIMARY_QUEUE CONT_QUEUE CONT_QUEUE_FILE", file=sys.stderr)
        return 2

    primary_queue, cont_queue, cont_file = sys.argv[1:]
    run_dir = ROOT / "model_queue_runs" / primary_queue
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / f"{cont_queue}.started"
    log_path = run_dir / f"{cont_queue}.continuation.log"

    while True:
        if marker.exists():
            return 0

        state = load_state(primary_queue)
        if state.get("state") == "complete":
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"starting {cont_queue} from {cont_file}\n")
            subprocess.Popen(
                [sys.executable, "-u", "start_model_queue.py", cont_queue, cont_file],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=open(run_dir / f"{cont_queue}.launcher.log", "ab", buffering=0),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            marker.write_text(str(time.time()), encoding="utf-8")
            return 0

        time.sleep(120)


if __name__ == "__main__":
    raise SystemExit(main())

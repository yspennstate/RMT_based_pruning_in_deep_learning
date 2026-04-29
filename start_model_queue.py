#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


OPTUNA_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a model queue runner under the heartbeat wrapper.")
    parser.add_argument("queue_name")
    parser.add_argument("queue_file")
    args = parser.parse_args()

    queue_dir = OPTUNA_ROOT / "model_queue_runs" / args.queue_name
    queue_dir.mkdir(parents=True, exist_ok=True)
    wrapper_log = open(queue_dir / "wrapper.log", "ab", buffering=0)
    command = [
        sys.executable,
        "-u",
        str(OPTUNA_ROOT / "removed_matrix_audit" / "remote_runner_wrapper.py"),
        "--status-path",
        str(queue_dir / "runner_status.json"),
        "--stdout-path",
        str(queue_dir / "stdout.log"),
        "--workdir",
        str(OPTUNA_ROOT),
        "--heartbeat-seconds",
        "30",
        "--",
        sys.executable,
        "-u",
        str(OPTUNA_ROOT / "model_queue_runner.py"),
        "--queue-name",
        args.queue_name,
        "--queue-file",
        str(Path(args.queue_file).resolve()),
    ]
    proc = subprocess.Popen(
        command,
        cwd=str(OPTUNA_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=wrapper_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(proc.pid)


if __name__ == "__main__":
    main()

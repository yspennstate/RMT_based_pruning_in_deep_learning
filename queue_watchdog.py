#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import time


ROOT = pathlib.Path(os.environ.get("RMT_OPTUNA_RUN", "./optuna_run"))
CHECK_SECONDS = 90
STALE_HEARTBEAT_SECONDS = 8 * 60


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def cmdline(pid: int) -> str:
    try:
        return (pathlib.Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
    except Exception:
        return ""


def matching_pids(queue_name: str) -> list[int]:
    me = os.getpid()
    parent = os.getppid()
    pids: list[int] = []
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in (me, parent):
            continue
        cmd = cmdline(pid)
        checks = (
            "model_queue_runner.py" in cmd and f"--queue-name {queue_name}" in cmd,
            "remote_runner_wrapper.py" in cmd and f"model_queue_runs/{queue_name}/runner_status.json" in cmd,
            "hybrid_mag20_then_v8_model_queue.py" in cmd and f"/{queue_name}" in cmd,
            "hybrid_mag_until_drop_then_v8_model_queue.py" in cmd and f"/{queue_name}" in cmd,
            "hybrid_mag_until_drop_then_v8_model.py" in cmd and f"/{queue_name}" in cmd,
            "run_finetune_magnitude_model_exec_queue.py" in cmd and f"/{queue_name}" in cmd,
            "run_removed_matrix_audit" in cmd and f"/{queue_name}" in cmd,
            "build_model_rmt_cache.py" in cmd and f"/{queue_name}" in cmd,
        )
        if any(checks):
            pids.append(pid)
    return pids


def supervisor_pids(queue_name: str) -> list[int]:
    me = os.getpid()
    parent = os.getppid()
    pids: list[int] = []
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in (me, parent):
            continue
        cmd = cmdline(pid)
        if "remote_suffix_supervisor.py" in cmd and f"--queue {queue_name}" in cmd:
            pids.append(pid)
    return pids


def kill_queue(queue_name: str) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in matching_pids(queue_name) + supervisor_pids(queue_name):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        time.sleep(5)


def start_queue(queue_name: str, queue_file: str, queue_dir: pathlib.Path) -> None:
    log = open(queue_dir / "watchdog_restarts.log", "ab", buffering=0)
    subprocess.Popen(
        [sys.executable, "-u", "start_model_queue.py", queue_name, queue_file],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: queue_watchdog.py QUEUE_NAME QUEUE_FILE", file=sys.stderr)
        return 2
    queue_name, queue_file = sys.argv[1], sys.argv[2]
    os.chdir(ROOT)
    queue_dir = ROOT / "model_queue_runs" / queue_name
    queue_dir.mkdir(parents=True, exist_ok=True)
    log_path = queue_dir / "queue_watchdog.log"

    while True:
        state = load_json(queue_dir / "queue_state.json")
        runner = load_json(queue_dir / "runner_status.json")
        state_name = state.get("state", "missing")
        runner_state = runner.get("state", "missing")
        heartbeat = float(runner.get("heartbeat_at") or 0.0)
        heartbeat_stale = bool(heartbeat and time.time() - heartbeat > STALE_HEARTBEAT_SECONDS)
        live_pids = matching_pids(queue_name)

        if state_name == "complete":
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(f"[{utc()}] complete; watchdog exiting\n")
            return 0

        should_restart = (
            state_name in {"failed", "missing"}
            or (runner_state != "running" and not live_pids)
            or heartbeat_stale
            or not live_pids
        )
        if should_restart:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(
                    f"[{utc()}] restarting state={state_name} runner={runner_state} "
                    f"heartbeat_stale={heartbeat_stale} live_pids={live_pids}\n"
                )
            kill_queue(queue_name)
            start_queue(queue_name, queue_file, queue_dir)
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(f"[{utc()}] restart launched\n")

        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())

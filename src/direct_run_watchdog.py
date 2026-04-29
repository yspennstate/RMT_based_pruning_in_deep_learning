#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time


ROOT = pathlib.Path(os.environ.get("RMT_OPTUNA_RUN", "./optuna_run"))
WRAPPER = ROOT / "removed_matrix_audit" / "remote_runner_wrapper.py"
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


def run_complete(
    status_path: pathlib.Path,
    results_path: pathlib.Path | None = None,
    checkpoint_path: pathlib.Path | None = None,
) -> bool:
    run_dir = status_path.parent
    results_path = results_path or (run_dir / "results.json")
    checkpoint_path = checkpoint_path or (run_dir / "checkpoints" / "keep_s70.pt")
    payload = load_json(results_path)
    steps = payload.get("steps", [])
    has_s70 = any(isinstance(step, dict) and abs(float(step.get("target_sparsity", -1.0)) - 0.70) < 1e-9 for step in steps)
    return has_s70 and checkpoint_path.exists()


def cmdline(pid: int) -> str:
    try:
        return (pathlib.Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
    except Exception:
        return ""


def matching_pids(status_path: pathlib.Path, command_terms: list[str]) -> list[int]:
    me = os.getpid()
    parent = os.getppid()
    pids: list[int] = []
    status_token = str(status_path)
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in (me, parent):
            continue
        cmd = cmdline(pid)
        if "direct_run_watchdog.py" in cmd:
            continue
        if status_token in cmd or all(term in cmd for term in command_terms):
            pids.append(pid)
    return pids


def kill_run(status_path: pathlib.Path, command_terms: list[str]) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in matching_pids(status_path, command_terms):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        time.sleep(5)


def start_run(status_path: pathlib.Path, stdout_path: pathlib.Path, command: list[str]) -> None:
    wrapper_log = open(status_path.parent / "direct_watchdog_restarts.log", "ab", buffering=0)
    subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(WRAPPER),
            "--status-path",
            str(status_path),
            "--stdout-path",
            str(stdout_path),
            "--workdir",
            str(ROOT),
            "--heartbeat-seconds",
            "30",
            "--",
            *command,
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=wrapper_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--stdout-path", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--complete-results-path")
    parser.add_argument("--complete-checkpoint-path")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("missing command")

    os.chdir(ROOT)
    status_path = pathlib.Path(args.status_path)
    stdout_path = pathlib.Path(args.stdout_path)
    command_terms = [pathlib.Path(command[1]).name if len(command) > 1 else command[0]]
    log_path = status_path.parent / f"{args.name}_direct_watchdog.log"
    complete_results_path = pathlib.Path(args.complete_results_path) if args.complete_results_path else None
    complete_checkpoint_path = pathlib.Path(args.complete_checkpoint_path) if args.complete_checkpoint_path else None

    while True:
        status = load_json(status_path)
        state = status.get("state", "missing")
        returncode = status.get("returncode", status.get("exit_code"))
        heartbeat = float(status.get("heartbeat_at") or 0.0)
        heartbeat_stale = bool(heartbeat and time.time() - heartbeat > STALE_HEARTBEAT_SECONDS)
        live = matching_pids(status_path, command_terms)

        if run_complete(status_path, complete_results_path, complete_checkpoint_path):
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(f"[{utc()}] complete; watchdog exiting\n")
            return 0
        if state in {"complete", "completed", "succeeded"} and returncode in {0, "0", None}:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(f"[{utc()}] wrapper reports complete; watchdog exiting\n")
            return 0

        if state != "running" or heartbeat_stale or not live:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(f"[{utc()}] restarting state={state} heartbeat_stale={heartbeat_stale} live_pids={live}\n")
            kill_run(status_path, command_terms)
            start_run(status_path, stdout_path, command)
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(f"[{utc()}] restart launched\n")

        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())

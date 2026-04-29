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


def read_json(path: pathlib.Path) -> dict:
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


def matching_primary_pids(queue_name: str) -> list[int]:
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
        if (
            ("model_queue_runner.py" in cmd and f"--queue-name {queue_name}" in cmd)
            or ("remote_runner_wrapper.py" in cmd and f"model_queue_runs/{queue_name}/runner_status.json" in cmd)
            or ("hybrid_mag20_then_v8_model_queue.py" in cmd and f"/{queue_name}" in cmd)
            or ("hybrid_mag_until_drop_then_v8_model_queue.py" in cmd and f"/{queue_name}" in cmd)
            or ("hybrid_mag_until_drop_then_v8_model.py" in cmd and f"/{queue_name}" in cmd)
            or ("run_finetune_magnitude_model_exec_queue.py" in cmd and f"/{queue_name}" in cmd)
            or ("run_removed_matrix_audit" in cmd and f"/{queue_name}" in cmd)
            or ("build_model_rmt_cache.py" in cmd and f"/{queue_name}" in cmd)
        ):
            pids.append(pid)
    return pids


def stop_primary_queue(queue_name: str) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in matching_primary_pids(queue_name):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        time.sleep(3)


def run_complete(run_dir: pathlib.Path) -> bool:
    results_path = run_dir / "results.json"
    checkpoint_path = run_dir / "checkpoints" / "keep_s70.pt"
    if not results_path.exists() or not checkpoint_path.exists():
        return False
    payload = read_json(results_path)
    for step in payload.get("steps", []):
        if isinstance(step, dict) and abs(float(step.get("target_sparsity", -1.0)) - 0.70) < 1e-9:
            return True
    return False


def start_continuation(primary_queue: str, cont_queue: str, cont_file: str) -> None:
    run_dir = ROOT / "model_queue_runs" / primary_queue
    marker = run_dir / f"{cont_queue}.started"
    if marker.exists():
        return
    launch_log = open(run_dir / f"{cont_queue}.launcher.log", "ab", buffering=0)
    subprocess.Popen(
        [sys.executable, "-u", "start_model_queue.py", cont_queue, cont_file],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=launch_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    marker.write_text(str(time.time()), encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: queue_switch_after_current.py PRIMARY_QUEUE CONT_QUEUE CONT_QUEUE_FILE", file=sys.stderr)
        return 2

    primary_queue, cont_queue, cont_file = sys.argv[1:]
    state_path = ROOT / "model_queue_runs" / primary_queue / "queue_state.json"
    initial = read_json(state_path)
    initial_index = initial.get("index")
    initial_model = initial.get("model_name")
    run_dir = pathlib.Path(initial.get("run_dir") or "")
    log_path = ROOT / "model_queue_runs" / primary_queue / f"{cont_queue}.switch.log"

    while True:
        state = read_json(state_path)
        current_index = state.get("index")
        current_model = state.get("model_name")
        state_name = state.get("state")

        moved_to_next = (
            state_name == "running_model"
            and initial_index is not None
            and (current_index != initial_index or current_model != initial_model)
        )
        primary_complete = state_name == "complete"
        current_complete = run_dir.is_absolute() and run_complete(run_dir)

        if moved_to_next or primary_complete or current_complete:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"switching: state={state_name} initial={initial_index}:{initial_model} "
                    f"current={current_index}:{current_model} current_complete={current_complete}\n"
                )
            stop_primary_queue(primary_queue)
            start_continuation(primary_queue, cont_queue, cont_file)
            return 0

        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())

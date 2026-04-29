#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import time


ROOT = pathlib.Path(__file__).resolve().parent


def load_json(path: pathlib.Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def cmdline(pid: int) -> str:
    try:
        return (pathlib.Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
    except Exception:
        return ""


def matching_pids(terms: list[str]) -> list[int]:
    pids: list[int] = []
    me = os.getpid()
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        cmd = cmdline(pid)
        if cmd.startswith("bash -c ") and "remote_suffix_supervisor.py" in cmd:
            continue
        if all(term in cmd for term in terms):
            pids.append(pid)
    return pids


def gpu() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0].strip()
    except Exception:
        return "unavailable"


def tail(path: pathlib.Path) -> tuple[int | None, str]:
    try:
        age = int(time.time() - path.stat().st_mtime)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return age, lines[-1] if lines else ""
    except Exception:
        return None, ""


def latest_result(run_dir: pathlib.Path) -> str:
    payload = load_json(run_dir / "results.json")
    steps = payload.get("steps", [])
    if not steps:
        return ""
    step = steps[-1]
    if not isinstance(step, dict):
        return ""
    target = step.get("target_sparsity", "")
    top1 = step.get("keep_post_ft_top1", step.get("top1", ""))
    return f"s={target} top1={top1}"


def checkpoint_count(run_dir: pathlib.Path) -> int:
    return len(list((run_dir / "checkpoints").glob("keep_s*.pt")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue")
    parser.add_argument("--direct-deit", action="store_true")
    args = parser.parse_args()

    if args.direct_deit:
        name = "direct_deit"
        run_dir = ROOT / "randomness_audit_results_deit_hybrid" / "full_run_2026_04_26_deit_tiny_exact_magnitude_to20_then_v8"
        log_path = run_dir / "stdout.suffix_resume.log"
        supervisor_terms = ["remote_suffix_supervisor.py", "--direct-deit"]
        live_terms_options = [["run_removed_matrix_audit", str(run_dir)]]
        state = "direct"
    else:
        name = str(args.queue)
        state_path = ROOT / "model_queue_runs" / name / "queue_state.json"
        state_payload = load_json(state_path)
        state = str(state_payload.get("state", "missing"))
        run_dir = pathlib.Path(str(state_payload.get("run_dir", "")))
        log_path = ROOT / "model_queue_runs" / name / "stdout.suffix_resume.log"
        if name == "queue_b":
            log_path = ROOT / "model_queue_runs" / name / "stdout.log"
        supervisor_terms = ["remote_suffix_supervisor.py", "--queue", name]
        live_terms_options = [
            ["run_removed_matrix_audit", str(run_dir)],
            ["hybrid_mag20_then_v8_model_queue.py", f"/{name}"],
            ["model_queue_runner.py", "--queue-name", name],
        ]

    live: list[int] = []
    for terms in live_terms_options:
        live.extend(matching_pids(terms))
    live = sorted(set(live))
    supervisors = matching_pids(supervisor_terms)
    age, last = tail(log_path)
    payload = {
        "name": name,
        "state": state,
        "gpu": gpu(),
        "supervisors": supervisors,
        "live": live,
        "log_age_sec": age,
        "last_log": last[-220:],
        "latest_result": latest_result(run_dir),
        "checkpoint_count": checkpoint_count(run_dir),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

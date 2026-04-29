#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parent
CHECK_SECONDS = 120
FULL_SPARSITIES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: pathlib.Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def result_targets(run_dir: pathlib.Path) -> set[float]:
    payload = load_json(run_dir / "results.json")
    steps = payload.get("steps", [])
    out: set[float] = set()
    for step in steps:
        if isinstance(step, dict) and "target_sparsity" in step:
            try:
                out.add(round(float(step["target_sparsity"]), 2))
            except Exception:
                pass
    return out


def checkpoint_targets(run_dir: pathlib.Path) -> set[float]:
    out: set[float] = set()
    for path in (run_dir / "checkpoints").glob("keep_s*.pt"):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size <= 1024 * 1024:
            continue
        stem = path.stem
        if stem.startswith("keep_s"):
            try:
                out.add(round(int(stem.removeprefix("keep_s")) / 100.0, 2))
            except Exception:
                pass
    return out


def run_complete(run_dir: pathlib.Path) -> bool:
    return 0.70 in result_targets(run_dir) and 0.70 in checkpoint_targets(run_dir)


def run_started(run_dir: pathlib.Path) -> bool:
    return bool(result_targets(run_dir) or checkpoint_targets(run_dir))


def cmdline(pid: int) -> str:
    try:
        return (pathlib.Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
    except Exception:
        return ""


def live_pids(queue_name: str | None, run_dir: pathlib.Path | None) -> list[int]:
    me = os.getpid()
    parent = os.getppid()
    pids: list[int] = []
    run_token = str(run_dir) if run_dir else ""
    queue_token = f"/{queue_name}" if queue_name else ""
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in (me, parent):
            continue
        cmd = cmdline(pid)
        if "remote_suffix_supervisor.py" in cmd:
            continue
        checks = [
            bool(run_token and run_token in cmd and "run_removed_matrix_audit" in cmd),
            bool(queue_name and "model_queue_runner.py" in cmd and f"--queue-name {queue_name}" in cmd),
            bool(queue_name and "remote_runner_wrapper.py" in cmd and f"model_queue_runs/{queue_name}/runner_status.json" in cmd),
            bool(queue_token and "hybrid_mag20_then_v8_model_queue.py" in cmd and queue_token in cmd),
            bool(queue_token and "hybrid_mag_until_drop_then_v8_model_queue.py" in cmd and queue_token in cmd),
            bool(queue_token and "hybrid_mag_until_drop_then_v8_model.py" in cmd and queue_token in cmd),
            bool(queue_token and "run_finetune_magnitude_model_exec_queue.py" in cmd and queue_token in cmd),
            bool(queue_token and "run_removed_matrix_audit" in cmd and queue_token in cmd),
            bool(queue_token and "build_model_rmt_cache.py" in cmd and queue_token in cmd),
        ]
        if any(checks):
            pids.append(pid)
    return pids


def append_log(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc()}] {text}\n")


def launch_suffix(queue_name: str | None, queue_file: str | None, direct_deit: bool, log_path: pathlib.Path) -> None:
    cmd = [sys.executable, "-u", str(ROOT / "remote_resume_suffix.py")]
    if direct_deit:
        cmd.append("--direct-deit")
    else:
        cmd.extend(["--queue", str(queue_name)])
        if queue_file:
            cmd.extend(["--queue-file", queue_file])
    out = open(log_path.with_name(log_path.stem + ".resume.log"), "ab", buffering=0)
    subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    append_log(log_path, f"launched suffix resume: {' '.join(cmd)}")


def launch_queue_runner(queue_name: str, queue_file: str, log_path: pathlib.Path) -> None:
    env = os.environ.copy()
    env["RMT_SKIP_PRE_FT_EVAL"] = "1"
    out = open(ROOT / "model_queue_runs" / queue_name / "stdout.supervisor_queue_runner.log", "ab", buffering=0)
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "model_queue_runner.py"),
        "--queue-name",
        queue_name,
        "--queue-file",
        str(pathlib.Path(queue_file).resolve()),
        "--skip-deps",
    ]
    subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    append_log(log_path, f"launched queue runner: {' '.join(cmd)}")


def launch_queue_supervisor(queue_name: str, queue_file: str, log_path: pathlib.Path) -> None:
    out_dir = ROOT / "model_queue_runs" / queue_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = open(out_dir / "remote_suffix_supervisor.wrapper.log", "ab", buffering=0)
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "remote_suffix_supervisor.py"),
        "--queue",
        queue_name,
        "--queue-file",
        queue_file,
    ]
    subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    append_log(log_path, f"launched queue supervisor: {' '.join(cmd)}")


def direct_run_dir() -> pathlib.Path:
    return ROOT / "randomness_audit_results_deit_hybrid" / "full_run_2026_04_26_deit_tiny_exact_magnitude_to20_then_v8"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue")
    parser.add_argument("--queue-file")
    parser.add_argument("--direct-deit", action="store_true")
    parser.add_argument("--after-queue")
    args = parser.parse_args()
    if bool(args.queue) == args.direct_deit:
        raise SystemExit("provide exactly one of --queue or --direct-deit")

    queue_name = args.queue
    log_dir = ROOT / "model_queue_runs" / (queue_name or "direct_deit")
    log_path = log_dir / "remote_suffix_supervisor.log"
    append_log(log_path, "started")

    while True:
        if args.direct_deit:
            run_dir = direct_run_dir()
            if run_complete(run_dir):
                if args.after_queue and args.queue_file:
                    launch_queue_supervisor(args.after_queue, args.queue_file, log_path)
                    append_log(log_path, f"direct run complete; handed off to {args.after_queue}")
                    return 0
                append_log(log_path, "direct run complete; exiting")
                return 0
            live = live_pids(None, run_dir)
            if not live and run_started(run_dir):
                launch_suffix(None, None, True, log_path)
            time.sleep(CHECK_SECONDS)
            continue

        state_path = ROOT / "model_queue_runs" / str(queue_name) / "queue_state.json"
        state = load_json(state_path)
        if state.get("state") == "complete":
            append_log(log_path, "queue complete; exiting")
            return 0
        run_dir_raw = state.get("run_dir")
        run_dir = pathlib.Path(str(run_dir_raw)) if run_dir_raw else None
        live = live_pids(str(queue_name), run_dir)
        if live:
            time.sleep(CHECK_SECONDS)
            continue

        if run_dir and run_complete(run_dir):
            if not args.queue_file:
                append_log(log_path, "current run complete but no queue file was supplied")
            else:
                launch_queue_runner(str(queue_name), args.queue_file, log_path)
            time.sleep(CHECK_SECONDS)
            continue

        if run_dir and run_started(run_dir):
            launch_suffix(str(queue_name), args.queue_file, False, log_path)
        elif args.queue_file:
            launch_queue_runner(str(queue_name), args.queue_file, log_path)
        else:
            append_log(log_path, "no live process and insufficient state to launch")
        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())

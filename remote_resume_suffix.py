#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parent
FULL_SPARSITIES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def arg_value(argv: list[str], flag: str, default: str) -> str:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def model_slug(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-")
    if not slug:
        raise ValueError(f"bad model name: {model_name!r}")
    return slug


def result_targets(run_dir: pathlib.Path) -> set[float]:
    path = run_dir / "results.json"
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    steps = payload.get("steps", []) if isinstance(payload, dict) else []
    out: set[float] = set()
    for step in steps:
        if isinstance(step, dict) and "target_sparsity" in step:
            out.add(round(float(step["target_sparsity"]), 2))
    return out


def checkpoint_targets(run_dir: pathlib.Path) -> set[float]:
    out: set[float] = set()
    checkpoint_dir = run_dir / "checkpoints"
    for path in checkpoint_dir.glob("keep_s*.pt"):
        match = re.fullmatch(r"keep_s(\d+)\.pt", path.name)
        if match and path.stat().st_size > 1024 * 1024:
            out.add(round(int(match.group(1)) / 100.0, 2))
    return out


def live_for_run(run_dir: pathlib.Path) -> list[int]:
    token = str(run_dir)
    pids: list[int] = []
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
        except Exception:
            continue
        if token in cmd and "run_removed_matrix_audit" in cmd:
            pids.append(pid)
    return pids


def load_queue_item(queue_file: str | None, state: dict[str, object]) -> dict[str, object]:
    if not queue_file:
        return {}
    path = pathlib.Path(queue_file)
    if not path.is_absolute():
        path = ROOT / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, list):
        return {}
    index = state.get("index")
    if isinstance(index, int) and 0 <= index < len(payload) and isinstance(payload[index], dict):
        return payload[index]
    model_name = str(state.get("model_name") or "")
    run_name = str(state.get("run_name") or "")
    for item in payload:
        if not isinstance(item, dict):
            continue
        if item.get("model_name") == model_name:
            if not run_name or item.get("run_name") in (None, run_name):
                return item
    return {}


def params_from_item(item: dict[str, object]) -> dict[str, str]:
    batch_size_val = str(item.get("batch_size_val", "256"))
    return {
        "batch_size_train": str(item.get("batch_size_train", "128")),
        "batch_size_val": batch_size_val,
        "probe_batch_size": str(item.get("probe_batch_size", batch_size_val)),
        "num_workers": str(item.get("num_workers", "4")),
        "prefetch_factor": str(item.get("prefetch_factor", "2")),
        "phase_a_min_batches": str(item.get("phase_a_min_batches_per_epoch", "128")),
    }


def queue_config(queue_name: str, queue_file: str | None = None) -> tuple[str, pathlib.Path, pathlib.Path, dict[str, str], pathlib.Path]:
    state_path = ROOT / "model_queue_runs" / queue_name / "queue_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    argv = state.get("cmd") or []
    if argv:
        model_name = arg_value(argv, "--model-name", str(state["model_name"]))
        run_dir = pathlib.Path(str(state["run_dir"]))
        cache_root = pathlib.Path(arg_value(argv, "--cache-root", str(ROOT / "rmt_cache_models")))
        params = {
            "batch_size_train": arg_value(argv, "--batch-size-train", "128"),
            "batch_size_val": arg_value(argv, "--batch-size-val", "256"),
            "probe_batch_size": arg_value(argv, "--probe-batch-size", arg_value(argv, "--batch-size-val", "256")),
            "num_workers": arg_value(argv, "--num-workers", "4"),
            "prefetch_factor": arg_value(argv, "--prefetch-factor", "2"),
            "phase_a_min_batches": arg_value(argv, "--phase-a-min-batches-per-epoch", "128"),
        }
    else:
        item = load_queue_item(queue_file, state)
        model_name = str(state.get("model_name") or item.get("model_name"))
        run_dir = pathlib.Path(str(state.get("run_dir") or (ROOT / "randomness_audit_results_model_queue" / queue_name / str(state.get("run_name")))))
        cache_root = ROOT / "rmt_cache_models"
        params = params_from_item(item)
    return model_name, run_dir, cache_root / model_slug(model_name), params, ROOT / "model_queue_runs" / queue_name / "stdout.suffix_resume.log"


def direct_deit_config() -> tuple[str, pathlib.Path, pathlib.Path, dict[str, str], pathlib.Path]:
    model_name = "deit_tiny_patch16_224"
    run_dir = ROOT / "randomness_audit_results_deit_hybrid" / "full_run_2026_04_26_deit_tiny_exact_magnitude_to20_then_v8"
    params = {
        "batch_size_train": "128",
        "batch_size_val": "256",
        "probe_batch_size": "256",
        "num_workers": "4",
        "prefetch_factor": "2",
        "phase_a_min_batches": "128",
    }
    return model_name, run_dir, ROOT / "rmt_cache_models" / model_name, params, run_dir / "stdout.suffix_resume.log"


def build_command(model_name: str, run_dir: pathlib.Path, cache_dir: pathlib.Path, params: dict[str, str], sparsities: list[float]) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(ROOT / "run_removed_matrix_audit_v8_model_exec.py"),
        "--model-name-override",
        model_name,
        "--cache-dir-override",
        str(cache_dir),
        "--optuna-root",
        str(ROOT),
        "--resume-run-dir",
        str(run_dir),
        "--sparsities",
        ",".join(f"{s:.2f}" for s in sparsities),
        "--batch-size-train",
        params["batch_size_train"],
        "--batch-size-val",
        params["batch_size_val"],
        "--probe-batch-size",
        params["probe_batch_size"],
        "--num-workers",
        params["num_workers"],
        "--prefetch-factor",
        params["prefetch_factor"],
        "--train-probe-batches",
        "4",
        "--val-batches",
        "1000",
        "--keep-ft-epochs",
        "1.0",
        "--keep-ft-lr",
        "5e-5",
        "--keep-ft-optimizer",
        "sgd",
        "--prune-weight-mode",
        "paper_rmt",
        "--phase-b-schedule",
        "linear",
        "--phase-b-linear-start-multiplier",
        "1.0",
        "--phase-b-linear-end-multiplier",
        "2.0",
        "--phase-a-epochs",
        "1",
        "--phase-a-batch-multiplier",
        "2",
        "--phase-a-batch-budget-mode",
        "trainable_fraction",
        "--phase-a-min-batches-per-epoch",
        params["phase_a_min_batches"],
        "--phase-a-lr-multiplier",
        "8.0",
        "--phase-a-min-lr-ratio",
        "0.25",
        "--phase-a-optimizer",
        "adamw",
        "--selection-logit-l2-weight",
        "1.0",
        "--reinsert-budget-mode",
        "absolute_total_fraction",
        "--reinsert-total-fraction",
        "0.03",
        "--final-sparsity-accounting",
        "post_reinsert_exact",
        "--reinsert-rank-mode",
        "rmt_magnitude",
        "--save-checkpoints",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue")
    parser.add_argument("--queue-file")
    parser.add_argument("--direct-deit", action="store_true")
    args = parser.parse_args()
    if bool(args.queue) == args.direct_deit:
        raise SystemExit("provide exactly one of --queue or --direct-deit")

    model_name, run_dir, cache_dir, params, stdout_path = direct_deit_config() if args.direct_deit else queue_config(args.queue, args.queue_file)
    results = result_targets(run_dir)
    checkpoints = checkpoint_targets(run_dir)
    completed = sorted(results & checkpoints) if results and checkpoints else sorted(results or checkpoints)
    if not completed:
        raise SystemExit(f"no completed checkpoint/result pair found in {run_dir}")
    last = max(completed)
    remaining = [s for s in FULL_SPARSITIES if s > last + 1e-9]
    if not remaining:
        print(f"{model_name}: already complete through s={last:.2f}")
        return 0
    live = live_for_run(run_dir)
    if live:
        print(f"{model_name}: already running for {run_dir}: {live}")
        return 0

    command = build_command(model_name, run_dir, cache_dir, params, remaining)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    launch_record = stdout_path.with_suffix(stdout_path.suffix + ".launch.json")
    launch_record.write_text(
        json.dumps(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model_name": model_name,
                "run_dir": str(run_dir),
                "completed": completed,
                "remaining": remaining,
                "cmd": command,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["RMT_SKIP_PRE_FT_EVAL"] = "1"
    out = open(stdout_path, "ab", buffering=0)
    proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    pid_path = stdout_path.with_suffix(stdout_path.suffix + ".pid")
    pid_path.write_text(str(proc.pid) + "\n", encoding="utf-8")
    print(f"{model_name}: launched pid={proc.pid} from s={last:.2f}; remaining={','.join(f'{s:.2f}' for s in remaining)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

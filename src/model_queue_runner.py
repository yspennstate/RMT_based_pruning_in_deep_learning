#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


OPTUNA_ROOT = Path(__file__).resolve().parent
QUEUE_ROOT = OPTUNA_ROOT / "model_queue_runs"
OUTPUT_ROOT = OPTUNA_ROOT / "randomness_audit_results_model_queue"
MAG_OUTPUT_ROOT = OPTUNA_ROOT / "magnitude_prefix_results_model_queue"
CACHE_ROOT = OPTUNA_ROOT / "rmt_cache_models"
THRESHOLD_MAG_OUTPUT_ROOT = OPTUNA_ROOT / "magnitude_until_drop_results_model_queue"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequentially run model hybrid pruning jobs on one pod.")
    parser.add_argument("--queue-name", required=True)
    parser.add_argument("--queue-file", required=True)
    parser.add_argument("--skip-deps", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not slug:
        raise ValueError(f"Could not derive slug from {value!r}")
    return slug


def load_queue(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"Expected list in {path}, found {type(payload)!r}")
    return [item for item in payload if isinstance(item, dict)]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def ensure_deps() -> None:
    code = "import timm, datasets, pyarrow, scipy, sklearn, safetensors, matplotlib, TracyWidom, PIL.Image, PIL.ExifTags; PIL.Image.ExifTags = getattr(PIL.Image, 'ExifTags', PIL.ExifTags)"
    result = subprocess.run([sys.executable, "-c", code])
    if result.returncode == 0:
        return
    packages = [
        "timm==1.0.26",
        "datasets==4.8.4",
        "pyarrow==24.0.0",
        "scipy==1.15.3",
        "scikit-learn",
        "pandas",
        "safetensors",
        "huggingface_hub",
        "matplotlib",
        "TracyWidom",
        "pillow==12.2.0",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)


def run_complete(run_dir: Path) -> bool:
    results_path = run_dir / "results.json"
    checkpoint_path = run_dir / "checkpoints" / "keep_s70.pt"
    if not results_path.exists() or not checkpoint_path.exists():
        return False
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    steps = payload.get("steps", []) if isinstance(payload, dict) else []
    return any(isinstance(step, dict) and abs(float(step.get("target_sparsity", -1.0)) - 0.70) < 1e-9 for step in steps)


def command_for(item: dict, run_name: str, queue_name: str, output_root: Path, mag_root: Path) -> list[str]:
    batch_size_train = int(item.get("batch_size_train", 128))
    batch_size_val = int(item.get("batch_size_val", 256))
    probe_batch_size = int(item.get("probe_batch_size", batch_size_val))
    num_workers = int(item.get("num_workers", 4))
    prefetch_factor = int(item.get("prefetch_factor", 2))
    phase_a_min_batches = int(item.get("phase_a_min_batches_per_epoch", 128))
    return [
        sys.executable,
        "-u",
        str(OPTUNA_ROOT / "hybrid_mag_until_drop_then_v8_model_queue.py"),
        "--model-name",
        str(item["model_name"]),
        "--run-name",
        run_name,
        "--output-root",
        str(output_root),
        "--mag-output-root",
        str(mag_root),
        "--threshold-mag-output-root",
        str(THRESHOLD_MAG_OUTPUT_ROOT / queue_name),
        "--cache-root",
        str(CACHE_ROOT),
        "--batch-size-train",
        str(batch_size_train),
        "--batch-size-val",
        str(batch_size_val),
        "--probe-batch-size",
        str(probe_batch_size),
        "--num-workers",
        str(num_workers),
        "--prefetch-factor",
        str(prefetch_factor),
        "--phase-a-min-batches-per-epoch",
        str(phase_a_min_batches),
    ]


def main() -> int:
    args = parse_args()
    queue_name = slugify(args.queue_name)
    queue_file = Path(args.queue_file)
    queue = load_queue(queue_file)

    queue_dir = QUEUE_ROOT / queue_name
    output_root = OUTPUT_ROOT / queue_name
    mag_root = MAG_OUTPUT_ROOT / queue_name
    state_path = queue_dir / "queue_state.json"
    history_path = queue_dir / "queue_history.json"
    queue_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("RMT_SKIP_PRE_FT_EVAL", "1")
    if not args.skip_deps:
        ensure_deps()

    history: list[dict] = []
    if history_path.exists():
        try:
            existing = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                history = existing
        except Exception:
            history = []

    for index, item in enumerate(queue):
        model_name = str(item["model_name"])
        run_name = str(item.get("run_name") or f"full_run_2026_04_26_{slugify(model_name)}_hybrid_mag20_v8")
        run_dir = output_root / run_name
        write_json(
            state_path,
            {
                "queue_name": queue_name,
                "state": "starting_model",
                "index": index,
                "total": len(queue),
                "model_name": model_name,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "updated_at": utc_now(),
            },
        )
        if run_complete(run_dir):
            history.append({"model_name": model_name, "run_name": run_name, "state": "already_complete", "ts": utc_now()})
            write_json(history_path, history)
            continue

        command = command_for(item, run_name, queue_name, output_root, mag_root)
        write_json(
            state_path,
            {
                "queue_name": queue_name,
                "state": "running_model",
                "index": index,
                "total": len(queue),
                "model_name": model_name,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "cmd": command,
                "updated_at": utc_now(),
            },
        )
        started = time.time()
        returncode = subprocess.run(command, cwd=OPTUNA_ROOT).returncode
        elapsed = time.time() - started
        history.append(
            {
                "model_name": model_name,
                "run_name": run_name,
                "returncode": returncode,
                "elapsed_sec": elapsed,
                "ts": utc_now(),
            }
        )
        write_json(history_path, history)
        if returncode != 0:
            write_json(
                state_path,
                {
                    "queue_name": queue_name,
                    "state": "failed",
                    "index": index,
                    "total": len(queue),
                    "model_name": model_name,
                    "run_name": run_name,
                    "returncode": returncode,
                    "updated_at": utc_now(),
                },
            )
            return int(returncode)

    write_json(
        state_path,
        {
            "queue_name": queue_name,
            "state": "complete",
            "total": len(queue),
            "updated_at": utc_now(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

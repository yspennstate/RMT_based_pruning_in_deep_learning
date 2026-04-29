#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
OPTUNA_ROOT = THIS_FILE.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Queue wrapper for validation-drop-gated magnitude-to-SER handoff. "
            "If a run has already passed the old transition point, delegate to the legacy wrapper."
        )
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mag-output-root", required=True)
    parser.add_argument("--threshold-mag-output-root", default="")
    parser.add_argument("--cache-root", default=str(OPTUNA_ROOT / "rmt_cache_models"))
    parser.add_argument("--drop-threshold-pp", type=float, default=0.7)
    parser.add_argument("--batch-size-train", type=int, default=128)
    parser.add_argument("--batch-size-val", type=int, default=256)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--phase-a-min-batches-per-epoch", type=int, default=128)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[hybrid_mag_until_drop_queue] {message}", flush=True)


def load_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def legacy_has_transition_progress(run_dir: Path) -> bool:
    results = load_json(run_dir / "results.json")
    if isinstance(results, dict):
        for step in results.get("steps", []):
            if not isinstance(step, dict):
                continue
            try:
                target_s = float(step.get("target_sparsity", -1.0))
            except Exception:
                continue
            if target_s > 0.20 + 1.0e-9:
                return True
            if (
                abs(target_s - 0.25) < 1.0e-9
                and step.get("source_method") != "classical_magnitude_until_drop"
            ):
                return True
    return (run_dir / "checkpoints" / "keep_s25.pt").exists()


def legacy_complete(run_dir: Path) -> bool:
    results = load_json(run_dir / "results.json")
    if not isinstance(results, dict):
        return False
    if not (run_dir / "checkpoints" / "keep_s70.pt").exists():
        return False
    return any(
        isinstance(step, dict) and abs(float(step.get("target_sparsity", -1.0)) - 0.70) < 1.0e-9
        for step in results.get("steps", [])
    )


def legacy_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(OPTUNA_ROOT / "hybrid_mag20_then_v8_model_queue.py"),
        "--model-name",
        args.model_name,
        "--run-name",
        args.run_name,
        "--output-root",
        args.output_root,
        "--mag-output-root",
        args.mag_output_root,
        "--cache-root",
        args.cache_root,
        "--batch-size-train",
        str(args.batch_size_train),
        "--batch-size-val",
        str(args.batch_size_val),
        "--probe-batch-size",
        str(args.probe_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--phase-a-min-batches-per-epoch",
        str(args.phase_a_min_batches_per_epoch),
    ]


def threshold_command(args: argparse.Namespace, threshold_mag_root: Path, source_mag_run_dir: Path | None) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(OPTUNA_ROOT / "hybrid_mag_until_drop_then_v8_model.py"),
        "--model-name",
        args.model_name,
        "--run-name",
        args.run_name,
        "--output-root",
        args.output_root,
        "--mag-output-root",
        str(threshold_mag_root),
        "--cache-root",
        args.cache_root,
        "--drop-threshold-pp",
        str(args.drop_threshold_pp),
        "--reference-mode",
        "baseline",
        "--batch-size-train",
        str(args.batch_size_train),
        "--batch-size-val",
        str(args.batch_size_val),
        "--probe-batch-size",
        str(args.probe_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--phase-a-min-batches-per-epoch",
        str(args.phase_a_min_batches_per_epoch),
    ]
    if source_mag_run_dir is not None:
        command.extend(["--source-mag-run-dir", str(source_mag_run_dir)])
    return command


def main() -> int:
    args = parse_args()
    run_dir = Path(args.output_root) / args.run_name
    legacy_mag_run_dir = Path(args.mag_output_root) / f"{args.run_name}_prefix"
    threshold_mag_root = (
        Path(args.threshold_mag_output_root)
        if args.threshold_mag_output_root
        else OPTUNA_ROOT / "magnitude_until_drop_results_model_queue" / Path(args.output_root).name
    )

    if legacy_complete(run_dir):
        log(f"run already complete: {run_dir}")
        return 0

    if legacy_has_transition_progress(run_dir):
        log("existing run has post-transition SER/RMT progress; delegating to legacy wrapper")
        return subprocess.run(legacy_command(args), cwd=OPTUNA_ROOT).returncode

    source = legacy_mag_run_dir if legacy_mag_run_dir.exists() else None
    if source is not None:
        log(f"seeding threshold magnitude run from existing prefix: {source}")
    else:
        log("no existing prefix found; threshold magnitude run starts from dense model")

    return subprocess.run(threshold_command(args, threshold_mag_root, source), cwd=OPTUNA_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())

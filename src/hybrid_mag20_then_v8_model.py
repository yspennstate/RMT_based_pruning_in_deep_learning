#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch


THIS_FILE = Path(__file__).resolve()
OPTUNA_ROOT = THIS_FILE.parent

PREFIX_TARGETS = [0.05, 0.10, 0.15, 0.20]
FULL_SPARSITIES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exact magnitude-to-0.20, then SER prune-restore, for an arbitrary timm ViT/DeiT model."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", default=str(OPTUNA_ROOT / "randomness_audit_results_deit_hybrid"))
    parser.add_argument("--mag-output-root", default=str(OPTUNA_ROOT / "magnitude_prefix_results_deit"))
    parser.add_argument("--cache-root", default=str(OPTUNA_ROOT / "rmt_cache_models"))
    parser.add_argument(
        "--v8-reference-run-dir",
        default=str(
            OPTUNA_ROOT
            / "randomness_audit_results_v8"
            / "full_run_2026_04_22_v8_paper_rmt_prune08_restore03_exact_newpod"
        ),
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[hybrid_mag20_then_v8_model] {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp.replace(path)


def model_slug(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-")
    if not slug:
        raise ValueError(f"Could not derive slug from model name: {model_name!r}")
    return slug


def keep_checkpoint_path(checkpoint_dir: Path, target_s: float) -> Path:
    return checkpoint_dir / f"keep_s{int(round(target_s * 100)):02d}.pt"


def magnitude_checkpoint_path(mag_checkpoint_dir: Path, target_s: float) -> Path:
    cycle_idx = PREFIX_TARGETS.index(target_s)
    return mag_checkpoint_dir / f"cycle_{cycle_idx:02d}_s{int(round(target_s * 100)):02d}.pt"


def load_hybrid_completed_targets(results_path: Path) -> set[float]:
    if not results_path.exists():
        return set()
    with open(results_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return set()
    steps = payload.get("steps", [])
    completed: set[float] = set()
    for step in steps:
        if isinstance(step, dict) and "target_sparsity" in step:
            completed.add(float(step["target_sparsity"]))
    return completed


def hybrid_has_stage2_progress(results_path: Path, checkpoint_dir: Path) -> bool:
    completed = load_hybrid_completed_targets(results_path)
    if any(target_s > 0.20 for target_s in completed):
        return True
    return keep_checkpoint_path(checkpoint_dir, 0.25).exists()


def hybrid_stage2_complete(results_path: Path, checkpoint_dir: Path) -> bool:
    completed = load_hybrid_completed_targets(results_path)
    return keep_checkpoint_path(checkpoint_dir, 0.70).exists() and 0.70 in completed


def load_magnitude_results(mag_results_path: Path) -> list[dict]:
    if not mag_results_path.exists():
        return []
    with open(mag_results_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"Expected list payload in {mag_results_path}, found {type(payload)!r}")
    return [row for row in payload if isinstance(row, dict)]


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def magnitude_prefix_complete(mag_results_path: Path, mag_checkpoint_dir: Path) -> bool:
    if not magnitude_checkpoint_path(mag_checkpoint_dir, 0.20).exists():
        return False
    found = {float(row["target_s"]) for row in load_magnitude_results(mag_results_path) if "target_s" in row}
    return all(target_s in found for target_s in PREFIX_TARGETS)


def run_exact_magnitude_prefix_if_needed(
    model_name_value: str,
    mag_run_dir: Path,
    mag_results_path: Path,
    mag_checkpoint_dir: Path,
    results_path: Path,
    checkpoint_dir: Path,
) -> None:
    if hybrid_has_stage2_progress(results_path, checkpoint_dir):
        log("hybrid run already progressed beyond s=0.20; skipping magnitude prefix rerun")
        return
    if magnitude_prefix_complete(mag_results_path, mag_checkpoint_dir):
        log(f"exact magnitude prefix already complete at {mag_run_dir}")
        return

    ensure_dir(mag_run_dir)
    ensure_dir(mag_checkpoint_dir)

    command = [
        sys.executable,
        "-u",
        str(OPTUNA_ROOT / "run_finetune_magnitude_model_exec.py"),
        "--model-name-override",
        model_name_value,
        "--output-dir",
        str(mag_run_dir),
        "--max-cycles",
        str(len(PREFIX_TARGETS)),
    ]
    log(f"starting stage 1: exact run_finetune_magnitude.py through s=0.20 for {model_name_value}")
    subprocess.run(command, cwd=OPTUNA_ROOT, check=True)

    if not magnitude_prefix_complete(mag_results_path, mag_checkpoint_dir):
        raise RuntimeError("Exact magnitude prefix did not produce the expected s=0.20 checkpoint.")
    log("stage 1 complete")


def build_prefix_steps_from_magnitude_results(mag_results_path: Path) -> list[dict]:
    rows_by_target: dict[float, dict] = {}
    for row in load_magnitude_results(mag_results_path):
        if "target_s" not in row:
            continue
        target_s = float(row["target_s"])
        if target_s not in PREFIX_TARGETS:
            continue
        rows_by_target[target_s] = row

    missing = [target_s for target_s in PREFIX_TARGETS if target_s not in rows_by_target]
    if missing:
        raise RuntimeError(f"Magnitude results missing prefix targets: {missing}")

    steps: list[dict] = []
    for step_index, target_s in enumerate(PREFIX_TARGETS):
        row = rows_by_target[target_s]
        steps.append(
            {
                "step_index": step_index,
                "target_sparsity": target_s,
                "source_method": "exact_magnitude_prefix",
                "planned_target_sparsity_post_reinsert": float(row.get("target_s", target_s)),
                "achieved_sparsity_post_reinsert": float(row.get("achieved_s", target_s)),
                "pre_ft_top1": optional_float(row.get("pre_ft_top1")),
                "val_top1": float(row.get("post_ft_top1", 0.0)),
                "phase_b": {
                    "epochs": int(row.get("epochs", 0)),
                    "base_lr": float(row.get("lr", 0.0)),
                },
                "ts": row.get("ts", ""),
            }
        )
    return steps


def write_hybrid_prefix_checkpoint(src: Path, dst: Path) -> None:
    payload = torch.load(src, map_location="cpu", weights_only=False)
    converted = {
        "model_state_dict": payload["model_state_dict"],
        "protected_masks": {},
    }
    tmp = dst.with_suffix(".tmp")
    torch.save(converted, tmp)
    tmp.replace(dst)


def prepare_v8_handoff_if_needed(
    model_name_value: str,
    run_dir: Path,
    checkpoint_dir: Path,
    results_path: Path,
    handoff_meta_path: Path,
    mag_run_dir: Path,
    mag_results_path: Path,
    mag_checkpoint_dir: Path,
    v8_reference_run_dir: Path,
) -> None:
    if hybrid_has_stage2_progress(results_path, checkpoint_dir):
        log("hybrid run directory already has stage-2 progress; keeping existing handoff artifacts")
        return

    ensure_dir(run_dir)
    ensure_dir(checkpoint_dir)

    log("preparing SER resume directory from exact magnitude checkpoints")
    for target_s in PREFIX_TARGETS:
        src = magnitude_checkpoint_path(mag_checkpoint_dir, target_s)
        if not src.exists():
            raise FileNotFoundError(src)
        write_hybrid_prefix_checkpoint(src, keep_checkpoint_path(checkpoint_dir, target_s))

    results = {"steps": build_prefix_steps_from_magnitude_results(mag_results_path)}
    save_json(results_path, results)
    save_json(
        handoff_meta_path,
        {
            "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_name": model_name_value,
            "model_slug": model_slug(model_name_value),
            "prefix_targets": PREFIX_TARGETS,
            "magnitude_prefix_run_dir": str(mag_run_dir),
            "v8_reference_run_dir": str(v8_reference_run_dir),
        },
    )
    log(f"stage 1 handoff prepared in {run_dir}")


def ensure_model_cache(model_name_value: str, cache_dir: Path) -> None:
    ensure_dir(cache_dir)
    command = [
        sys.executable,
        "-u",
        str(OPTUNA_ROOT / "build_model_rmt_cache.py"),
        "--model-name",
        model_name_value,
        "--cache-dir",
        str(cache_dir),
    ]
    log(f"ensuring model-specific RMT cache at {cache_dir}")
    subprocess.run(command, cwd=OPTUNA_ROOT, check=True)


def stage2_command(model_name_value: str, cache_dir: Path, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(OPTUNA_ROOT / "run_removed_matrix_audit_v8_model_exec.py"),
        "--model-name-override",
        model_name_value,
        "--cache-dir-override",
        str(cache_dir),
        "--optuna-root",
        str(OPTUNA_ROOT),
        "--resume-run-dir",
        str(run_dir),
        "--sparsities",
        ",".join(f"{target_s:.2f}" for target_s in FULL_SPARSITIES),
        "--batch-size-train",
        "128",
        "--batch-size-val",
        "256",
        "--probe-batch-size",
        "256",
        "--num-workers",
        "4",
        "--prefetch-factor",
        "2",
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
        "128",
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


def run_exact_v8_stage_if_needed(
    model_name_value: str,
    cache_dir: Path,
    run_dir: Path,
    results_path: Path,
    checkpoint_dir: Path,
) -> None:
    if hybrid_stage2_complete(results_path, checkpoint_dir):
        log("stage 2 already complete")
        return
    log("starting stage 2: SER continuation")
    subprocess.run(stage2_command(model_name_value, cache_dir, run_dir), cwd=OPTUNA_ROOT, check=True)


def main() -> None:
    args = parse_args()
    slug = model_slug(args.model_name)

    output_root = Path(args.output_root)
    mag_output_root = Path(args.mag_output_root)
    cache_root = Path(args.cache_root)
    v8_reference_run_dir = Path(args.v8_reference_run_dir)

    run_dir = output_root / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    results_path = run_dir / "results.json"
    handoff_meta_path = run_dir / "prefix_handoff.json"

    mag_run_dir = mag_output_root / f"{args.run_name}_prefix"
    mag_checkpoint_dir = mag_run_dir / "checkpoints"
    mag_results_path = mag_run_dir / "finetune_results.json"

    cache_dir = cache_root / slug

    if not hybrid_has_stage2_progress(results_path, checkpoint_dir):
        run_exact_magnitude_prefix_if_needed(
            args.model_name,
            mag_run_dir,
            mag_results_path,
            mag_checkpoint_dir,
            results_path,
            checkpoint_dir,
        )
        prepare_v8_handoff_if_needed(
            args.model_name,
            run_dir,
            checkpoint_dir,
            results_path,
            handoff_meta_path,
            mag_run_dir,
            mag_results_path,
            mag_checkpoint_dir,
            v8_reference_run_dir,
        )
    else:
        log("existing hybrid run detected beyond prefix; resuming SER stage directly")

    ensure_model_cache(args.model_name, cache_dir)
    run_exact_v8_stage_if_needed(args.model_name, cache_dir, run_dir, results_path, checkpoint_dir)


if __name__ == "__main__":
    main()

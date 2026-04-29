#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import gc
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch


THIS_FILE = Path(__file__).resolve()
OPTUNA_ROOT = THIS_FILE.parent

FULL_SPARSITIES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
S20_SPARSITY = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume a model from an existing s=0.20 magnitude checkpoint, continue "
            "classical magnitude until a validation drop threshold is crossed, then "
            "hand off to the SER prune-restore continuation."
        )
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-mag-run-dir", default="")
    parser.add_argument("--output-root", default=str(OPTUNA_ROOT / "randomness_audit_results_threshold_hybrid"))
    parser.add_argument("--mag-output-root", default=str(OPTUNA_ROOT / "magnitude_until_drop_results"))
    parser.add_argument("--cache-root", default=str(OPTUNA_ROOT / "rmt_cache_models"))
    parser.add_argument("--drop-threshold-pp", type=float, default=0.7)
    parser.add_argument(
        "--reference-mode",
        choices=["s20_start", "baseline", "first_completed"],
        default="s20_start",
        help=(
            "Accuracy reference for the drop threshold. s20_start preserves the original "
            "one-off ViT-Large/Hiera experiment; baseline is the queue default."
        ),
    )
    parser.add_argument("--batch-size-train", type=int, default=128)
    parser.add_argument("--batch-size-val", type=int, default=256)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--phase-a-min-batches-per-epoch", type=int, default=128)
    parser.add_argument("--zero-tolerance", type=float, default=5.0e-4)
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def log(message: str) -> None:
    print(f"[hybrid_mag_until_drop] {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def model_slug(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-")
    if not slug:
        raise ValueError(f"Could not derive slug from model name: {model_name!r}")
    return slug


def target_key(target_s: float) -> str:
    return f"{float(target_s):.4f}"


def keep_checkpoint_path(checkpoint_dir: Path, target_s: float) -> Path:
    return checkpoint_dir / f"keep_s{int(round(target_s * 100)):02d}.pt"


def magnitude_checkpoint_path(mag_checkpoint_dir: Path, target_s: float) -> Path:
    cycle_idx = FULL_SPARSITIES.index(target_s)
    return mag_checkpoint_dir / f"cycle_{cycle_idx:02d}_s{int(round(target_s * 100)):02d}.pt"


def load_magnitude_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = load_json(path)
    if not isinstance(payload, list):
        raise TypeError(f"Expected list in {path}, found {type(payload)!r}")
    return [row for row in payload if isinstance(row, dict)]


def completed_magnitude_targets(results_path: Path, checkpoint_dir: Path) -> set[float]:
    completed: set[float] = set()
    rows = load_magnitude_results(results_path)
    for row in rows:
        if "target_s" not in row:
            continue
        target_s = float(row["target_s"])
        if target_s <= 0:
            continue
        if magnitude_checkpoint_path(checkpoint_dir, target_s).exists():
            completed.add(round(target_s, 2))
    return completed


def latest_completed_target(results_path: Path, checkpoint_dir: Path) -> float | None:
    completed = sorted(completed_magnitude_targets(results_path, checkpoint_dir))
    return completed[-1] if completed else None


def row_for_target(rows: list[dict], target_s: float) -> dict | None:
    for row in rows:
        if abs(float(row.get("target_s", -1.0)) - target_s) < 1.0e-9:
            return row
    return None


def patch_pillow_exiftags() -> None:
    try:
        import PIL.ExifTags
        import PIL.Image
    except Exception:
        return
    if not hasattr(PIL.Image, "ExifTags"):
        PIL.Image.ExifTags = PIL.ExifTags


def seed_magnitude_run_from_source(source_run_dir: Path | None, mag_run_dir: Path) -> None:
    if source_run_dir is None:
        ensure_dir(mag_run_dir)
        ensure_dir(mag_run_dir / "checkpoints")
        if not (mag_run_dir / "finetune_results.json").exists():
            save_json(mag_run_dir / "finetune_results.json", [])
        return

    ensure_dir(mag_run_dir)
    ensure_dir(mag_run_dir / "checkpoints")

    source_results = source_run_dir / "finetune_results.json"
    if not source_results.exists():
        raise FileNotFoundError(source_results)
    source_checkpoint_dir = source_run_dir / "checkpoints"

    target_results = mag_run_dir / "finetune_results.json"
    if not target_results.exists():
        rows = []
        for row in load_magnitude_results(source_results):
            target_s = float(row.get("target_s", -1.0))
            if target_s <= FULL_SPARSITIES[-1] + 1.0e-9:
                copied = dict(row)
                if target_s > 0:
                    copied["source_method"] = "seeded_existing_magnitude"
                rows.append(copied)
        save_json(target_results, rows)

    for target_s in FULL_SPARSITIES:
        src = magnitude_checkpoint_path(source_checkpoint_dir, target_s)
        dst = magnitude_checkpoint_path(mag_run_dir / "checkpoints", target_s)
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    source_log = source_run_dir / "finetune_log.txt"
    target_log = mag_run_dir / "finetune_log.txt"
    if source_log.exists() and not target_log.exists():
        shutil.copy2(source_log, target_log)


def verify_checkpoint(mag_run_dir: Path, target_s: float, tolerance: float) -> dict:
    checkpoint_path = magnitude_checkpoint_path(mag_run_dir / "checkpoints", target_s)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    masks = payload.get("masks")
    state = payload.get("model_state_dict")
    if not isinstance(masks, dict) or not isinstance(state, dict):
        raise TypeError(f"Unexpected magnitude checkpoint format in {checkpoint_path}")

    mask_total = 0
    mask_zero = 0
    state_total = 0
    state_zero = 0
    mismatch = 0
    for name, mask in masks.items():
        mask_bool = mask.detach().cpu().bool()
        mask_total += int(mask_bool.numel())
        mask_zero += int((~mask_bool).sum().item())
        weight_key = f"{name}.weight"
        if weight_key in state:
            weight = state[weight_key].detach().cpu()
            zero_weight = weight == 0
            state_total += int(weight.numel())
            state_zero += int(zero_weight.sum().item())
            mismatch += int((zero_weight != (~mask_bool)).sum().item())

    mask_sparsity = mask_zero / max(mask_total, 1)
    state_sparsity = state_zero / max(state_total, 1)
    verification = {
        "checked_at": utc_now(),
        "checkpoint": str(checkpoint_path),
        "target_sparsity": target_s,
        "mask_zero_count": mask_zero,
        "mask_total_count": mask_total,
        "mask_sparsity": mask_sparsity,
        "state_zero_count": state_zero,
        "state_total_count": state_total,
        "state_sparsity": state_sparsity,
        "mask_state_mismatch_count": mismatch,
        "zero_tolerance": tolerance,
        "passed": abs(mask_sparsity - target_s) <= tolerance
        and abs(state_sparsity - target_s) <= tolerance
        and mismatch == 0,
    }
    save_json(mag_run_dir / f"s{int(round(target_s * 100)):02d}_zero_verification.json", verification)
    if not verification["passed"]:
        raise RuntimeError(f"s={target_s:.2f} sparsity verification failed: {verification}")
    log(
        f"verified s={target_s:.2f} checkpoint: mask_s={mask_sparsity:.6f} "
        f"state_s={state_sparsity:.6f} mismatch={mismatch}"
    )
    return verification


def load_run_finetune_magnitude_module(log_file: Path):
    patch_pillow_exiftags()
    source_path = OPTUNA_ROOT / "run_finetune_magnitude.py"
    spec = importlib.util.spec_from_file_location("run_finetune_magnitude_for_threshold_baseline", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.LOG_FILE = log_file
    return module


def compute_dense_baseline_top1(
    *,
    model_name: str,
    mag_run_dir: Path,
    batch_size_val: int,
    num_workers: int,
) -> float:
    module = load_run_finetune_magnitude_module(mag_run_dir / "finetune_log.txt")
    module.BATCH_SIZE_VAL = int(batch_size_val)
    module.NUM_WORKERS = int(num_workers)

    original_create_model = module.timm.create_model

    def create_model_override(_model_name, *create_args, **create_kwargs):
        return original_create_model(model_name, *create_args, **create_kwargs)

    module.timm.create_model = create_model_override
    module.log(f"threshold baseline eval for {model_name}")
    model = module.timm.create_model("vit_base_patch16_224", pretrained=True)
    data_config = module.timm.data.resolve_model_data_config(model)
    preprocess_val = module.timm.data.create_transform(**data_config, is_training=False)
    preprocess_train = module.timm.data.create_transform(**data_config, is_training=True)
    model.to(module.DEVICE)
    _train_loader, val_loader = module.build_loaders(preprocess_train, preprocess_val)
    top1 = float(module.evaluate_model(model, val_loader, label="threshold dense baseline"))
    del model, val_loader, _train_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return top1


def ensure_baseline_reference(
    *,
    model_name: str,
    mag_run_dir: Path,
    batch_size_val: int,
    num_workers: int,
) -> float:
    results_path = mag_run_dir / "finetune_results.json"
    rows = load_magnitude_results(results_path)
    for row in rows:
        if abs(float(row.get("target_s", -1.0))) < 1.0e-9 and row.get("post_ft_top1") is not None:
            return float(row["post_ft_top1"])

    baseline_top1 = compute_dense_baseline_top1(
        model_name=model_name,
        mag_run_dir=mag_run_dir,
        batch_size_val=batch_size_val,
        num_workers=num_workers,
    )
    baseline_row = {
        "cycle_idx": -1,
        "target_s": 0.0,
        "method": "baseline",
        "epochs": 0,
        "lr": 0.0,
        "pre_ft_top1": baseline_top1,
        "post_ft_top1": baseline_top1,
        "achieved_s": 0.0,
        "source_method": "threshold_dense_baseline_eval",
        "ts": dt.datetime.utcnow().isoformat(),
    }
    replaced = False
    updated_rows: list[dict] = []
    for row in rows:
        if abs(float(row.get("target_s", -1.0))) < 1.0e-9:
            updated_rows.append({**row, **baseline_row})
            replaced = True
        else:
            updated_rows.append(row)
    if not replaced:
        updated_rows.insert(0, baseline_row)
    save_json(results_path, updated_rows)
    return baseline_top1


def reference_top1_for_mode(
    *,
    model_name: str,
    mag_run_dir: Path,
    mode: str,
    batch_size_val: int,
    num_workers: int,
) -> tuple[float, float, str]:
    rows = load_magnitude_results(mag_run_dir / "finetune_results.json")
    if mode == "baseline":
        return (
            0.0,
            ensure_baseline_reference(
                model_name=model_name,
                mag_run_dir=mag_run_dir,
                batch_size_val=batch_size_val,
                num_workers=num_workers,
            ),
            "dense_baseline_top1",
        )
    if mode == "s20_start":
        row = row_for_target(rows, S20_SPARSITY)
        if row is None or row.get("post_ft_top1") is None:
            raise RuntimeError("Cannot determine threshold reference from s=0.20 post-FT accuracy.")
        return S20_SPARSITY, float(row["post_ft_top1"]), "post_ft_top1_at_s20_start_checkpoint"
    completed = sorted(
        target_s
        for target_s in completed_magnitude_targets(mag_run_dir / "finetune_results.json", mag_run_dir / "checkpoints")
        if target_s > 0
    )
    if not completed:
        return (
            0.0,
            ensure_baseline_reference(
                model_name=model_name,
                mag_run_dir=mag_run_dir,
                batch_size_val=batch_size_val,
                num_workers=num_workers,
            ),
            "dense_baseline_top1_no_completed_magnitude",
        )
    target_s = completed[0]
    row = row_for_target(rows, target_s)
    if row is None or row.get("post_ft_top1") is None:
        raise RuntimeError(f"Cannot determine threshold reference from s={target_s:.2f}.")
    return target_s, float(row["post_ft_top1"]), "first_completed_magnitude_post_ft_top1"


def run_magnitude_until_threshold(
    *,
    model_name: str,
    mag_run_dir: Path,
    batch_size_train: int,
    batch_size_val: int,
    num_workers: int,
    drop_threshold_pp: float,
    reference_mode: str,
) -> dict:
    mag_results_path = mag_run_dir / "finetune_results.json"
    mag_checkpoint_dir = mag_run_dir / "checkpoints"
    reference_sparsity, reference_top1, reference_description = reference_top1_for_mode(
        model_name=model_name,
        mag_run_dir=mag_run_dir,
        mode=reference_mode,
        batch_size_val=batch_size_val,
        num_workers=num_workers,
    )
    threshold_top1 = reference_top1 - drop_threshold_pp

    meta_path = mag_run_dir / "magnitude_until_drop_meta.json"
    meta = {
        "model_name": model_name,
        "reference_mode": reference_mode,
        "reference_description": reference_description,
        "reference_sparsity": reference_sparsity,
        "reference_top1": reference_top1,
        "drop_threshold_pp": drop_threshold_pp,
        "threshold_top1": threshold_top1,
        "updated_at": utc_now(),
        "transition_target_s": None,
        "transition_reason": None,
    }

    while True:
        rows = load_magnitude_results(mag_results_path)
        completed = sorted(completed_magnitude_targets(mag_results_path, mag_checkpoint_dir))
        latest = completed[-1] if completed else None
        if latest is not None and latest > reference_sparsity + 1.0e-9:
            latest_row = row_for_target(rows, latest)
            if latest_row is not None and latest_row.get("post_ft_top1") is not None:
                latest_top1 = float(latest_row["post_ft_top1"])
                if latest_top1 < threshold_top1:
                    meta.update(
                        {
                            "transition_target_s": latest,
                            "transition_observed_top1": latest_top1,
                            "transition_drop_pp": reference_top1 - latest_top1,
                            "transition_reason": "post_ft_top1_drop_exceeded_threshold",
                            "updated_at": utc_now(),
                        }
                    )
                    save_json(meta_path, meta)
                    log(
                        f"magnitude threshold crossed at s={latest:.2f}: "
                        f"top1={latest_top1:.3f}, reference={reference_top1:.3f}, "
                        f"drop={reference_top1 - latest_top1:.3f}pp"
                    )
                    return meta

        if latest is not None and latest >= FULL_SPARSITIES[-1]:
            meta.update(
                {
                    "transition_target_s": None,
                    "transition_reason": "never_crossed_threshold_before_final_sparsity",
                    "updated_at": utc_now(),
                }
            )
            save_json(meta_path, meta)
            log("magnitude reached final sparsity without crossing threshold; no RMT continuation remains")
            return meta

        next_index = 0 if latest is None else FULL_SPARSITIES.index(round(latest, 2)) + 1
        max_cycles = next_index + 1
        next_target = FULL_SPARSITIES[next_index]
        log(
            f"running classical magnitude through s={next_target:.2f} "
            f"(max_cycles={max_cycles}, threshold_top1={threshold_top1:.3f})"
        )
        command = [
            sys.executable,
            "-u",
            str(OPTUNA_ROOT / "run_finetune_magnitude_model_exec_queue.py"),
            "--model-name-override",
            model_name,
            "--output-dir",
            str(mag_run_dir),
            "--max-cycles",
            str(max_cycles),
            "--batch-size-train",
            str(batch_size_train),
            "--batch-size-val",
            str(batch_size_val),
            "--num-workers",
            str(num_workers),
        ]
        subprocess.run(command, cwd=OPTUNA_ROOT, check=True)


def build_handoff_steps_from_magnitude(mag_results_path: Path, completed_targets: list[float]) -> list[dict]:
    rows = load_magnitude_results(mag_results_path)
    steps: list[dict] = []
    for step_index, target_s in enumerate(FULL_SPARSITIES):
        if target_s not in completed_targets:
            continue
        row = row_for_target(rows, target_s)
        if row is None:
            raise RuntimeError(f"Missing magnitude result row for s={target_s:.2f}")
        steps.append(
            {
                "step_index": step_index,
                "target_sparsity": target_s,
                "source_method": "classical_magnitude_until_drop",
                "planned_target_sparsity_post_reinsert": float(row.get("target_s", target_s)),
                "achieved_sparsity_post_reinsert": float(row.get("achieved_s", target_s)),
                "pre_ft_top1": row.get("pre_ft_top1"),
                "val_top1": float(row.get("post_ft_top1", 0.0)),
                "phase_b": {
                    "epochs": int(row.get("epochs", 0)),
                    "base_lr": float(row.get("lr", 0.0)),
                },
                "ts": row.get("ts", ""),
            }
        )
    return steps


def write_handoff_checkpoint(src: Path, dst: Path) -> None:
    payload = torch.load(src, map_location="cpu", weights_only=False)
    converted = {
        "model_state_dict": payload["model_state_dict"],
        "protected_masks": {},
    }
    tmp = dst.with_suffix(".tmp")
    torch.save(converted, tmp)
    tmp.replace(dst)


def hybrid_completed_targets(results_path: Path, checkpoint_dir: Path) -> set[float]:
    if not results_path.exists():
        return set()
    payload = load_json(results_path)
    if not isinstance(payload, dict):
        return set()
    completed: set[float] = set()
    for row in payload.get("steps", []):
        if not isinstance(row, dict) or "target_sparsity" not in row:
            continue
        target_s = round(float(row["target_sparsity"]), 2)
        if keep_checkpoint_path(checkpoint_dir, target_s).exists():
            completed.add(target_s)
    return completed


def hybrid_complete(results_path: Path, checkpoint_dir: Path) -> bool:
    completed = hybrid_completed_targets(results_path, checkpoint_dir)
    return round(FULL_SPARSITIES[-1], 2) in completed


def prepare_rmt_handoff_if_needed(
    *,
    model_name: str,
    run_dir: Path,
    checkpoint_dir: Path,
    results_path: Path,
    mag_run_dir: Path,
    transition_meta: dict,
) -> None:
    existing = hybrid_completed_targets(results_path, checkpoint_dir)
    mag_completed = sorted(completed_magnitude_targets(mag_run_dir / "finetune_results.json", mag_run_dir / "checkpoints"))
    if existing and max(existing) >= max(mag_completed):
        log("RMT run directory already contains handoff checkpoints; keeping existing handoff")
        return

    ensure_dir(run_dir)
    ensure_dir(checkpoint_dir)
    log(f"preparing RMT handoff from magnitude targets: {mag_completed}")
    for target_s in mag_completed:
        src = magnitude_checkpoint_path(mag_run_dir / "checkpoints", target_s)
        dst = keep_checkpoint_path(checkpoint_dir, target_s)
        if not dst.exists():
            write_handoff_checkpoint(src, dst)

    results = {"steps": build_handoff_steps_from_magnitude(mag_run_dir / "finetune_results.json", mag_completed)}
    save_json(results_path, results)
    save_json(
        run_dir / "magnitude_until_drop_handoff.json",
        {
            "prepared_at": utc_now(),
            "model_name": model_name,
            "magnitude_run_dir": str(mag_run_dir),
            "handoff_completed_targets": mag_completed,
            "transition_meta": transition_meta,
        },
    )


def ensure_model_cache(model_name: str, cache_dir: Path) -> None:
    ensure_dir(cache_dir)
    command = [
        sys.executable,
        "-u",
        str(OPTUNA_ROOT / "build_model_rmt_cache.py"),
        "--model-name",
        model_name,
        "--cache-dir",
        str(cache_dir),
    ]
    log(f"ensuring model-specific RMT cache at {cache_dir}")
    subprocess.run(command, cwd=OPTUNA_ROOT, check=True)


def stage2_command(model_name: str, cache_dir: Path, run_dir: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(OPTUNA_ROOT / "run_removed_matrix_audit_v8_model_exec.py"),
        "--model-name-override",
        model_name,
        "--cache-dir-override",
        str(cache_dir),
        "--optuna-root",
        str(OPTUNA_ROOT),
        "--resume-run-dir",
        str(run_dir),
        "--sparsities",
        ",".join(f"{target_s:.2f}" for target_s in FULL_SPARSITIES),
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
        str(args.phase_a_min_batches_per_epoch),
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


def run_rmt_continuation_if_needed(
    *,
    model_name: str,
    cache_dir: Path,
    run_dir: Path,
    results_path: Path,
    checkpoint_dir: Path,
    args: argparse.Namespace,
) -> None:
    if hybrid_complete(results_path, checkpoint_dir):
        log("RMT continuation already complete")
        return
    completed = hybrid_completed_targets(results_path, checkpoint_dir)
    if completed and max(completed) >= FULL_SPARSITIES[-1]:
        log("all sparsity targets already completed by magnitude; no RMT continuation to run")
        return
    log("starting RMT continuation after magnitude threshold handoff")
    subprocess.run(stage2_command(model_name, cache_dir, run_dir, args), cwd=OPTUNA_ROOT, check=True)


def main() -> None:
    args = parse_args()
    source_mag_run_dir = Path(args.source_mag_run_dir) if args.source_mag_run_dir else None
    output_root = Path(args.output_root)
    mag_output_root = Path(args.mag_output_root)
    cache_root = Path(args.cache_root)
    slug = model_slug(args.model_name)

    run_dir = output_root / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    results_path = run_dir / "results.json"
    mag_run_dir = mag_output_root / f"{args.run_name}_magnitude_until_drop"
    cache_dir = cache_root / slug

    log(f"model={args.model_name}")
    log(f"run_dir={run_dir}")
    log(f"mag_run_dir={mag_run_dir}")
    seed_magnitude_run_from_source(source_mag_run_dir, mag_run_dir)
    if magnitude_checkpoint_path(mag_run_dir / "checkpoints", S20_SPARSITY).exists():
        verify_checkpoint(mag_run_dir, S20_SPARSITY, args.zero_tolerance)
    transition_meta = run_magnitude_until_threshold(
        model_name=args.model_name,
        mag_run_dir=mag_run_dir,
        batch_size_train=args.batch_size_train,
        batch_size_val=args.batch_size_val,
        num_workers=args.num_workers,
        drop_threshold_pp=args.drop_threshold_pp,
        reference_mode=args.reference_mode,
    )
    prepare_rmt_handoff_if_needed(
        model_name=args.model_name,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        results_path=results_path,
        mag_run_dir=mag_run_dir,
        transition_meta=transition_meta,
    )
    ensure_model_cache(args.model_name, cache_dir)
    run_rmt_continuation_if_needed(
        model_name=args.model_name,
        cache_dir=cache_dir,
        run_dir=run_dir,
        results_path=results_path,
        checkpoint_dir=checkpoint_dir,
        args=args,
    )


if __name__ == "__main__":
    main()

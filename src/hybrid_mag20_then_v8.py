#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import torch


THIS_FILE = Path(__file__).resolve()
OPTUNA_ROOT = THIS_FILE.parent
REMOVED_MATRIX_DIR = OPTUNA_ROOT / "removed_matrix_audit"

RUN_NAME = "full_run_2026_04_26_exact_magnitude_to20_then_v8"
OUTPUT_ROOT = OPTUNA_ROOT / "randomness_audit_results_v11_hybrid"
RUN_DIR = OUTPUT_ROOT / RUN_NAME
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
RESULTS_PATH = RUN_DIR / "results.json"
HANDOFF_META_PATH = RUN_DIR / "prefix_handoff.json"

MAG_RUN_NAME = "full_run_2026_04_26_exact_magnitude_prefix_to20"
MAG_OUTPUT_ROOT = OPTUNA_ROOT / "magnitude_prefix_results"
MAG_RUN_DIR = MAG_OUTPUT_ROOT / MAG_RUN_NAME
MAG_CHECKPOINT_DIR = MAG_RUN_DIR / "checkpoints"
MAG_RESULTS_PATH = MAG_RUN_DIR / "finetune_results.json"
MAG_LOG_PATH = MAG_RUN_DIR / "finetune_log.txt"

V8_REFERENCE_RUN_DIR = (
    OPTUNA_ROOT
    / "randomness_audit_results_v8"
    / "full_run_2026_04_22_v8_paper_rmt_prune08_restore03_exact_newpod"
)

PREFIX_TARGETS = [0.05, 0.10, 0.15, 0.20]
FULL_SPARSITIES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def log(message: str) -> None:
    print(f"[hybrid_mag20_then_v8] {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp.replace(path)


def sparsity_key(value: float) -> str:
    return f"{float(value):.6f}"


def load_hybrid_completed_targets() -> set[float]:
    if not RESULTS_PATH.exists():
        return set()
    with open(RESULTS_PATH, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return set()
    steps = payload.get("steps", [])
    completed: set[float] = set()
    for step in steps:
        if isinstance(step, dict) and "target_sparsity" in step:
            completed.add(float(step["target_sparsity"]))
    return completed


def keep_checkpoint_path(target_s: float) -> Path:
    return CHECKPOINT_DIR / f"keep_s{int(round(target_s * 100)):02d}.pt"


def magnitude_checkpoint_path(target_s: float) -> Path:
    cycle_idx = PREFIX_TARGETS.index(target_s)
    return MAG_CHECKPOINT_DIR / f"cycle_{cycle_idx:02d}_s{int(round(target_s * 100)):02d}.pt"


def hybrid_has_stage2_progress() -> bool:
    completed = load_hybrid_completed_targets()
    if any(target_s > 0.20 for target_s in completed):
        return True
    return keep_checkpoint_path(0.25).exists()


def hybrid_stage2_complete() -> bool:
    completed = load_hybrid_completed_targets()
    return keep_checkpoint_path(0.70).exists() and 0.70 in completed


def load_magnitude_results() -> list[dict]:
    if not MAG_RESULTS_PATH.exists():
        return []
    with open(MAG_RESULTS_PATH, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"Expected list payload in {MAG_RESULTS_PATH}, found {type(payload)!r}")
    return [row for row in payload if isinstance(row, dict)]


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def magnitude_prefix_complete() -> bool:
    if not magnitude_checkpoint_path(0.20).exists():
        return False
    found = {float(row["target_s"]) for row in load_magnitude_results() if "target_s" in row}
    return all(target_s in found for target_s in PREFIX_TARGETS)


def run_exact_magnitude_prefix_if_needed() -> None:
    if hybrid_has_stage2_progress():
        log("hybrid run already progressed beyond s=0.20; skipping magnitude prefix rerun")
        return
    if magnitude_prefix_complete():
        log(f"exact magnitude prefix already complete at {MAG_RUN_DIR}")
        return

    ensure_dir(MAG_OUTPUT_ROOT)
    ensure_dir(MAG_RUN_DIR)
    ensure_dir(MAG_CHECKPOINT_DIR)

    log("starting stage 1: exact run_finetune_magnitude.py through s=0.20")
    sys.path.insert(0, str(OPTUNA_ROOT))
    import run_finetune_magnitude as mag

    mag.OUT_DIR = MAG_RUN_DIR
    mag.CKPT_DIR = MAG_CHECKPOINT_DIR
    mag.RESULTS_FILE = MAG_RESULTS_PATH
    mag.LOG_FILE = MAG_LOG_PATH
    mag.CYCLES = list(mag.CYCLES[: len(PREFIX_TARGETS)])
    ensure_dir(mag.OUT_DIR)
    ensure_dir(mag.CKPT_DIR)
    mag.main()

    if not magnitude_prefix_complete():
        raise RuntimeError("Exact magnitude prefix did not produce the expected s=0.20 checkpoint.")
    log("stage 1 complete")


def build_prefix_steps_from_magnitude_results() -> list[dict]:
    rows_by_target: dict[float, dict] = {}
    for row in load_magnitude_results():
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


def write_hybrid_prefix_checkpoint(target_s: float) -> None:
    src = magnitude_checkpoint_path(target_s)
    if not src.exists():
        raise FileNotFoundError(src)
    payload = torch.load(src, map_location="cpu", weights_only=False)
    converted = {
        "model_state_dict": payload["model_state_dict"],
        "protected_masks": {},
    }
    dst = keep_checkpoint_path(target_s)
    tmp = dst.with_suffix(".tmp")
    torch.save(converted, tmp)
    tmp.replace(dst)


def prepare_v8_handoff_if_needed() -> None:
    if hybrid_has_stage2_progress():
        log("hybrid run directory already has stage-2 progress; keeping existing handoff artifacts")
        return

    ensure_dir(OUTPUT_ROOT)
    ensure_dir(RUN_DIR)
    ensure_dir(CHECKPOINT_DIR)

    log("preparing v8 resume directory from exact magnitude checkpoints")
    for target_s in PREFIX_TARGETS:
        write_hybrid_prefix_checkpoint(target_s)

    results = {"steps": build_prefix_steps_from_magnitude_results()}
    save_json(RESULTS_PATH, results)
    save_json(
        HANDOFF_META_PATH,
        {
            "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "prefix_targets": PREFIX_TARGETS,
            "magnitude_prefix_run_dir": str(MAG_RUN_DIR),
            "v8_reference_run_dir": str(V8_REFERENCE_RUN_DIR),
        },
    )
    log(f"stage 1 handoff prepared in {RUN_DIR}")


def stage2_command() -> list[str]:
    return [
        sys.executable,
        "-u",
        str(REMOVED_MATRIX_DIR / "run_removed_matrix_audit_v8.py"),
        "--optuna-root",
        str(OPTUNA_ROOT),
        "--resume-run-dir",
        str(RUN_DIR),
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
        "2",
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


def run_exact_v8_stage_if_needed() -> None:
    if hybrid_stage2_complete():
        log("stage 2 already complete")
        return
    log("starting stage 2: exact v8 continuation")
    subprocess.run(stage2_command(), cwd=OPTUNA_ROOT, check=True)


def main() -> None:
    if not hybrid_has_stage2_progress():
        run_exact_magnitude_prefix_if_needed()
        prepare_v8_handoff_if_needed()
    else:
        log("existing hybrid run detected beyond prefix; resuming exact v8 stage directly")
    run_exact_v8_stage_if_needed()


if __name__ == "__main__":
    main()

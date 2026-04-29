#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import timm


HERE = Path(__file__).resolve().parent
OPTUNA_ROOT = HERE

sys.path.insert(0, str(OPTUNA_ROOT))
from theory_pruning import fit_mp, iter_target_layers_modules  # noqa: E402


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model-specific RMT caches for audit-stage pruning.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def compute_layer_stats(weight_2d: np.ndarray, k_hill: int = 20) -> dict[str, float | int | list[int]]:
    singular_values = np.linalg.svd(weight_2d.astype(np.float64), compute_uv=False)
    singular_values = singular_values[singular_values > 0]
    if singular_values.size < 2:
        raise ValueError("Need at least two nonzero singular values to compute RMT stats.")

    sigma_sq, lambda_plus, splus = fit_mp(weight_2d.astype(np.float64))
    k_spike = int((singular_values > splus).sum())
    stable_rank = float((singular_values ** 2).sum() / (singular_values[0] ** 2))

    prob = (singular_values ** 2) / (singular_values ** 2).sum()
    eff_rank = float(np.exp(-(prob * np.log(prob + 1.0e-12)).sum()))

    spike_energy = float((singular_values[:k_spike] ** 2).sum()) if k_spike > 0 else 0.0
    bulk_energy = float((singular_values[k_spike:] ** 2).sum())
    total_energy = float((singular_values ** 2).sum())
    snr = spike_energy / max(bulk_energy, 1.0e-12)

    eigs = singular_values ** 2
    eigs_sorted = np.sort(eigs)[::-1]
    k = min(k_hill, len(eigs_sorted) - 1)
    if k < 2:
        hill_alpha = 99.0
    else:
        log_top = np.log(eigs_sorted[:k])
        log_kth = np.log(max(eigs_sorted[k], 1.0e-12))
        hill_alpha = float(1.0 + 1.0 / max(np.mean(log_top - log_kth), 1.0e-6))

    return {
        "shape": [int(weight_2d.shape[0]), int(weight_2d.shape[1])],
        "sigma_sq": float(sigma_sq),
        "lambda_plus": float(lambda_plus),
        "splus": float(splus),
        "K_spike": k_spike,
        "stable_rank": stable_rank,
        "eff_rank": eff_rank,
        "snr": snr,
        "hill_alpha": hill_alpha,
        "spike_energy_frac": spike_energy / total_energy,
        "frob_norm": float(np.linalg.norm(weight_2d)),
        "top_sv_over_splus": float(singular_values[0] / splus),
    }


def build_cache(model_name: str, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    splus_path = cache_dir / "rmt_splus_metrics.json"
    stats_path = cache_dir / "rmt_layer_stats.json"

    log(f"loading model {model_name}")
    model = timm.create_model(model_name, pretrained=True)

    splus = {}
    layer_stats = {}

    for layer_name, module in iter_target_layers_modules(model):
        weight = module.weight.detach().cpu().numpy()
        weight_2d = weight.reshape(weight.shape[0], -1) if weight.ndim == 4 else weight
        stats = compute_layer_stats(weight_2d)
        splus[layer_name] = float(stats["splus"])
        layer_stats[layer_name] = stats
        log(
            f"cache {layer_name:38s} "
            f"splus={stats['splus']:.6f} "
            f"alpha={stats['hill_alpha']:.3f} "
            f"K={stats['K_spike']}"
        )

    with open(splus_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": model_name,
                "n_layers": len(splus),
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "splus": splus,
            },
            handle,
            indent=2,
        )

    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(layer_stats, handle, indent=2)

    log(f"wrote {splus_path}")
    log(f"wrote {stats_path}")


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    splus_path = cache_dir / "rmt_splus_metrics.json"
    stats_path = cache_dir / "rmt_layer_stats.json"
    if not args.force and splus_path.exists() and stats_path.exists():
        log(f"cache already exists at {cache_dir}")
        return
    build_cache(args.model_name, cache_dir)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
    main()

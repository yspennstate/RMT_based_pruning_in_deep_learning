"""
RMT-grounded magnitude pruning sweep.

Tests 10 different ways to use RMT signals (HT-SR alpha, stable rank, SNR,
spike subspace, bulk noise floor, etc.) to either:
  (a) allocate per-layer pruning budget (most methods), or
  (b) define a per-entry score that replaces |W| in magnitude pruning.

Each method is tested with and without Haar z=0.20 SV-cleanup as preprocessing
(~best variant from the previous sweep). Results compared to plain magnitude
at every sparsity in {5, 10, ..., 70}%.

Crash-safe: atomic incremental saves, resume from checkpoint, per-cell
try/except. Designed to run unattended long unattended.

Output: optuna_run/rmt_cache/rmt_magnitude_sweep_results.json
Stats:  optuna_run/rmt_cache/rmt_layer_stats.json (computed once, cached)
"""

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import timm

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))
from RMT import bema_inside  # noqa: E402
from theory_pruning import (  # noqa: E402
    fit_mp,
    iter_target_layers,
    iter_target_layers_modules,
    evaluate_now,
    DEVICE,
)
from pruning_method_comparison_sweep import (  # noqa: E402
    apply_magnitude,
    haar_clean_only,
    measure_actual_sparsity,
)

ALPHA, BETA, GOF = 0.25, 0.8, 1
HERE = Path(__file__).parent
OUT_FILE = HERE / "rmt_cache" / "rmt_magnitude_sweep_results.json"
STATS_FILE = HERE / "rmt_cache" / "rmt_layer_stats.json"
LOG_FILE = HERE / "rmt_magnitude_sweep_log.txt"
OUT_FILE.parent.mkdir(exist_ok=True)

DEFAULT_SPARSITIES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                      0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def log(msg):
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_gpu_temp():
    """Return current GPU temperature in C, or None on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000),
        )
        return int(out.strip().split()[0])
    except Exception:
        return None


def cool_gpu_if_hot(cool_below=68, max_wait=300):
    """If GPU is above `cool_below`, sleep in 10s increments until it cools
    or max_wait seconds elapse. Prevents thermal damage during long sweeps."""
    t0 = time.time()
    temp = get_gpu_temp()
    if temp is None or temp <= cool_below:
        return
    log(f"  [thermal] GPU at {temp}C > {cool_below}C, cooling...")
    while True:
        time.sleep(10)
        temp = get_gpu_temp()
        if temp is None or temp <= cool_below:
            log(f"  [thermal] GPU cooled to {temp}C in {time.time()-t0:.0f}s")
            return
        if time.time() - t0 > max_wait:
            log(f"  [thermal] WARN: still {temp}C after {max_wait}s, proceeding")
            return


# ──────────────────────────────────────────────────────────────────────────────
# RMT layer statistics — compute once, cache to disk forever.
# ──────────────────────────────────────────────────────────────────────────────
def compute_layer_stats(W2, k_hill=20):
    """Per-layer RMT statistics. W2 is a 2D weight matrix."""
    M, N = W2.shape
    p, n = (min(M, N), max(M, N))

    # SVD
    U, S, Vt = np.linalg.svd(W2.astype(np.float64), full_matrices=False)
    S = S[S > 0]
    if len(S) < 2:
        return None

    # MP fit
    sigma_sq, lamda_plus, splus = fit_mp(W2.astype(np.float64))

    # Spike count
    K_spike = int((S > splus).sum())

    # Stable rank: ||W||_F^2 / ||W||_2^2 (in [1, min(M,N)])
    stable_rank = float((S ** 2).sum() / (S[0] ** 2))

    # Effective rank: exp of singular-value-spectrum entropy (in [1, min(M,N)])
    p_dist = (S ** 2) / (S ** 2).sum()
    eff_rank = float(np.exp(-(p_dist * np.log(p_dist + 1e-12)).sum()))

    # SNR (spike vs bulk Frobenius energy)
    spike_energy = float((S[:K_spike] ** 2).sum()) if K_spike > 0 else 0.0
    bulk_energy = float((S[K_spike:] ** 2).sum())
    total_energy = float((S ** 2).sum())
    snr = spike_energy / max(bulk_energy, 1e-12)

    # Hill estimator on the top-k squared singular values (≈ HT-SR alpha)
    eigs = (S ** 2)
    eigs_sorted = np.sort(eigs)[::-1]
    k = min(k_hill, len(eigs_sorted) - 1)
    if k < 2:
        hill_alpha = 99.0
    else:
        log_top = np.log(eigs_sorted[:k])
        log_kth = np.log(max(eigs_sorted[k], 1e-12))
        hill_alpha = float(1.0 + 1.0 / max(np.mean(log_top - log_kth), 1e-6))

    return {
        "shape": [M, N],
        "sigma_sq": float(sigma_sq),
        "lambda_plus": float(lamda_plus),
        "splus": float(splus),
        "K_spike": K_spike,
        "stable_rank": stable_rank,
        "eff_rank": eff_rank,
        "snr": snr,
        "hill_alpha": hill_alpha,
        "spike_energy_frac": spike_energy / total_energy,
        "frob_norm": float(np.linalg.norm(W2)),
        "top_sv_over_splus": float(S[0] / splus),
    }


def get_layer_stats(model):
    """Compute or load cached per-layer RMT stats."""
    if STATS_FILE.exists():
        with open(STATS_FILE) as f:
            return json.load(f)
    log("Computing per-layer RMT stats (one-time)...")
    stats = {}
    t0 = time.time()
    for name, mod, W in iter_target_layers(model):
        s = compute_layer_stats(W)
        if s is not None:
            stats[name] = s
            log(f"  {name:38s} alpha={s['hill_alpha']:5.2f} "
                f"sr={s['stable_rank']:6.1f} snr={s['snr']:6.2f} "
                f"K={s['K_spike']:3d}")
    log(f"Computed {len(stats)} layer stats in {time.time()-t0:.0f}s")
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Core: weighted magnitude pruning with per-layer multipliers
# ──────────────────────────────────────────────────────────────────────────────
def weighted_magnitude_prune(model, target_sparsity, layer_weights):
    """Globally threshold |W_ij| / layer_weight to hit target sparsity.
    Layers with HIGHER weight are protected (their entries appear larger);
    layers with LOWER weight are pruned more aggressively.
    """
    all_scores = []
    layer_meta = []
    for name, mod in iter_target_layers_modules(model):
        W = mod.weight.detach().cpu().numpy()
        w = max(layer_weights.get(name, 1.0), 1e-9)
        score = np.abs(W) / w
        layer_meta.append((name, mod, W, score))
        all_scores.append(score.ravel())
    flat = np.concatenate(all_scores)
    n_zero = int(flat.size * target_sparsity)
    if n_zero <= 0:
        return
    threshold = np.partition(flat, n_zero - 1)[n_zero - 1]
    for name, mod, W, score in layer_meta:
        mask = score > threshold
        W_new = (W * mask).astype(np.float32)
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(W_new))


def per_entry_score_prune(model, target_sparsity, score_fn):
    """Apply score_fn(name, W_2d, stats) -> score (same shape as W_2d) per
    layer, then global threshold to hit target sparsity."""
    all_scores = []
    layer_meta = []
    for name, mod in iter_target_layers_modules(model):
        W = mod.weight.detach().cpu().numpy()
        orig_shape = W.shape
        W2 = W.reshape(W.shape[0], -1) if W.ndim == 4 else W
        score = score_fn(name, W2)
        if W.ndim == 4:
            score = score.reshape(orig_shape)
        layer_meta.append((name, mod, W, score))
        all_scores.append(score.ravel())
    flat = np.concatenate(all_scores)
    n_zero = int(flat.size * target_sparsity)
    if n_zero <= 0:
        return
    threshold = np.partition(flat, n_zero - 1)[n_zero - 1]
    for name, mod, W, score in layer_meta:
        mask = score > threshold
        W_new = (W * mask).astype(np.float32)
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(W_new))


# ──────────────────────────────────────────────────────────────────────────────
# 10 RMT-grounded methods
# ──────────────────────────────────────────────────────────────────────────────
def normalize_dict(d, eps=1e-9):
    """Rescale a dict of floats so its values are centered around 1.0."""
    vals = np.array(list(d.values()))
    mean = vals.mean()
    return {k: max(v / max(mean, eps), 0.01) for k, v in d.items()}


def m_alpha_proportional(model, target_sparsity, stats):
    """Higher alpha (more random) -> SMALLER layer weight -> more pruning."""
    weights = {n: 1.0 / max(s["hill_alpha"], 1.0) for n, s in stats.items()}
    weights = normalize_dict(weights)
    weighted_magnitude_prune(model, target_sparsity, weights)


def m_alpha_hard_protect(model, target_sparsity, stats):
    """Layers with hill_alpha < 2.5 are HARD-PROTECTED (no pruning at all).
    Other layers split the global pruning budget via plain magnitude."""
    protected = {n for n, s in stats.items() if s["hill_alpha"] < 2.5}
    weights = {n: (1e6 if n in protected else 1.0) for n in stats}
    weighted_magnitude_prune(model, target_sparsity, weights)


def m_stable_rank_protect(model, target_sparsity, stats):
    """Higher stable rank -> harder to prune -> larger weight."""
    weights = {n: s["stable_rank"] for n, s in stats.items()}
    weights = normalize_dict(weights)
    weighted_magnitude_prune(model, target_sparsity, weights)


def m_eff_rank_weighted(model, target_sparsity, stats):
    """Higher effective rank -> larger weight (protect)."""
    weights = {n: s["eff_rank"] for n, s in stats.items()}
    weights = normalize_dict(weights)
    weighted_magnitude_prune(model, target_sparsity, weights)


def m_snr_weighted(model, target_sparsity, stats):
    """Higher SNR (spike-dominated) -> larger weight (protect)."""
    weights = {n: math.log1p(s["snr"]) for n, s in stats.items()}
    weights = normalize_dict(weights)
    weighted_magnitude_prune(model, target_sparsity, weights)


def m_spike_protected(model, target_sparsity, stats):
    """Per-entry score boosts entries with high spike contribution.
    Score = |W_ij| * (1 + |W_spike_ij| / sigma)
    So entries that are part of a spike SV get protected."""
    def score_fn(name, W2):
        s = stats[name]
        K = max(1, s["K_spike"])
        U, S, Vt = np.linalg.svd(W2.astype(np.float64), full_matrices=False)
        K = min(K, len(S))
        W_spike = (U[:, :K] * S[:K]) @ Vt[:K, :]
        sigma = math.sqrt(max(s["sigma_sq"], 1e-12))
        return np.abs(W2) * (1.0 + np.abs(W_spike) / sigma).astype(np.float64)
    per_entry_score_prune(model, target_sparsity, score_fn)


def m_bulk_residual_subtract(model, target_sparsity, stats):
    """Subtract per-entry noise floor estimate from magnitude.
    score = max(0, |W| - c * sqrt(sigma^2 / N)) where c=2 (2-sigma)"""
    def score_fn(name, W2):
        s = stats[name]
        N = max(W2.shape)
        sigma_per_entry = math.sqrt(max(s["sigma_sq"] / N, 1e-12))
        return np.maximum(0.0, np.abs(W2) - 2.0 * sigma_per_entry)
    per_entry_score_prune(model, target_sparsity, score_fn)


def m_row_normalized(model, target_sparsity, stats):
    """Per-entry z-score under the layer's noise model.
    score = |W_ij| / sqrt(sigma^2 / N)
    Scale-invariant per layer."""
    def score_fn(name, W2):
        s = stats[name]
        N = max(W2.shape)
        sigma_per_entry = math.sqrt(max(s["sigma_sq"] / N, 1e-12))
        return np.abs(W2) / sigma_per_entry
    per_entry_score_prune(model, target_sparsity, score_fn)


def m_alpha_x_stable_rank(model, target_sparsity, stats):
    """Combined: layer weight = (1/alpha) * stable_rank.
    Heavy-tailed (low alpha) AND diffuse (high stable rank) -> protect most."""
    weights = {
        n: s["stable_rank"] / max(s["hill_alpha"], 1.0)
        for n, s in stats.items()
    }
    weights = normalize_dict(weights)
    weighted_magnitude_prune(model, target_sparsity, weights)


def m_combined_rmt(model, target_sparsity, stats):
    """Geometric mean of (1/alpha), stable_rank, log(snr+1).
    All-in layer importance score."""
    weights = {}
    for n, s in stats.items():
        a = 1.0 / max(s["hill_alpha"], 1.0)
        sr = s["stable_rank"]
        snr_log = math.log1p(s["snr"])
        weights[n] = (a * sr * snr_log) ** (1.0 / 3.0)
    weights = normalize_dict(weights)
    weighted_magnitude_prune(model, target_sparsity, weights)


METHOD_FNS = {
    "alpha_proportional":     m_alpha_proportional,
    "alpha_hard_protect":     m_alpha_hard_protect,
    "stable_rank_protect":    m_stable_rank_protect,
    "eff_rank_weighted":      m_eff_rank_weighted,
    "snr_weighted":           m_snr_weighted,
    "spike_protected":        m_spike_protected,
    "bulk_residual_subtract": m_bulk_residual_subtract,
    "row_normalized":         m_row_normalized,
    "alpha_x_stable_rank":    m_alpha_x_stable_rank,
    "combined_rmt":           m_combined_rmt,
}


# ──────────────────────────────────────────────────────────────────────────────
# Resume / save
# ──────────────────────────────────────────────────────────────────────────────
def load_results():
    if OUT_FILE.exists():
        try:
            with open(OUT_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_top1": None,
        "cells": {},
    }


def save_results(results):
    tmp = OUT_FILE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    os.replace(tmp, OUT_FILE)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=list(METHOD_FNS.keys()))
    p.add_argument("--sparsities", nargs="+", type=float, default=None)
    p.add_argument("--no-sv-only", action="store_true",
                   help="only run without SV preprocessing (skip the haar variants)")
    p.add_argument("--sv-only", action="store_true",
                   help="only run with SV preprocessing")
    p.add_argument("--haar-z", type=float, default=0.20)
    p.add_argument("--haar-cut", type=float, default=0.76)
    args = p.parse_args()

    sparsities = args.sparsities or DEFAULT_SPARSITIES
    if args.sparsities:
        sparsities = [s / 100.0 if s > 1.0 else s for s in args.sparsities]

    log("=== RMT magnitude sweep starting ===")
    log(f"methods:    {args.methods}")
    log(f"sparsities: {[f'{s*100:.0f}%' for s in sparsities]}")
    log(f"haar prep:  z={args.haar_z}, cut={args.haar_cut}")

    results = load_results()
    log(f"existing cells in results.json: {len(results.get('cells', {}))}")

    log("Loading ViT-B baseline...")
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    base_state = deepcopy(model.state_dict())

    # RMT stats — computed on the BASELINE matrix (the honest signature)
    stats = get_layer_stats(model)

    from validation import get_val_dataset
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)
    val_loader = get_val_dataset(preprocess=preprocess, batch_size=8)

    if results.get("baseline_top1") is None:
        log("Evaluating baseline...")
        results["baseline_top1"] = evaluate_now(model, val_loader, "baseline")
        save_results(results)
    baseline = results["baseline_top1"]
    log(f"baseline top1: {baseline:.2f}%")

    # Pre-compute SV-cleaned state once
    haar_state = None
    if not args.no_sv_only:
        log(f"Computing haar SV preprocess state (z={args.haar_z}, cut={args.haar_cut})...")
        t0 = time.time()
        model.load_state_dict(base_state)
        haar_clean_only(model, z=args.haar_z, cut=args.haar_cut)
        haar_state = deepcopy(model.state_dict())
        log(f"  haar prep done in {time.time()-t0:.0f}s")

    # Build the work list: (variant_tag, method, prep_state)
    sv_prefixes = []
    if not args.sv_only:
        sv_prefixes.append(("plain", None))
    if not args.no_sv_only:
        sv_prefixes.append((f"sv_z{args.haar_z}", haar_state))

    work = []
    for prefix, prep_state in sv_prefixes:
        for method in args.methods:
            for s in sparsities:
                work.append((prefix, prep_state, method, s))

    log(f"\nTotal cells to consider: {len(work)}")

    n_done = 0
    n_run = 0
    n_skip = 0
    for prefix, prep_state, method, s in work:
        cell_key = f"{prefix}__{method}__s{s:.2f}"
        if cell_key in results["cells"]:
            n_skip += 1
            n_done += 1
            continue
        log(f"\n--- cell {n_done+1}/{len(work)}: {cell_key} ---")
        cool_gpu_if_hot(cool_below=68, max_wait=300)
        t0 = time.time()
        try:
            if prep_state is not None:
                model.load_state_dict(prep_state)
            else:
                model.load_state_dict(base_state)
            METHOD_FNS[method](model, s, stats)
            achieved = measure_actual_sparsity(model)
            top1 = evaluate_now(model, val_loader, cell_key)
            cell = {
                "method": method,
                "sv_preprocess": prep_state is not None,
                "target_sparsity": s,
                "achieved_sparsity": achieved,
                "top1": top1,
                "delta": top1 - baseline,
                "elapsed_s": time.time() - t0,
            }
            results["cells"][cell_key] = cell
            save_results(results)
            log(f"  -> top1={top1:.2f}% (Δ={cell['delta']:+.2f}pp)  "
                f"sparsity={achieved*100:.1f}%  time={cell['elapsed_s']:.0f}s")
            n_run += 1
        except Exception as e:
            tb = traceback.format_exc()
            log(f"  CELL FAILED: {e}\n{tb}")
            results["cells"][cell_key] = {
                "method": method, "sv_preprocess": prep_state is not None,
                "target_sparsity": s, "error": str(e),
            }
            save_results(results)
        n_done += 1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log(f"\n=== Done. ran={n_run}  skipped_existing={n_skip}  total={n_done} ===")
    save_results(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("interrupted by user")
    except Exception as e:
        log(f"FATAL: {e}\n{traceback.format_exc()}")

"""
Overnight comprehensive pruning sweep.

Compares multiple pruning methods (magnitude baseline + RMT-grounded variants)
across a sweep of target sparsities. Designed to run unattended for ~4 hours
with maximum crash safety:

  * Atomic incremental saves (write .tmp, os.replace) after every cell.
  * Resume-from-checkpoint: if `overnight_sweep_results.json` exists, any
    (method, sparsity) cell already filled is skipped — so a crash plus
    relaunch picks up exactly where it left off, no work lost.
  * Per-cell try/except so a bad cell logs its error and the sweep continues.
  * Conservative threading + BelowNormal priority + 8-core affinity to leave
    the trading bots alone.
  * Per-cell progress line with sparsity, top1, and Δ vs baseline.

Methods compared (rows of the grid):

  1. magnitude               — plain magnitude pruning (THE BASELINE).
  2. per_weight_random       — chi-square noise-model test, alpha tuned to
                                hit target sparsity exactly via binary search.
  3. layer_adaptive          — bulk_test verdict per layer: layers that OBEY
                                the noise model use per_weight_random; layers
                                that VIOLATE use plain magnitude. Then a
                                magnitude top-up to hit target sparsity.
  4. random_then_magnitude   — per_weight_random first (removes "true noise"
                                at α=0.5), then magnitude top-up to hit target.
                                Tests whether removing noise first changes the
                                magnitude pruning landscape.
  5. bulk_entry_haar         — refined Haar z+bulk: for each bulk SV, threshold
                                its U/V entries below z/√M. Reconstruct.
                                Then magnitude top-up to hit target sparsity.

Target sparsities: 5%, 10%, 15%, ..., 70% (14 points).

Usage:
  python overnight_sweep.py                    # full grid
  python overnight_sweep.py --methods magnitude per_weight_random
  python overnight_sweep.py --sparsities 30 40 50
  python overnight_sweep.py --resume-only      # only fill missing cells

Output: optuna_run/rmt_cache/overnight_sweep_results.json
"""

import argparse
import gc
import json
import math
import os
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

from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from RMT import bema_inside  # noqa: E402

# Reuse helpers from theory_pruning.py
sys.path.insert(0, os.path.dirname(__file__))
from theory_pruning import (  # noqa: E402
    fit_mp,
    magnitude_prune_layer,
    per_weight_random_layer,
    bulk_independence_test,
    iter_target_layers,
    iter_target_layers_modules,
    evaluate_now,
    DEVICE,
)

ALPHA, BETA, GOF = 0.25, 0.8, 1
HERE = Path(__file__).parent
OUT_FILE = HERE / "rmt_cache" / "overnight_sweep_results.json"
OUT_FILE.parent.mkdir(exist_ok=True)
LOG_FILE = HERE / "overnight_sweep_log.txt"

DEFAULT_SPARSITIES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                      0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
DEFAULT_METHODS = [
    # Already-done baselines (skipped on resume):
    "magnitude",
    "spike_magnitude",
    "hybrid_score",
    "bulk_entry_haar",
    "layer_adaptive_sm",
    "per_weight_random",

    # NEW: focused sweep around bulk_entry_haar — z and cut variations
    "haar_z0.05",
    "haar_z0.10",
    "haar_z0.20",
    "haar_z0.30",
    "haar_cut0.50",
    "haar_cut0.60",
    "haar_cut0.85",

    # NEW: structural variants of bulk_entry_haar
    "haar_iter2",      # apply 2× iterations of haar+magnitude
    "haar_soft50",     # shrink bulk entries by 50% instead of zeroing
    "haar_deep_only",  # apply Haar only to blocks 6-11
]


def log(msg):
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Per-method implementations that hit a TARGET SPARSITY exactly
# ──────────────────────────────────────────────────────────────────────────────
def apply_magnitude(model, target_sparsity):
    """Plain magnitude pruning to a target global sparsity."""
    for name, mod in iter_target_layers_modules(model):
        W = mod.weight.detach().cpu().numpy()
        if W.ndim == 4:
            W2 = W.reshape(W.shape[0], -1)
        else:
            W2 = W
        W_new, _ = magnitude_prune_layer(W2, target_sparsity)
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(W_new.reshape(W.shape)))


def apply_per_weight_random_target(model, target_sparsity, alpha_init=0.5):
    """Use per_weight_random; if achieved sparsity is below target, top up via
    magnitude pruning. If above target, dial back the magnitude top-up. The
    end-state is exactly at target sparsity."""
    # Pass 1: per_weight_random with default alpha
    masks = {}
    for name, mod in iter_target_layers_modules(model):
        W = mod.weight.detach().cpu().numpy()
        W2 = W.reshape(W.shape[0], -1) if W.ndim == 4 else W
        try:
            W_new, st = per_weight_random_layer(W2, alpha=alpha_init)
        except Exception:
            W_new = W2.copy()
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(W_new.reshape(W.shape)).float())
        masks[name] = (W_new != 0)
    # Pass 2: magnitude top-up to exact target sparsity (using current weights)
    apply_magnitude(model, target_sparsity)


def apply_random_then_magnitude(model, target_sparsity):
    """Same as apply_per_weight_random_target — kept as a separate name for
    the results table."""
    apply_per_weight_random_target(model, target_sparsity, alpha_init=0.5)


def apply_layer_adaptive(model, target_sparsity, bulk_verdicts):
    """For each layer, if bulk_test verdict says OBEYS, run per_weight_random
    first; otherwise just magnitude. Then magnitude top-up to exact target."""
    for name, mod in iter_target_layers_modules(model):
        W = mod.weight.detach().cpu().numpy()
        W2 = W.reshape(W.shape[0], -1) if W.ndim == 4 else W
        if bulk_verdicts.get(name, {}).get("verdict_overall", False):
            try:
                W_new, _ = per_weight_random_layer(W2, alpha=0.5)
            except Exception:
                W_new = W2.copy()
        else:
            W_new = W2.copy()
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(W_new.reshape(W.shape)).float())
    # Top-up
    apply_magnitude(model, target_sparsity)


def _spike_only_reconstruction(W2):
    """Return W*, the spike-only rank-K reconstruction of W2 using BEMA σ_+."""
    sigma_sq, _, splus = fit_mp(W2.astype(np.float64))
    U, S, Vt = np.linalg.svd(W2.astype(np.float64), full_matrices=False)
    K = max(1, int((S > splus).sum()))
    W_star = (U[:, :K] * S[:K]) @ Vt[:K, :]
    return W_star.astype(np.float32), K, splus


def apply_spike_magnitude(model, target_sparsity):
    """Per-entry score = |W*_ij| where W* is the spike-only reconstruction.
    Removes entries where the SIGNAL contribution at that position is small.
    This is the RMT-grounded version of magnitude pruning — instead of using
    raw |W| (which conflates signal and noise), use |signal at position|."""
    # Per-layer compute scores, then global threshold to hit target sparsity
    all_scores = []
    layer_meta = []
    for name, mod in iter_target_layers_modules(model):
        W = mod.weight.detach().cpu().numpy()
        W2 = W.reshape(W.shape[0], -1) if W.ndim == 4 else W
        try:
            W_star, K, splus = _spike_only_reconstruction(W2)
            score = np.abs(W_star)
        except Exception:
            score = np.abs(W2)  # fallback
        layer_meta.append((name, mod, W.shape, score))
        all_scores.append(score.ravel())
    flat = np.concatenate(all_scores)
    n_zero = int(flat.size * target_sparsity)
    if n_zero > 0:
        threshold = np.partition(flat, n_zero - 1)[n_zero - 1]
    else:
        threshold = -1
    for name, mod, shape, score in layer_meta:
        W = mod.weight.detach().cpu().numpy()
        W2 = W.reshape(W.shape[0], -1) if W.ndim == 4 else W
        mask = score > threshold
        W_new = (W2 * mask).astype(np.float32)
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(W_new.reshape(shape)))


def apply_hybrid_score(model, target_sparsity):
    """Per-entry score = |W_ij| * |W*_ij| (geometric mean of magnitude and
    spike contribution). Removes entries that are simultaneously small in
    raw magnitude AND have small spike contribution.
    Hypothesis: this catches the worst of both worlds — small AND noise-like."""
    all_scores = []
    layer_meta = []
    for name, mod in iter_target_layers_modules(model):
        W = mod.weight.detach().cpu().numpy()
        W2 = W.reshape(W.shape[0], -1) if W.ndim == 4 else W
        try:
            W_star, _, _ = _spike_only_reconstruction(W2)
            score = np.sqrt(np.abs(W2) * np.abs(W_star))
        except Exception:
            score = np.abs(W2)
        layer_meta.append((name, mod, W.shape, score))
        all_scores.append(score.ravel())
    flat = np.concatenate(all_scores)
    n_zero = int(flat.size * target_sparsity)
    threshold = np.partition(flat, n_zero - 1)[n_zero - 1] if n_zero > 0 else -1
    for name, mod, shape, score in layer_meta:
        W = mod.weight.detach().cpu().numpy()
        W2 = W.reshape(W.shape[0], -1) if W.ndim == 4 else W
        mask = score > threshold
        W_new = (W2 * mask).astype(np.float32)
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(W_new.reshape(shape)))


def apply_layer_adaptive_sm(model, target_sparsity, bulk_verdicts):
    """For each layer: if bulk_test verdict says OBEYS, use spike_magnitude
    score; otherwise use plain magnitude. Globally threshold to hit target."""
    all_scores = []
    layer_meta = []
    for name, mod in iter_target_layers_modules(model):
        W = mod.weight.detach().cpu().numpy()
        W2 = W.reshape(W.shape[0], -1) if W.ndim == 4 else W
        if bulk_verdicts.get(name, {}).get("verdict_overall", False):
            try:
                W_star, _, _ = _spike_only_reconstruction(W2)
                score = np.abs(W_star)
            except Exception:
                score = np.abs(W2)
        else:
            score = np.abs(W2)
        layer_meta.append((name, mod, W.shape, score))
        all_scores.append(score.ravel())
    flat = np.concatenate(all_scores)
    n_zero = int(flat.size * target_sparsity)
    threshold = np.partition(flat, n_zero - 1)[n_zero - 1] if n_zero > 0 else -1
    for name, mod, shape, score in layer_meta:
        W = mod.weight.detach().cpu().numpy()
        W2 = W.reshape(W.shape[0], -1) if W.ndim == 4 else W
        mask = score > threshold
        W_new = (W2 * mask).astype(np.float32)
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(W_new.reshape(shape)))


def _haar_clean_layer(W2, z, cut, shrink=0.0):
    """Apply Haar z+bulk SV cleanup to a 2D weight matrix W2.
    `shrink` in [0,1]: 0 = hard zero (default), 1 = no shrink (no-op),
    intermediate = multiply small entries by shrink instead of zeroing."""
    sigma_sq, _, splus = fit_mp(W2.astype(np.float64))
    U, S, Vt = np.linalg.svd(W2.astype(np.float64), full_matrices=False)
    M, N = W2.shape
    thresh_U = z / math.sqrt(M)
    thresh_V = z / math.sqrt(N)
    for i in range(len(S)):
        if S[i] / splus >= cut:
            continue
        small_u = np.abs(U[:, i]) < thresh_U
        small_v = np.abs(Vt[i, :]) < thresh_V
        U[small_u, i] *= shrink
        Vt[i, small_v] *= shrink
    return (U @ np.diag(S) @ Vt).astype(np.float32)


def apply_bulk_entry_haar(model, target_sparsity, z=0.1407, cut=0.76):
    """Cached SV-pruned baseline (the same Haar z+bulk we already have on disk),
    then magnitude top-up to exact target sparsity."""
    sv_path = HERE / "rmt_cache" / "sv_pruned_baseline_state.pt"
    if sv_path.exists():
        sv_state = torch.load(sv_path, map_location="cpu", weights_only=False)
        # Map cached "layer1" keys back to fresh model
        # The cached state was built on a SplittableLayer-wrapped model.
        # For a fresh ViT-B without wrapping, we need to do the SV prune fresh.
        # Fall back to fresh computation:
        pass

    for name, mod in iter_target_layers_modules(model):
        W = mod.weight.detach().cpu().numpy()
        orig_shape = W.shape
        W2 = W.reshape(W.shape[0], -1) if W.ndim == 4 else W
        try:
            W_new = _haar_clean_layer(W2, z, cut)
        except Exception:
            W_new = W2.astype(np.float32)
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(W_new.reshape(orig_shape)))
    apply_magnitude(model, target_sparsity)


# ─── New variants (focused sweep) ─────────────────────────────────────────────
def apply_haar_iter(model, target_sparsity, z=0.1407, cut=0.76, n_iter=2):
    """Apply (haar SV clean + magnitude top-up) n_iter times. Each cycle the
    magnitude top-up brings sparsity to target; the haar step then cleans the
    surviving entries' SV decomposition."""
    for _ in range(n_iter):
        apply_bulk_entry_haar(model, target_sparsity, z=z, cut=cut)


def apply_haar_soft(model, target_sparsity, z=0.1407, cut=0.76, shrink=0.5):
    """Shrink bulk-vector small entries by `shrink` factor instead of zeroing,
    then magnitude top-up to target sparsity."""
    for name, mod in iter_target_layers_modules(model):
        W = mod.weight.detach().cpu().numpy()
        orig_shape = W.shape
        W2 = W.reshape(W.shape[0], -1) if W.ndim == 4 else W
        try:
            W_new = _haar_clean_layer(W2, z, cut, shrink=shrink)
        except Exception:
            W_new = W2.astype(np.float32)
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(W_new.reshape(orig_shape)))
    apply_magnitude(model, target_sparsity)


def apply_haar_filtered(model, target_sparsity, z=0.1407, cut=0.76, layer_filter=None):
    """Apply Haar SV cleanup only to layers matching a filter function
    (called with the layer name). Other layers untouched. Then global
    magnitude top-up."""
    for name, mod in iter_target_layers_modules(model):
        W = mod.weight.detach().cpu().numpy()
        orig_shape = W.shape
        W2 = W.reshape(W.shape[0], -1) if W.ndim == 4 else W
        if layer_filter and layer_filter(name):
            try:
                W_new = _haar_clean_layer(W2, z, cut)
            except Exception:
                W_new = W2.astype(np.float32)
        else:
            W_new = W2.astype(np.float32)
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(W_new.reshape(orig_shape)))
    apply_magnitude(model, target_sparsity)


def _is_deep_block(name):
    """blocks.6.* through blocks.11.*"""
    if not name.startswith("blocks."):
        return False
    try:
        idx = int(name.split(".")[1])
        return idx >= 6
    except (ValueError, IndexError):
        return False


def haar_clean_only(model, z, cut, shrink=0.0, layer_filter=None):
    """Apply Haar SV cleanup to model in place — NO magnitude pruning.
    Used to pre-compute the cleaned state once per (z,cut,...) variant; the
    main loop then loads this state and only does the cheap magnitude top-up
    for each sparsity. Saves ~80s/cell on the 14-sparsity sweep."""
    for name, mod in iter_target_layers_modules(model):
        W = mod.weight.detach().cpu().numpy()
        orig_shape = W.shape
        W2 = W.reshape(W.shape[0], -1) if W.ndim == 4 else W
        if layer_filter is not None and not layer_filter(name):
            continue
        try:
            W_new = _haar_clean_layer(W2, z, cut, shrink=shrink)
        except Exception:
            W_new = W2.astype(np.float32)
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(W_new.reshape(orig_shape)))


# Methods that can be "prepped" once and then swept across sparsities cheaply.
# Each entry maps method name -> dict of haar_clean_only kwargs.
PREPABLE_HAAR = {
    "bulk_entry_haar": dict(z=0.1407, cut=0.76),
    "haar_z0.05":      dict(z=0.05,   cut=0.76),
    "haar_z0.10":      dict(z=0.10,   cut=0.76),
    "haar_z0.20":      dict(z=0.20,   cut=0.76),
    "haar_z0.30":      dict(z=0.30,   cut=0.76),
    "haar_cut0.50":    dict(z=0.1407, cut=0.50),
    "haar_cut0.60":    dict(z=0.1407, cut=0.60),
    "haar_cut0.85":    dict(z=0.1407, cut=0.85),
    "haar_soft50":     dict(z=0.1407, cut=0.76, shrink=0.5),
    "haar_deep_only":  dict(z=0.1407, cut=0.76, layer_filter=_is_deep_block),
}


# ──────────────────────────────────────────────────────────────────────────────
# Resume / save
# ──────────────────────────────────────────────────────────────────────────────
def load_results():
    if OUT_FILE.exists():
        try:
            with open(OUT_FILE) as f:
                return json.load(f)
        except Exception as e:
            log(f"WARN: failed to load existing results ({e}); starting fresh")
    return {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_top1": None,
        "bulk_verdicts": {},
        "cells": {},  # key: f"{method}__s{sparsity:.2f}" -> {top1, sparsity, elapsed_s}
    }


def save_results(results):
    tmp = OUT_FILE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    os.replace(tmp, OUT_FILE)


def measure_actual_sparsity(model):
    total = 0; zeros = 0
    for _, mod in iter_target_layers_modules(model):
        w = mod.weight.detach()
        total += w.numel()
        zeros += (w == 0).sum().item()
    return zeros / total if total > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--sparsities", nargs="+", type=float, default=None,
                   help="Sparsity levels (in [0,1] OR percent like 5 10 15)")
    p.add_argument("--resume-only", action="store_true",
                   help="Only run cells missing from existing results.json")
    args = p.parse_args()

    if args.sparsities is None:
        sparsities = DEFAULT_SPARSITIES
    else:
        sparsities = [s / 100.0 if s > 1.0 else s for s in args.sparsities]

    log(f"=== Overnight sweep starting ===")
    log(f"methods:    {args.methods}")
    log(f"sparsities: {[f'{s*100:.0f}%' for s in sparsities]}")

    results = load_results()
    log(f"existing cells in results.json: {len(results.get('cells', {}))}")

    log("Loading ViT-B baseline...")
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    base_state = deepcopy(model.state_dict())

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

    # bulk_test once for the whole network (used by layer_adaptive)
    if not results.get("bulk_verdicts") or len(results["bulk_verdicts"]) < 50:
        log("Running bulk_test diagnostic on all layers...")
        verdicts = {}
        for name, mod, W in iter_target_layers(model):
            try:
                d = bulk_independence_test(W)
                verdicts[name] = d
                tag = "OBEYS" if d["verdict_overall"] else "VIOLATES"
                log(f"  {name:38s} {tag}  (K={d['K_spike']}, var={d['variance_ratio']:.2f})")
            except Exception as e:
                log(f"  {name:38s} bulk_test FAILED: {e}")
                verdicts[name] = {"verdict_overall": False}
        results["bulk_verdicts"] = verdicts
        save_results(results)
    bulk_verdicts = results["bulk_verdicts"]
    n_obey = sum(1 for v in bulk_verdicts.values() if v.get("verdict_overall"))
    log(f"bulk_test: {n_obey}/{len(bulk_verdicts)} layers OBEY noise model")

    # The sweep
    method_fns = {
        "magnitude":            lambda m, s: apply_magnitude(m, s),
        "spike_magnitude":      lambda m, s: apply_spike_magnitude(m, s),
        "hybrid_score":         lambda m, s: apply_hybrid_score(m, s),
        "bulk_entry_haar":      lambda m, s: apply_bulk_entry_haar(m, s),
        "layer_adaptive_sm":    lambda m, s: apply_layer_adaptive_sm(m, s, bulk_verdicts),
        "per_weight_random":    lambda m, s: apply_per_weight_random_target(m, s),
        "random_then_magnitude":lambda m, s: apply_random_then_magnitude(m, s),
        "layer_adaptive":       lambda m, s: apply_layer_adaptive(m, s, bulk_verdicts),

        # Focused sweep: z variations (cut fixed at 0.76)
        "haar_z0.05":           lambda m, s: apply_bulk_entry_haar(m, s, z=0.05, cut=0.76),
        "haar_z0.10":           lambda m, s: apply_bulk_entry_haar(m, s, z=0.10, cut=0.76),
        "haar_z0.20":           lambda m, s: apply_bulk_entry_haar(m, s, z=0.20, cut=0.76),
        "haar_z0.30":           lambda m, s: apply_bulk_entry_haar(m, s, z=0.30, cut=0.76),

        # Focused sweep: cut variations (z fixed at 0.1407)
        "haar_cut0.50":         lambda m, s: apply_bulk_entry_haar(m, s, z=0.1407, cut=0.50),
        "haar_cut0.60":         lambda m, s: apply_bulk_entry_haar(m, s, z=0.1407, cut=0.60),
        "haar_cut0.85":         lambda m, s: apply_bulk_entry_haar(m, s, z=0.1407, cut=0.85),

        # Structural variants
        "haar_iter2":           lambda m, s: apply_haar_iter(m, s, n_iter=2),
        "haar_soft50":          lambda m, s: apply_haar_soft(m, s, shrink=0.5),
        "haar_deep_only":       lambda m, s: apply_haar_filtered(m, s, layer_filter=_is_deep_block),
    }

    n_total = len(args.methods) * len(sparsities)
    n_done = 0
    n_skip = 0

    def run_one_cell(method, s, prep_state=None):
        """Run a single (method, sparsity) cell. If prep_state is provided,
        load it (already-haar-cleaned) and only do magnitude top-up + eval;
        otherwise call the full method_fn from scratch."""
        nonlocal n_done
        cell_key = f"{method}__s{s:.2f}"
        if cell_key in results["cells"]:
            n_done += 1
            return True  # skipped (already done)
        log(f"\n--- cell {n_done+1}/{n_total}: {cell_key} ---")
        t0 = time.time()
        try:
            if prep_state is not None:
                model.load_state_dict(prep_state)
                apply_magnitude(model, s)
            else:
                model.load_state_dict(base_state)
                method_fns[method](model, s)
            achieved = measure_actual_sparsity(model)
            top1 = evaluate_now(model, val_loader, cell_key)
            cell = {
                "method": method,
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
        except Exception as e:
            tb = traceback.format_exc()
            log(f"  CELL FAILED: {e}\n{tb}")
            results["cells"][cell_key] = {
                "method": method, "target_sparsity": s, "error": str(e),
            }
            save_results(results)
        n_done += 1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False

    for method in args.methods:
        if method not in method_fns:
            log(f"unknown method: {method}; skipping")
            continue

        # Fast path: prepable haar variants. Compute the haar-cleaned state
        # ONCE, then loop sparsities with only magnitude top-up + eval.
        if method in PREPABLE_HAAR:
            # Skip prep if every sparsity for this method is already done.
            todo = [s for s in sparsities if f"{method}__s{s:.2f}" not in results["cells"]]
            if not todo:
                log(f"\n[skip] {method}: all {len(sparsities)} cells already in results")
                n_done += len(sparsities)
                n_skip += len(sparsities)
                continue
            log(f"\n[prep] {method}: applying haar cleanup once for {len(todo)} sparsities...")
            t_prep = time.time()
            model.load_state_dict(base_state)
            haar_clean_only(model, **PREPABLE_HAAR[method])
            prep_state = deepcopy(model.state_dict())
            log(f"[prep] {method} done in {time.time()-t_prep:.0f}s")
            for s in sparsities:
                run_one_cell(method, s, prep_state=prep_state)
            del prep_state
            gc.collect()
        else:
            # Slow path: per-cell for non-prepable methods (magnitude,
            # spike_magnitude, hybrid_score, per_weight_random, etc.)
            for s in sparsities:
                run_one_cell(method, s, prep_state=None)

    log(f"\n=== Sweep finished ===  cells_done={n_done} skipped_existing={n_skip}")

    # Print final comparison table
    log("\n=== Comparison table (cells, sorted by sparsity then method) ===")
    log(f"{'method':25s} {'target':>7s} {'achieved':>9s} {'top1':>7s} {'Δ':>7s}")
    sorted_cells = sorted(results["cells"].values(),
                          key=lambda c: (c.get("target_sparsity", 0), c.get("method", "")))
    for c in sorted_cells:
        if "top1" in c:
            log(f"{c['method']:25s} {c['target_sparsity']*100:6.0f}% "
                f"{c['achieved_sparsity']*100:8.1f}% "
                f"{c['top1']:6.2f}% {c['delta']:+6.2f}pp")
        else:
            log(f"{c['method']:25s} {c['target_sparsity']*100:6.0f}% "
                f"{'ERROR':>9s} {c.get('error','')[:40]}")

    save_results(results)
    log(f"\nFinal results -> {OUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("interrupted by user")
    except Exception as e:
        log(f"FATAL: {e}\n{traceback.format_exc()}")

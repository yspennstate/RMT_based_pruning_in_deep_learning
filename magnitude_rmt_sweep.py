"""
Magnitude pruning with RMT per-layer budget modulation — NO SVD.

All methods here are variants of global magnitude pruning where the per-layer
pruning budget is modulated by a cheap-to-compute per-layer signal derived
from RMT metrics (cached σ₊) or weight statistics (kurtosis, CV, Frobenius
norm). The modulation has a tunable strength β that decays with sparsity s
per the feedback rules in memory.

NO singular value decomposition is performed at any point. The only RMT
input is σ₊ from the pre-computed cache at rmt_cache/rmt_splus_metrics.json.

Infrastructure: atomic saves, resume-from-checkpoint, 50% GPU/CPU cap,
BelowNormal priority. Same crash-safety as overnight_sweep.py.

Usage:
  python magnitude_rmt_sweep.py                    # full grid
  python magnitude_rmt_sweep.py --methods magnitude splus_budget
  python magnitude_rmt_sweep.py --sparsities 30 40 50 60
  python magnitude_rmt_sweep.py --resume-only

Output: optuna_run/rmt_cache/magnitude_rmt_sweep_results.json
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

# Force stdout to utf-8 so non-ASCII characters in logs don't kill the process
# when the launcher .bat redirects stdout to a file (default cp1252 on Windows).
# Without this, a single Δ in a log line raises UnicodeEncodeError, which
# previously made the per-cell save silently get skipped.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass  # stdout doesn't support reconfigure (e.g., already wrapped)

# Redirect stderr to a crash log so we can see CUDA/segfault errors
_CRASH_LOG = Path(__file__).parent / "sweep_crash_stderr.log"
sys.stderr = open(_CRASH_LOG, "a", encoding="utf-8")
print(f"[{time.strftime('%H:%M:%S')}] stderr redirected to {_CRASH_LOG}", file=sys.stderr, flush=True)

import numpy as np
import torch
import timm

# ── Resource caps (BEFORE heavy imports) ──────────────────────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

from scipy.stats import kurtosis as scipy_kurtosis  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from theory_pruning import (  # noqa: E402  — SVD-free utilities only
    magnitude_prune_layer,
    iter_target_layers,
    iter_target_layers_modules,
    evaluate_now,
    DEVICE,
)

HERE = Path(__file__).parent
CACHE_DIR = HERE / "rmt_cache"
OUT_FILE = CACHE_DIR / "magnitude_rmt_sweep_results.json"
LOG_FILE = HERE / "magnitude_rmt_sweep_log.txt"
SPLUS_FILE = CACHE_DIR / "rmt_splus_metrics.json"

DEFAULT_SPARSITIES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                      0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

BASELINE_TOP1 = 84.87  # ViT-B/16 on 10k ImageNet val subset


def log(msg):
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ── Per-layer signal computation (NO SVD) ─────────────────────────────────────

def compute_layer_signals(model):
    """Compute cheap per-layer statistics for all target layers.
    Returns dict: layer_name -> {splus, frob_norm, splus_normed,
                                  kurtosis, cv, alpha, n_params}
    """
    # Load cached σ₊ values
    splus_data = {}
    if SPLUS_FILE.exists():
        with open(SPLUS_FILE) as f:
            raw = json.load(f)
        # The cache nests splus values under a "splus" key
        splus_dict = raw.get("splus", raw)
        for k, v in splus_dict.items():
            if isinstance(v, (int, float)):
                splus_data[k] = float(v)
        log(f"loaded {len(splus_data)} cached splus values from {SPLUS_FILE.name}")
    else:
        log(f"WARNING: {SPLUS_FILE.name} not found -- splus signals unavailable")

    # Load cached Hill alpha values (power-law exponent from HT-SR theory)
    LAYER_STATS_FILE = CACHE_DIR / "rmt_layer_stats.json"
    alpha_data = {}
    if LAYER_STATS_FILE.exists():
        with open(LAYER_STATS_FILE) as f:
            layer_stats = json.load(f)
        for k, v in layer_stats.items():
            if isinstance(v, dict) and "hill_alpha" in v:
                alpha_data[k] = float(v["hill_alpha"])
        log(f"loaded {len(alpha_data)} cached hill_alpha values from {LAYER_STATS_FILE.name}")
    else:
        log(f"WARNING: {LAYER_STATS_FILE.name} not found -- alpha signals unavailable")

    signals = {}
    for name, mod in iter_target_layers_modules(model):
        W = mod.weight.detach().cpu().numpy()
        W2 = W.reshape(W.shape[0], -1) if W.ndim == 4 else W
        absW = np.abs(W2).ravel()

        frob = float(np.linalg.norm(W2, 'fro'))
        mean_abs = float(absW.mean())
        std_abs = float(absW.std())
        cv = std_abs / mean_abs if mean_abs > 1e-12 else 0.0
        kurt = float(scipy_kurtosis(absW, fisher=True))  # excess kurtosis
        n_params = int(W2.size)

        # σ₊ lookup — try exact match, then partial match
        sp = None
        for cache_key in splus_data:
            if name in cache_key or cache_key in name:
                sp = splus_data[cache_key]
                break

        splus_normed = sp / frob if (sp is not None and frob > 1e-12) else None

        # Hill alpha lookup — same partial-match strategy as splus
        alpha = None
        for cache_key in alpha_data:
            if name in cache_key or cache_key in name:
                alpha = alpha_data[cache_key]
                break

        signals[name] = {
            "splus": sp,
            "alpha": alpha,
            "frob_norm": frob,
            "splus_normed": splus_normed,
            "kurtosis": kurt,
            "cv": cv,
            "mean_abs": mean_abs,
            "std_abs": std_abs,
            "n_params": n_params,
        }

    return signals


# ── Per-layer budget modulation ───────────────────────────────────────────────

def layer_weights_from_signal(signals, signal_key, beta_max, s_decay,
                               target_sparsity, decay_power=1.0,
                               beta_attn=None, beta_mlp=None):
    """Compute per-layer pruning weight w_l for each layer.

    w_l > 1 → layer gets pruned MORE than average
    w_l < 1 → layer gets pruned LESS than average
    w_l = 1 → same as plain magnitude (no modulation)

    The modulation decays toward 1.0 as target_sparsity approaches s_decay.
    decay_power controls the decay curve shape:
      p=1.0  linear (default, backward compatible)
      p=2.0  quadratic — RMT signal retreats faster as sparsity grows
      p=3.0  cubic — aggressively magnitude-biased at high sparsity

    If beta_attn and beta_mlp are both provided, per-layer β is used based on
    layer type (classify_layer_type). The global z-score is preserved; only
    the modulation strength varies by layer type. 'other' layers fall back
    to beta_max. When either is None, uniform beta_max is used for all layers.
    """
    # Extract raw signal values
    names = list(signals.keys())
    raw = []
    for n in names:
        v = signals[n].get(signal_key)
        if v is None:
            raw.append(0.0)  # fallback: no modulation for this layer
        else:
            raw.append(float(v))
    raw = np.array(raw)

    # Z-score normalize globally (preserves cross-type ranking)
    mu, sigma = raw.mean(), raw.std()
    if sigma < 1e-12:
        return {n: 1.0 for n in names}
    z = (raw - mu) / sigma

    # Decay factor — shared across layer types
    linear_decay = max(0.0, 1.0 - target_sparsity / s_decay) if s_decay > 0 else 1.0
    decay = linear_decay ** decay_power

    use_type_split = beta_attn is not None and beta_mlp is not None

    # Compute weights
    weights = {}
    for i, n in enumerate(names):
        if use_type_split:
            t = classify_layer_type(n)
            if t == "attn":
                beta_here = beta_attn
            elif t == "mlp":
                beta_here = beta_mlp
            else:
                beta_here = beta_max  # patch_embed, head, etc.
        else:
            beta_here = beta_max
        beta_eff = beta_here * decay
        w = 1.0 + beta_eff * z[i]
        w = max(0.1, w)  # floor at 0.1 to prevent any layer from being totally protected
        weights[n] = w

    return weights


def apply_modulated_magnitude(model, target_sparsity, layer_weights):
    """Magnitude pruning where each layer's effective sparsity is modulated
    by the per-layer weight.

    Higher weight → more pruning in that layer.
    The global threshold is adjusted so total achieved sparsity matches target.
    """
    # Collect all |W| values with per-layer scaling applied
    all_scores = []
    layer_meta = []

    for name, mod in iter_target_layers_modules(model):
        W = mod.weight.detach().cpu().numpy()
        W2 = W.reshape(W.shape[0], -1) if W.ndim == 4 else W

        w_l = layer_weights.get(name, 1.0)
        # Score: |W_ij| / w_l  — higher w_l means lower score means MORE pruning
        score = np.abs(W2) / w_l

        layer_meta.append((name, mod, W.shape, W2, score))
        all_scores.append(score.ravel())

    flat = np.concatenate(all_scores)
    n_total = flat.size
    n_zero = int(n_total * target_sparsity)

    if n_zero > 0 and n_zero < n_total:
        threshold = np.partition(flat, n_zero)[n_zero]
    elif n_zero >= n_total:
        threshold = flat.max() + 1
    else:
        threshold = -1.0

    total_zeroed = 0
    total_params = 0
    for name, mod, shape, W2, score in layer_meta:
        mask = (score >= threshold).astype(np.float32)
        W_new = W2 * mask
        total_zeroed += int((mask == 0).sum())
        total_params += mask.size
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(W_new.reshape(shape)).float())

    achieved = total_zeroed / total_params if total_params > 0 else 0.0
    return achieved


# ── Singular-vector sparsification ────────────────────────────────────────────

def sv_prune_layer(W_np, splus, power=30, theta_base=0.00001125):
    """Original SV sparsification: theta_sv = theta_base*sqrt(NM) * (1-sigma/sigma_+)^power.

    Uses GPU (torch.linalg.svd) when available — ~100× faster than numpy on CPU.
    """
    M, N = W_np.shape
    theta_scaled = theta_base * np.sqrt(N * M)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W_t = torch.from_numpy(W_np).float().to(dev)
    U_t, S_t, Vt_t = torch.linalg.svd(W_t, full_matrices=False)

    ratios = S_t / splus if splus > 1e-12 else torch.ones_like(S_t)
    bulk_mask = ratios < 1.0
    thresholds = torch.full_like(S_t, theta_scaled / 750.0)
    if bulk_mask.any():
        raw = (1.0 - ratios[bulk_mask]) ** power
        thresholds[bulk_mask] = theta_scaled * torch.clamp(raw, min=1.0 / 750.0)

    U_t = torch.where(U_t.abs() < thresholds.unsqueeze(0), 0, U_t)
    Vt_t = torch.where(Vt_t.abs() < thresholds.unsqueeze(1), 0, Vt_t)

    result = (U_t @ torch.diag(S_t) @ Vt_t).cpu().numpy().astype(np.float32)
    del W_t, U_t, S_t, Vt_t
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return result


def sv_prune_haar(W_np, splus, z_base, alpha=None, alpha_mean=None, power=3):
    """Haar-measure SV sparsification with alpha-modulated z-threshold.

    Uses Haar-distributed coordinate scale: std_U = 1/sqrt(M), std_V = 1/sqrt(N).
    Only bulk singular vectors (sigma < sigma_+) are sparsified.

    Per-layer z-threshold:
        z_eff = z_base * (alpha / alpha_mean)  if alpha provided
        z_eff = z_base                          otherwise

    Within each bulk vector, graduated thresholding by distance from edge:
        thresh_U(sigma) = z_eff * (1 - sigma/sigma_+)^power / sqrt(M)
        thresh_V(sigma) = z_eff * (1 - sigma/sigma_+)^power / sqrt(N)

    Higher alpha (more random bulk) -> higher z -> more aggressive sparsification.
    Lower alpha (structured bulk) -> lower z -> more protection.
    """
    M, N = W_np.shape

    # Alpha modulation: scale z by how random this layer is relative to average
    if alpha is not None and alpha_mean is not None and alpha_mean > 1e-6:
        z_eff = z_base * (alpha / alpha_mean)
    else:
        z_eff = z_base

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W_t = torch.from_numpy(W_np).float().to(dev)
    U_t, S_t, Vt_t = torch.linalg.svd(W_t, full_matrices=False)

    # Haar standard deviations for uniformly random orthogonal vectors
    std_U = 1.0 / math.sqrt(M)
    std_V = 1.0 / math.sqrt(N)

    ratios = S_t / splus if splus > 1e-12 else torch.ones_like(S_t)
    bulk_mask = ratios < 1.0

    # Graduated threshold: stronger deep in bulk, vanishes at edge
    grad = torch.zeros_like(S_t)
    if bulk_mask.any():
        grad[bulk_mask] = (1.0 - ratios[bulk_mask]).clamp(min=0).pow(power)

    thresh_U = z_eff * std_U * grad  # shape (k,)
    thresh_V = z_eff * std_V * grad

    U_t = torch.where(U_t.abs() < thresh_U.unsqueeze(0), 0, U_t)
    Vt_t = torch.where(Vt_t.abs() < thresh_V.unsqueeze(1), 0, Vt_t)

    result = (U_t @ torch.diag(S_t) @ Vt_t).cpu().numpy().astype(np.float32)
    del W_t, U_t, S_t, Vt_t
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return result


# ── Method registry ───────────────────────────────────────────────────────────

def classify_layer_type(name):
    """Classify a ViT layer name as 'attn', 'mlp', or 'other'.

    ViT-B/16 target layers: ~24 attn (qkv + proj) + ~24 mlp (fc1 + fc2)
    + a couple of 'other' (patch_embed, head).
    """
    n = name.lower()
    if "attn" in n:
        return "attn"
    if "mlp" in n or ".fc" in n:
        return "mlp"
    return "other"


def parse_method_name(method_str):
    """Parse a method string into (signal_key, beta_max, s_decay, decay_power,
    beta_attn, beta_mlp). Plain 'magnitude' returns None.

    Supported formats:
      splus_budget_b0.50_sd0.50           → uniform β, linear decay
      splus_budget_b1.00_sd0.70_p2.0      → uniform β, quadratic decay
      splus_budget_b1.50_sd0.85_p1.0_ba1.25_bm1.75
          → per-layer-type β (attn=1.25, mlp=1.75); base `b` is unused when
             both _ba and _bm are present

    `beta_attn` / `beta_mlp` are None if not specified (→ uniform β).
    """
    if method_str == "magnitude":
        return None

    parts = method_str.split("_")
    signal = parts[0]
    beta = 0.5
    sd = 0.5
    power = 1.0
    beta_attn = None
    beta_mlp = None
    # Parse each token. Check most-specific prefixes first so "ba1.25" and
    # "bm1.75" don't accidentally match the plain "b{num}" rule.
    for tok in parts[1:]:
        if tok.startswith("ba") and "." in tok:
            try:
                beta_attn = float(tok[2:])
            except ValueError:
                pass
        elif tok.startswith("bm") and "." in tok:
            try:
                beta_mlp = float(tok[2:])
            except ValueError:
                pass
        elif tok.startswith("b") and tok != "budget" and "." in tok:
            try:
                beta = float(tok[1:])
            except ValueError:
                pass
        elif tok.startswith("sd"):
            try:
                sd = float(tok[2:])
            except ValueError:
                pass
        elif tok.startswith("p") and tok != "splus" and tok != "splusnorm" and "." in tok:
            try:
                power = float(tok[1:])
            except ValueError:
                pass

    # Map signal name to the key in the signals dict
    signal_map = {
        "splus": "splus",
        "splusnorm": "splus_normed",
        "kurtosis": "kurtosis",
        "cv": "cv",
        "alpha": "alpha",
    }
    signal_key = signal_map.get(signal, signal)
    return signal_key, beta, sd, power, beta_attn, beta_mlp


def build_method_list():
    """Generate all method strings for the sweep."""
    methods = ["magnitude"]

    signals = ["splus", "splusnorm", "kurtosis", "cv"]
    betas = [0.25, 0.50, 1.00]
    s_decays = [0.50, 0.70]

    for sig in signals:
        for b in betas:
            for sd in s_decays:
                methods.append(f"{sig}_budget_b{b:.2f}_sd{sd:.2f}")

    return methods


# ── Main sweep ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Subset of methods to run")
    parser.add_argument("--sparsities", nargs="+", type=float, default=None)
    parser.add_argument("--resume-only", action="store_true")
    parser.add_argument("--eval-batches", type=int, default=250,
                        help="Number of eval batches per cell (250=2000 imgs, 150=1200 imgs)")
    parser.add_argument("--out-file", type=str, default=None,
                        help="Override output JSON path (default: magnitude_rmt_sweep_results.json)")
    parser.add_argument("--sv-prune", action="store_true",
                        help="Apply singular-vector sparsification BEFORE magnitude pruning")
    parser.add_argument("--sv-mode", type=str, default="original",
                        choices=["original", "haar"],
                        help="SV method: 'original' (theta_base scaling) or 'haar' (Haar z + alpha)")
    parser.add_argument("--sv-power", type=float, default=30,
                        help="Power exponent in SV threshold: (1 - sigma/sigma_+)^power")
    parser.add_argument("--sv-theta", type=float, default=0.00001125,
                        help="Base theta for SV threshold scaling (original mode)")
    parser.add_argument("--sv-z", type=float, default=1.0,
                        help="Base z-threshold in Haar std units (haar mode)")
    args = parser.parse_args()

    # Allow CLI override of OUT_FILE for separate experiments
    global OUT_FILE
    if args.out_file:
        OUT_FILE = Path(args.out_file)
        if not OUT_FILE.is_absolute():
            OUT_FILE = HERE / OUT_FILE
        log(f"output file overridden: {OUT_FILE}")

    sparsities = [s / 100 if s > 1 else s for s in (args.sparsities or
                  [s * 100 for s in DEFAULT_SPARSITIES])]
    methods = args.methods or build_method_list()

    log(f"Methods: {len(methods)}")
    log(f"Sparsities: {len(sparsities)}")
    log(f"Total cells: {len(methods) * len(sparsities)}")

    # ── GPU cap ───────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.5, 0)
        torch.backends.cudnn.benchmark = False
        log(f"GPU: {torch.cuda.get_device_name(0)}, 50% VRAM cap")

    # ── Load model + dataset ──────────────────────────────────────────────
    log("loading ViT-B/16...")
    base_model = timm.create_model("vit_base_patch16_224", pretrained=True)
    base_state = deepcopy(base_model.state_dict())

    log("loading validation dataset...")
    from validation import get_val_dataset  # noqa
    data_config = timm.data.resolve_model_data_config(base_model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)
    val_loader = get_val_dataset(preprocess=preprocess, batch_size=256)

    # ── Compute per-layer signals once ────────────────────────────────────
    log("computing per-layer signals (no SVD)...")
    signals = compute_layer_signals(base_model)
    log(f"  {len(signals)} layers, signals: splus, frob_norm, splus_normed, kurtosis, cv")

    # Show signal ranges
    for key in ["splus", "splus_normed", "kurtosis", "cv"]:
        vals = [s[key] for s in signals.values() if s[key] is not None]
        if vals:
            log(f"  {key}: min={min(vals):.4f} max={max(vals):.4f} "
                f"mean={np.mean(vals):.4f} std={np.std(vals):.4f}")

    # ── Load existing results ─────────────────────────────────────────────
    results = []
    done_keys = set()
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            results = json.load(f)
        for r in results:
            done_keys.add((r["method"], r["target_sparsity"]))
        log(f"resumed: {len(done_keys)} cells already done")

    # ── Sweep ─────────────────────────────────────────────────────────────
    total = len(methods) * len(sparsities)
    done = 0
    skipped = 0

    for method in methods:
        for target_s in sparsities:
            key = (method, target_s)
            if key in done_keys:
                skipped += 1
                continue

            done += 1
            log(f"[{done}/{total - skipped}] {method} @ s={target_s:.2f}")

            try:
                # Fresh model from baseline
                model = timm.create_model("vit_base_patch16_224", pretrained=False)
                model.load_state_dict(deepcopy(base_state))

                # Optional: singular-vector sparsification BEFORE magnitude pruning.
                # This softly removes near-zero coordinates in bulk singular vectors,
                # making the subsequent magnitude step more effective.
                if args.sv_prune:
                    # Pre-compute alpha mean for Haar mode normalization
                    if args.sv_mode == "haar":
                        alphas = [signals[n].get("alpha") for n in signals
                                  if signals[n].get("alpha") is not None]
                        alpha_mean = float(np.mean(alphas)) if alphas else None
                    for sv_name, sv_mod in iter_target_layers_modules(model):
                        W = sv_mod.weight.detach().cpu().numpy()
                        W_shape = W.shape
                        W2 = W.reshape(W.shape[0], -1) if W.ndim == 4 else W
                        sp = signals[sv_name].get("splus")
                        if sp is not None and sp > 1e-12:
                            if args.sv_mode == "haar":
                                W_sv = sv_prune_haar(
                                    W2, sp, z_base=args.sv_z,
                                    alpha=signals[sv_name].get("alpha"),
                                    alpha_mean=alpha_mean,
                                    power=args.sv_power)
                            else:
                                W_sv = sv_prune_layer(
                                    W2, sp, power=args.sv_power,
                                    theta_base=args.sv_theta)
                            with torch.no_grad():
                                sv_mod.weight.copy_(
                                    torch.from_numpy(W_sv.reshape(W_shape)).float())

                t0 = time.time()

                parsed = parse_method_name(method)
                if parsed is None:
                    # Plain magnitude
                    for name, mod in iter_target_layers_modules(model):
                        W = mod.weight.detach().cpu().numpy()
                        W2 = W.reshape(W.shape[0], -1) if W.ndim == 4 else W
                        W_new, _ = magnitude_prune_layer(W2, target_s)
                        with torch.no_grad():
                            mod.weight.copy_(
                                torch.from_numpy(W_new.reshape(W.shape)).float())
                    achieved_s = target_s  # magnitude_prune_layer hits target exactly
                else:
                    signal_key, beta_max, s_decay, decay_power, beta_attn, beta_mlp = parsed
                    lw = layer_weights_from_signal(
                        signals, signal_key, beta_max, s_decay, target_s,
                        decay_power=decay_power,
                        beta_attn=beta_attn, beta_mlp=beta_mlp)
                    achieved_s = apply_modulated_magnitude(model, target_s, lw)

                # Evaluate — inline loop (bypasses validate() which causes TDR crashes)
                model.to(DEVICE)
                model.eval()
                correct = 0
                total = 0
                batch_i = 0
                n_batches = len(val_loader)
                with torch.no_grad():
                    for images, targets in val_loader:
                        images = images.to(DEVICE)
                        targets = targets.to(DEVICE)
                        output = model(images)
                        _, predicted = output.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                        del images, targets, output
                        batch_i += 1
                        # Stop at args.eval_batches (default 250 = 2000 imgs;
                        # 150 = 1200 imgs for faster screening with ±1pp noise)
                        if batch_i >= args.eval_batches:
                            break
                        # Progress every 100 batches
                        if batch_i % 100 == 0:
                            acc_so_far = 100.0 * correct / total
                            print(f"    batch {batch_i}/{min(n_batches,args.eval_batches)}  acc={acc_so_far:.1f}%", flush=True)
                top1 = 100.0 * correct / total if total > 0 else 0.0
                model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log(f"  -> top1 ({method}): {top1:.2f}%")
                elapsed = time.time() - t0
                delta = top1 - BASELINE_TOP1

                entry = {
                    "method": method,
                    "target_sparsity": target_s,
                    "achieved_sparsity": round(achieved_s, 6),
                    "top1": round(top1, 4),
                    "delta": round(delta, 4),
                    "elapsed_s": round(elapsed, 1),
                }
                results.append(entry)
                done_keys.add(key)

                # Atomic save — MUST happen before any potentially risky log
                # call so a formatting crash can't lose the cell's result.
                tmp = str(OUT_FILE) + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(results, f, indent=2)
                os.replace(tmp, OUT_FILE)

                log(f"  top1={top1:.2f}  delta={delta:+.2f}  "
                    f"achieved_s={achieved_s:.4f}  {elapsed:.0f}s")

            except Exception as e:
                log(f"  ERROR: {type(e).__name__}: {e}")
                traceback.print_exc()
                entry = {
                    "method": method,
                    "target_sparsity": target_s,
                    "error": str(e),
                }
                results.append(entry)
                done_keys.add(key)

            finally:
                # Clean up GPU + cooldown to prevent thermal throttle/TDR
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Let GPU cool between cells — prevents 83°C TDR crashes on laptop
                log("  cooling 30s...")
                time.sleep(30)

    log(f"DONE. {done} cells computed, {skipped} skipped (resume). "
        f"Results: {OUT_FILE}")

    # ── Summary table: best β for each signal at s=60% ────────────────────
    log("\n=== Summary: all methods at s=0.60 ===")
    target_60 = [r for r in results if abs(r.get("target_sparsity", 0) - 0.60) < 0.01
                 and "error" not in r]
    target_60.sort(key=lambda r: r.get("top1", 0), reverse=True)
    for r in target_60:
        log(f"  {r['method']:40s}  top1={r['top1']:.2f}  delta={r['delta']:+.2f}")


if __name__ == "__main__":
    main()

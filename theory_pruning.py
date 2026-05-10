"""
Theory-grounded pruning for trained ViTs.

Empirical result from prior experiments (see project_vit_sv_pruning_findings):
    The MP "bulk = noise" assumption DOES NOT hold for trained ViTs.
    Hard bulk zeroing (spike_only) and Gavish-Donoho shrinkage both
    severely reduce accuracy. Bulk singular vectors carry structured,
    task-relevant information even when their entries look random in
    aggregate. Any whole-bulk removal method is guaranteed to fail.

The removable noise component is not a clean low-frequency
band of the spectrum. It lives at finer granularity — at the level of
individual entries within the bulk vectors, conditional on the spike/bulk
decomposition. Methods in this script must respect that constraint.

Candidate modes:
    1. per_weight_random  — entry-level test against the noise model;
                            removes individual entries, not whole SVs
    2. layer_adaptive     — use bulk_test verdict per layer to decide
                            whether the layer can tolerate any bulk removal
    3. soft_shrink        — multiply each bulk SV by a factor in [0, 1]
                            from a randomness score (gradual, not on/off)

Negative controls:
    spike_only            — known to reduce accuracy; kept as a sanity check
    gavish_donoho         — same; kept to confirm GD is no better than σ_+

The bulk_test diagnostic evaluates per-layer agreement with the
noise-model assumption is even approximately true. Layers that fail it
should be left alone by any method that touches their bulk.

═════════════════════════════════════════════════════════════════════════════
                                  MODES
═════════════════════════════════════════════════════════════════════════════

[NEGATIVE CONTROLS — known to crash accuracy, kept to verify the failure]

1. spike_only             KNOWN TO FAIL
   For each layer:
       SVD: W = U S V^T
       Fit MP via BEMA -> sigma^2, sigma_+ (bulk edge)
       W' = sum_{k: s_k > sigma_+} s_k u_k v_k^T
   Crashes accuracy because bulk SVs carry structured info. Kept as a
   sanity check / negative control. Not intended for production pruning.

2. gavish_donoho          KNOWN TO FAIL
   Same as spike_only with the optimal Gavish-Donoho threshold instead.
   Same outcome: accuracy crashes. Kept as a negative control.

[REAL CANDIDATES — respect the empirical finding that bulk has structure]

3. per_weight_random      ONE hyperparameter (alpha)
   Fine-grained per-WEIGHT test (not per-SV):
       Decompose: W = W_spike + W_bulk via the MP-derived spike subspace
       Per entry: chi^2 statistic = |W_bulk_ij|^2 / (sigma^2 / N)
       p_random_ij = 1 - chi2_cdf(chi^2, df=1)
   Remove entries with p_random > alpha (default 0.5: above the median
   of the chi-square null is "more likely random than not").

   Bayesian interpretation: each W_ij gets a posterior over {random,
   structured}; the method thresholds the posterior.

4. bulk_test              DIAGNOSTIC, no pruning, no hyperparameters
   For each layer, compute R = W - W_spike (the "removed" part) and run
   randomness tests on it:
       - Shapiro-Wilk normality on standardized entries
       - D'Agostino K^2 normality
       - Anderson-Darling vs N(0, 1)
       - Variance ratio: should be ~1 if noise model is right
       - Row pair correlations: should be ~0 if rows are i.i.d.
       - Largest singular value of R: should be ~ sigma_+ by construction
   Identifies layers that approximately satisfy the theory. Layers where R fails
   the tests have structured "bulk" — the theory's assumption is violated
   there and any spike-based pruning will hurt accuracy.

═════════════════════════════════════════════════════════════════════════════
                                  USAGE
═════════════════════════════════════════════════════════════════════════════

  python theory_pruning.py --mode bulk_test                # fast, no eval
  python theory_pruning.py --mode spike_only --eval        # the key experiment
  python theory_pruning.py --mode gavish_donoho --eval
  python theory_pruning.py --mode per_weight_random --alpha 0.3 --eval
  python theory_pruning.py --mode all --eval               # everything

The --eval flag runs ImageNet validation (10k subset, ~3 min on the GPU).
Without --eval, the script records the entries each mode would remove and
saves per-layer statistics for pre-evaluation inspection.

Output: optuna_run/rmt_cache/theory_pruning_results.json
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import timm

# Conservative threading for shared-workstation runs.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from RMT import bema_inside  # noqa: E402

ALPHA, BETA, GOF = 0.25, 0.8, 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HERE = Path(__file__).parent
OUT_FILE = HERE / "rmt_cache" / "theory_pruning_results.json"
OUT_FILE.parent.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Core: MP fit (cached if possible)
# ──────────────────────────────────────────────────────────────────────────────
def fit_mp(W):
    """Fit MP via BEMA. Returns (sigma_sq, lambda_plus, sigma_plus_sv)."""
    M, N = W.shape
    p, n = (M, N) if M <= N else (N, M)
    gram = (W @ W.T / N) if M <= N else (W.T @ W / N)
    eig = np.sort(np.linalg.eigvalsh(gram))
    sigma_sq, lamda_plus, _ = bema_inside(p, n, eig, ALPHA, 0.8)
    return float(sigma_sq), float(lamda_plus), float(math.sqrt(N * lamda_plus))


def gavish_donoho_threshold(splus, M, N):
    """GD optimal hard threshold expressed in terms of sigma_+ (the MP bulk edge
    in singular value space). Both are in singular-value units and are
    directly comparable to the singular values of W."""
    p = min(M, N)
    n = max(M, N)
    beta = p / n
    lam = math.sqrt(2 * (beta + 1) + (8 * beta) / (beta + 1 + math.sqrt(beta ** 2 + 14 * beta + 1)))
    # sigma * sqrt(n) = splus / (1 + sqrt(beta))   from MP bulk edge formula
    return lam * splus / (1 + math.sqrt(beta))


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1 & 2: spike-only / Gavish-Donoho
# ──────────────────────────────────────────────────────────────────────────────
def reconstruct_above_threshold(W, threshold_sv):
    """Return the matrix obtained by keeping only singular components with
    s_k > threshold_sv. Also return K (# kept) and the signal/noise split."""
    W = W.astype(np.float64)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    K = int((S > threshold_sv).sum())
    if K == 0:
        return np.zeros_like(W).astype(np.float32), 0, 0.0
    W_spike = (U[:, :K] * S[:K]) @ Vt[:K, :]
    return W_spike.astype(np.float32), K, float(S[:K].sum())


def spike_only_layer(W):
    sigma_sq, lamda_plus, splus = fit_mp(W.astype(np.float64))
    W_new, K, _ = reconstruct_above_threshold(W, splus)
    return W_new, {"sigma_sq": sigma_sq, "splus": splus, "K": K, "threshold": splus}


def gavish_donoho_layer(W):
    M, N = W.shape
    sigma_sq, lamda_plus, splus = fit_mp(W.astype(np.float64))
    threshold = gavish_donoho_threshold(splus, M, N)
    W_new, K, _ = reconstruct_above_threshold(W, threshold)
    return W_new, {"sigma_sq": sigma_sq, "splus": splus, "K": K,
                   "threshold": threshold, "threshold_over_splus": threshold / splus}


# ──────────────────────────────────────────────────────────────────────────────
# Mode 3: per-weight randomness via spike/bulk decomposition
# ──────────────────────────────────────────────────────────────────────────────
def magnitude_prune_layer(W, target_sparsity):
    """Plain magnitude pruning to a target sparsity. Used as the apples-to-
    apples baseline for any other method that produces a sparsity."""
    flat = np.abs(W).ravel()
    n_zero = int(W.size * target_sparsity)
    if n_zero <= 0:
        return W.astype(np.float32), {"sparsity": 0.0, "n_removed": 0}
    threshold = np.partition(flat, n_zero - 1)[n_zero - 1]
    mask = np.abs(W) > threshold
    W_new = (W * mask).astype(np.float32)
    return W_new, {"sparsity": float(1 - mask.mean()), "n_removed": int((~mask).sum())}


def per_weight_random_layer(W, alpha=0.5):
    """
    Decompose W = W_spike + W_bulk via the MP spike subspace, then test each
    bulk-residual entry against the noise model entry-by-entry:

        z^2_ij = (W_bulk_ij)^2 / (sigma^2 / N)        (chi-square 1 under H0)
        p_random_ij = 1 - F_chi2(z^2_ij; df=1)

    Removes entries where p_random_ij > alpha (default 0.5). Returns the
    masked matrix, the mask, and per-layer summary stats.
    """
    W = W.astype(np.float64)
    M, N = W.shape
    sigma_sq, _, splus = fit_mp(W)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    K = max(1, int((S > splus).sum()))
    W_spike = (U[:, :K] * S[:K]) @ Vt[:K, :]
    W_bulk = W - W_spike

    var_per_entry = sigma_sq / N
    z2 = (W_bulk ** 2) / max(var_per_entry, 1e-12)
    p_random = 1.0 - stats.chi2.cdf(z2, df=1)
    keep_mask = p_random <= alpha          # keep entries that look NOT-random
    W_new = (W * keep_mask).astype(np.float32)
    n_removed = int((~keep_mask).sum())
    return W_new, {
        "sigma_sq": sigma_sq, "splus": splus, "K": K,
        "alpha": alpha,
        "n_removed": n_removed,
        "pct_removed": 100 * n_removed / W.size,
        "p_random_mean": float(p_random.mean()),
        "p_random_median": float(np.median(p_random)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Mode 4: bulk-residual independence diagnostic (no pruning)
# ──────────────────────────────────────────────────────────────────────────────
def bulk_independence_test(W, max_pairs=200, sample_size=5000):
    """Test whether the bulk residual R = W - W_spike actually behaves like
    i.i.d. noise. Returns a dict of statistics — one verdict per test."""
    W = W.astype(np.float64)
    M, N = W.shape
    sigma_sq, _, splus = fit_mp(W)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    K = int((S > splus).sum())
    W_spike = (U[:, :K] * S[:K]) @ Vt[:K, :] if K > 0 else np.zeros_like(W)
    R = W - W_spike

    # Standardize to N(0,1) under noise hypothesis
    sigma_per_entry = math.sqrt(max(sigma_sq / N, 1e-12))
    R_std = R / sigma_per_entry
    rng = np.random.default_rng(0)
    flat_sample = R_std.ravel()
    if flat_sample.size > sample_size:
        flat_sample = rng.choice(flat_sample, sample_size, replace=False)

    out = {
        "shape": list(W.shape),
        "sigma_sq": sigma_sq,
        "sigma_plus_sv": splus,
        "K_spike": K,
        "K_bulk": min(M, N) - K,
    }

    # 1. Marginal normality
    try:
        out["shapiro_p"] = float(stats.shapiro(flat_sample).pvalue)
    except Exception:
        out["shapiro_p"] = None
    try:
        out["dagostino_p"] = float(stats.normaltest(R_std.ravel()).pvalue)
    except Exception:
        out["dagostino_p"] = None
    try:
        out["anderson_stat"] = float(stats.anderson(flat_sample, dist="norm").statistic)
    except Exception:
        out["anderson_stat"] = None

    # 2. Variance check (should be ~1 after standardization)
    out["variance_ratio"] = float((R_std ** 2).mean())

    # 3. Row pair correlations (random rows should be ~uncorrelated)
    pair_cors = []
    for _ in range(max_pairs):
        i, j = rng.choice(M, 2, replace=False)
        c = np.corrcoef(R[i], R[j])[0, 1]
        if not np.isnan(c):
            pair_cors.append(c)
    if pair_cors:
        pair_cors = np.array(pair_cors)
        out["row_corr_mean"] = float(pair_cors.mean())
        out["row_corr_max_abs"] = float(np.abs(pair_cors).max())

    # 4. Largest SV of R (should be ≤ sigma_+ by construction; check anyway)
    R_top_sv = float(np.linalg.svd(R, compute_uv=False)[0]) if R.any() else 0.0
    out["R_top_sv"] = R_top_sv
    out["R_top_sv_over_splus"] = R_top_sv / max(splus, 1e-12)

    # 5. Verdict per test (boolean — does this layer obey the noise model?)
    out["verdict_normal"]  = (out["shapiro_p"] is not None and out["shapiro_p"] > 0.01)
    out["verdict_var"]     = (0.5 < out["variance_ratio"] < 2.0)
    out["verdict_indep"]   = (out.get("row_corr_max_abs", 1.0) < 0.3)
    out["verdict_topsv"]   = (out["R_top_sv_over_splus"] < 1.2)
    out["verdict_overall"] = all([out["verdict_normal"], out["verdict_var"],
                                  out["verdict_indep"], out["verdict_topsv"]])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────
def iter_target_layers(model):
    """Yield (name, module, weight_array) for every Linear/Conv2d in the model."""
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)):
            W = mod.weight.detach().cpu().numpy()
            if W.ndim == 4:
                W2 = W.reshape(W.shape[0], -1)
            else:
                W2 = W
            yield name, mod, W2


def iter_target_layers_modules(model):
    """Same as iter_target_layers but only yields (name, module). Used when
    module shapes are needed without loading the weight data."""
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)):
            yield name, mod


def apply_to_all(model, transform_fn, log_per_layer=True):
    """Run `transform_fn(W) -> (W_new, stats)` on every Linear/Conv2d, write
    W_new back into the model. Returns dict of per-layer stats."""
    layer_stats = {}
    for name, mod, W in iter_target_layers(model):
        orig_shape = mod.weight.shape
        W_new, st = transform_fn(W)
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(W_new.reshape(orig_shape)).float())
        layer_stats[name] = st
        if log_per_layer:
            keys = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in st.items() if k != "p_random_mean")
            print(f"  {name:38s}  {keys}")
    return layer_stats


def evaluate_now(model, val_loader, label):
    if val_loader is None:
        return None
    from validation import evaluate
    model.to(DEVICE)
    top1, _ = evaluate(val_loader, model, DEVICE)
    top1 = top1.cpu().item() if isinstance(top1, torch.Tensor) else float(top1)
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  -> top1 ({label}): {top1:.2f}%")
    return top1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["spike_only", "gavish_donoho",
                                       "per_weight_random", "bulk_test", "all"],
                   default="all")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="per_weight_random threshold (entries with p_random > alpha removed)")
    p.add_argument("--eval", action="store_true",
                   help="actually evaluate accuracy on ImageNet val subset")
    p.add_argument("--max-bulk-test-layers", type=int, default=10,
                   help="cap on layers to run bulk_test on (it's per-layer, not cumulative)")
    args = p.parse_args()

    print("Loading ViT-B baseline...")
    model = timm.create_model("vit_base_patch16_224", pretrained=True)

    val_loader = None
    baseline_top1 = None
    if args.eval:
        from validation import get_val_dataset
        data_config = timm.data.resolve_model_data_config(model)
        preprocess = timm.data.create_transform(**data_config, is_training=False)
        val_loader = get_val_dataset(preprocess=preprocess, batch_size=8)
        baseline_top1 = evaluate_now(model, val_loader, "baseline")

    results = {
        "mode": args.mode,
        "alpha": args.alpha,
        "baseline_top1": baseline_top1,
        "modes": {},
    }

    def save_results():
        """Atomic-ish incremental save: write to .tmp then rename. Called after
        every mode finishes so a crash mid-experiment never loses work."""
        tmp = OUT_FILE.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2, default=str)
        os.replace(tmp, OUT_FILE)
        print(f"  [save] {OUT_FILE.name} updated  ({len(results['modes'])} modes done)", flush=True)

    def run_mode(name, transform_fn):
        print(f"\n=== Mode: {name} ===", flush=True)
        snapshot = deepcopy(model.state_dict())
        layer_stats = apply_to_all(model, transform_fn)
        top1 = evaluate_now(model, val_loader, name) if args.eval else None
        results["modes"][name] = {
            "layer_stats": layer_stats,
            "top1": top1,
            "delta_top1": (top1 - baseline_top1) if (top1 is not None and baseline_top1 is not None) else None,
        }
        # Restore
        model.load_state_dict(snapshot)
        gc.collect()
        save_results()   # crash safety: persist after every mode

    # Persist baseline immediately so even an early crash leaves something useful
    save_results()

    if args.mode in ("spike_only", "all"):
        run_mode("spike_only", spike_only_layer)
        # Spike_only does NOT sparsify (low-rank dense reconstruction). No
        # matched-sparsity magnitude comparison is meaningful here; use
        # accuracy alone for this known negative control.
    if args.mode in ("gavish_donoho", "all"):
        run_mode("gavish_donoho", gavish_donoho_layer)

    if args.mode in ("per_weight_random", "all"):
        # Sweep several alpha values, record each one's actual achieved sparsity,
        # then run plain magnitude pruning at the same sparsity for a direct
        # apples-to-apples comparison.
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9] if args.mode == "all" else [args.alpha]
        for a in alphas:
            tag = f"per_weight_random_a{a}"
            run_mode(tag, lambda W, a=a: per_weight_random_layer(W, alpha=a))
            # Compute global achieved sparsity from the stored layer stats.
            stats_dict = results["modes"][tag]["layer_stats"]
            removed = sum(s.get("n_removed", 0) for s in stats_dict.values())
            total = sum(int(np.prod(s.get("shape", [0]))) if s.get("shape") else 0 for s in stats_dict.values())
            # The shape isn't in stats; use the model itself
            total_actual = sum(mod.weight.numel() for _, mod in iter_target_layers_modules(model))
            if total_actual > 0:
                achieved_sparsity = removed / total_actual
                results["modes"][tag]["achieved_sparsity"] = achieved_sparsity
                # Matched magnitude baseline at the same global sparsity
                mag_tag = f"magnitude_match_{tag}_s{achieved_sparsity:.3f}"
                print(f"\n  >> matched-sparsity magnitude baseline at s={achieved_sparsity:.3f}")
                run_mode(mag_tag, lambda W, s=achieved_sparsity: magnitude_prune_layer(W, s))
                results["modes"][mag_tag]["achieved_sparsity"] = achieved_sparsity

    if args.mode in ("bulk_test", "all"):
        print(f"\n=== Mode: bulk_test (diagnostic, first {args.max_bulk_test_layers} layers) ===", flush=True)
        bulk_results = {}
        n = 0
        for name, mod, W in iter_target_layers(model):
            if n >= args.max_bulk_test_layers:
                break
            d = bulk_independence_test(W)
            bulk_results[name] = d
            verdict = "OBEYS" if d["verdict_overall"] else "VIOLATES"
            print(f"  {name:38s}  K={d['K_spike']:3d}  "
                  f"shap_p={(d['shapiro_p'] or 0):.3f}  "
                  f"var={d['variance_ratio']:.2f}  "
                  f"max|corr|={d.get('row_corr_max_abs', 0):.3f}  "
                  f"R_top/sig+={d['R_top_sv_over_splus']:.2f}  -> {verdict}", flush=True)
            n += 1
            results["modes"]["bulk_test"] = bulk_results
            save_results()   # crash safety: persist after each layer
        results["modes"]["bulk_test"] = bulk_results

    save_results()
    print(f"\nFinal save -> {OUT_FILE}", flush=True)

    # Final comparison table after evaluation.
    if args.eval:
        print("\n=== Accuracy summary (sorted by Δ) ===")
        print(f"  baseline                              {baseline_top1:.2f}%")
        rows = []
        for mode_name, mode_res in results["modes"].items():
            if isinstance(mode_res, dict) and mode_res.get("top1") is not None:
                rows.append((mode_name, mode_res["top1"], mode_res["delta_top1"],
                             mode_res.get("achieved_sparsity")))
        # Group: theory mode followed immediately by its matched-sparsity baseline
        for mode_name, top1, delta, sp in rows:
            sp_str = f" sparsity={sp*100:.1f}%" if sp is not None else ""
            print(f"  {mode_name:38s} {top1:6.2f}%  (Δ = {delta:+6.2f}pp){sp_str}")


if __name__ == "__main__":
    main()

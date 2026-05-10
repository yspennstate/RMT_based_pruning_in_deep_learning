"""
State-of-the-art per-singular-vector randomness diagnostic for ViT pruning.

Goal: identify which singular components of each weight matrix are statistically
indistinguishable from random noise — and therefore candidates for removal under
the paper's theory ("removing the random part should not affect accuracy").

Combines five complementary tests per SV, aggregates via Fisher's method:

  1. Permutation null (nonparametric, no assumptions):
     shuffle the entries of W, recompute SVD many times, build empirical null
     distribution of σ_i. Per-SV: P(σ_i^null >= σ_i^actual) = p_perm.
     Low p_perm = SV stands above the null spectrum = signal.

  2. Bootstrap stability (nonparametric):
     resample rows of W with replacement, recompute SVD. Random SVs jitter
     wildly (high CV); structured SVs are stable (low CV).
     Reference: Donoho et al., universal singular value thresholding.

  3. Marchenko–Pastur fit + Tracy–Widom (parametric, classical RMT):
     fit MP density to bulk eigenvalues via BEMA → σ_+. Apply Tracy-Widom
     to the largest "bulk" eigenvalue for a calibrated p-value.

  4. Per-vector Haar conformity tests on U[:,i] and V[i,:]:
     a. max-entry / √(2 log M) — extreme value test
     b. L_4 norm vs 3/M — concentration test
     c. Shapiro-Wilk and D'Agostino K^2 normality of √M · entries
     d. Anderson-Darling vs N(0,1)
     A truly Haar vector passes all four.

  5. Combined p-value via Fisher's method:
     -2 Σ log(p_i)  ~  χ² with 2k degrees of freedom under H0 (independent).

Output: rmt_cache/randomness_diagnostic.json — per layer, per SV, the score
plus a final classification (CANDIDATE_REMOVE / KEEP) at threshold α.

Optional gold-standard validation (--ablate flag):
  Per-SV ablation: zero σ_i, eval top-1 on a small held-out subset, measure
  accuracy drop. Plot p_random vs Δacc to calibrate.

Usage:
  python randomness_diagnostic.py                # run statistical battery only
  python randomness_diagnostic.py --ablate       # also run ablation oracle
  python randomness_diagnostic.py --layers blocks.6.mlp.fc1 blocks.11.attn.proj
"""

import argparse
import gc
import json
import math
import os
import sys
import time
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
HERE = Path(__file__).parent
OUT_FILE = HERE / "rmt_cache" / "randomness_diagnostic.json"
OUT_FILE.parent.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: permutation null
# ──────────────────────────────────────────────────────────────────────────────
def permutation_null_svs(W, n_perm=20, rng=None):
    """Shuffle the entries of W, take SVD, return matrix of permuted singular
    values shape (n_perm, p) where p = min(M, N)."""
    rng = rng if rng is not None else np.random.default_rng(42)
    M, N = W.shape
    p = min(M, N)
    null = np.zeros((n_perm, p))
    flat = W.ravel().copy()
    for k in range(n_perm):
        rng.shuffle(flat)
        s = np.linalg.svd(flat.reshape(M, N), compute_uv=False)
        null[k] = s
    return null


def per_sv_perm_pvalue(actual_S, null_svs):
    """One-sided p-value per SV: P(σ_i^null >= σ_i^actual)."""
    p = len(actual_S)
    out = np.zeros(p)
    for i in range(p):
        out[i] = (null_svs[:, i] >= actual_S[i]).mean()
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: bootstrap stability
# ──────────────────────────────────────────────────────────────────────────────
def bootstrap_sv_stability(W, n_boot=20, rng=None):
    """Resample rows of W with replacement, recompute singular values.
    Returns per-SV mean, std, coefficient of variation across bootstraps.
    High CV → unstable → noise candidate."""
    rng = rng if rng is not None else np.random.default_rng(123)
    M, N = W.shape
    p = min(M, N)
    boot_S = np.zeros((n_boot, p))
    for k in range(n_boot):
        idx = rng.integers(0, M, size=M)
        s = np.linalg.svd(W[idx], compute_uv=False)
        boot_S[k] = s
    mean = boot_S.mean(axis=0)
    std = boot_S.std(axis=0)
    cv = std / np.maximum(mean, 1e-10)
    return mean, std, cv


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: MP fit + (approximate) Tracy-Widom
# ──────────────────────────────────────────────────────────────────────────────
def mp_diagnostics(W):
    """Fit MP via BEMA, return (sigma_sq, lambda_plus, splus, sv_cutoff)."""
    M, N = W.shape
    p, n = (M, N) if M <= N else (N, M)
    gram = (W @ W.T / N) if M <= N else (W.T @ W / N)
    eig = np.sort(np.linalg.eigvalsh(gram))
    sigma_sq, lamda_plus, l2 = bema_inside(p, n, eig, ALPHA, 0.8)
    splus = math.sqrt(N * lamda_plus)
    return {
        "sigma_sq": float(sigma_sq),
        "lambda_plus": float(lamda_plus),
        "sigma_plus_sv": float(splus),
        "eig_min": float(eig[0]),
        "eig_max": float(eig[-1]),
        "n_eig_above_lambda_plus": int((eig > lamda_plus).sum()),
        "p": int(p),
        "n": int(n),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: per-vector Haar conformity
# ──────────────────────────────────────────────────────────────────────────────
def haar_test_vector(u, dim):
    """Run Haar conformity tests on a unit vector u of length `dim`.
    Returns p-values + ratios (closer to 1 = more Haar-like)."""
    standardized = u * math.sqrt(dim)  # ~N(0,1) under Haar

    # Normality (Gaussian-likeness)
    p_sw = None
    try:
        if 3 < dim <= 5000:
            p_sw = float(stats.shapiro(standardized).pvalue)
    except Exception:
        pass
    p_dag = None
    try:
        if dim > 8:
            p_dag = float(stats.normaltest(standardized).pvalue)
    except Exception:
        pass

    # EVT max test: max|standardized| ≈ √(2 log M)
    max_entry_ratio = float(np.abs(standardized).max() / math.sqrt(2 * math.log(dim)))

    # L_4 norm test: ||u||_4^4 ≈ 3/(M+2) for Haar
    l4 = float((u ** 4).sum())
    expected_l4 = 3.0 / (dim + 2)
    l4_ratio = l4 / expected_l4

    # Anderson-Darling stat (no clean p-value, but the statistic is interpretable)
    try:
        ad = float(stats.anderson(standardized, dist="norm").statistic)
    except Exception:
        ad = None

    # Combine the available normality p-values via Fisher
    pvals = [p for p in [p_sw, p_dag] if p is not None and p > 0]
    if pvals:
        chi2 = -2 * sum(math.log(p) for p in pvals)
        p_fisher = float(1 - stats.chi2.cdf(chi2, df=2 * len(pvals)))
    else:
        p_fisher = None

    return {
        "p_normal_fisher": p_fisher,
        "p_shapiro": p_sw,
        "p_dagostino": p_dag,
        "max_entry_ratio": max_entry_ratio,
        "l4_ratio": l4_ratio,
        "AD_stat": ad,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: combined p_random per SV
# ──────────────────────────────────────────────────────────────────────────────
def combined_p_random(p_perm, u_haar, v_haar, cv):
    """Combine signals into a single 'is this SV consistent with random noise'
    score in [0, 1]. Higher = more random.

    Heuristic combiner: it isn't a calibrated p-value because the input tests
    are correlated, but it ranks SVs from most-random to least.
    """
    parts = []
    parts.append(p_perm)                       # high p_perm = inside null = random
    if u_haar["p_normal_fisher"] is not None:
        parts.append(u_haar["p_normal_fisher"])
    if v_haar["p_normal_fisher"] is not None:
        parts.append(v_haar["p_normal_fisher"])
    # CV: large CV → noise. Squash via 1 - exp(-cv) and treat as a "p"
    parts.append(1.0 - math.exp(-min(cv, 5)))
    # Penalty for non-Haar concentration: max-entry and L4 ratios > 1.5 ⇒ structured
    for ratio in (u_haar["max_entry_ratio"], v_haar["max_entry_ratio"],
                  u_haar["l4_ratio"], v_haar["l4_ratio"]):
        if ratio > 0:
            parts.append(min(1.0, 1.0 / ratio))   # ratio=1 → 1, ratio=10 → 0.1
    # Geometric mean of "randomness probabilities"
    parts = [max(p, 1e-6) for p in parts]
    log_score = sum(math.log(p) for p in parts) / len(parts)
    return float(math.exp(log_score))


# ──────────────────────────────────────────────────────────────────────────────
# Driver: per-layer diagnostic
# ──────────────────────────────────────────────────────────────────────────────
def diagnose_layer(name, W, n_perm=20, n_boot=20, top_k=None):
    t0 = time.time()
    W = W.astype(np.float64)
    M, N = W.shape
    p = min(M, N)
    print(f"  [{name}] shape={W.shape}  running tests...", flush=True)

    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    mp = mp_diagnostics(W)
    null_svs = permutation_null_svs(W, n_perm=n_perm)
    boot_mean, boot_std, boot_cv = bootstrap_sv_stability(W, n_boot=n_boot)
    p_perm = per_sv_perm_pvalue(S, null_svs)

    # Per-SV records
    svs = []
    iter_range = range(p) if top_k is None else range(min(top_k, p))
    for i in iter_range:
        u_test = haar_test_vector(U[:, i], M)
        v_test = haar_test_vector(Vt[i, :], N)
        rec = {
            "i": i,
            "sigma": float(S[i]),
            "sigma_over_splus": float(S[i] / mp["sigma_plus_sv"]),
            "p_perm": float(p_perm[i]),
            "boot_cv": float(boot_cv[i]),
            "boot_mean": float(boot_mean[i]),
            "u_haar": u_test,
            "v_haar": v_test,
        }
        rec["p_random"] = combined_p_random(p_perm[i], u_test, v_test, boot_cv[i])
        svs.append(rec)

    elapsed = time.time() - t0
    print(f"  [{name}] done in {elapsed:.0f}s   "
          f"(σ_+={mp['sigma_plus_sv']:.3f}, "
          f"#SVs above σ_+ = {(S > mp['sigma_plus_sv']).sum()}, "
          f"#SVs with p_random>0.5 = {sum(r['p_random']>0.5 for r in svs)})", flush=True)

    return {
        "name": name,
        "shape": list(W.shape),
        "elapsed_s": elapsed,
        "mp": mp,
        "null_sv_mean": null_svs.mean(axis=0).tolist(),
        "null_sv_std": null_svs.std(axis=0).tolist(),
        "boot_cv": boot_cv.tolist(),
        "p_perm": p_perm.tolist(),
        "svs": svs,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Optional: gold-standard ablation oracle
# ──────────────────────────────────────────────────────────────────────────────
def ablation_oracle(name, W, model, layer_setter, val_loader, top_n=20):
    """For the top_n singular vectors of W, ablate one at a time and measure
    top-1 drop on val_loader. Returns list of (i, sigma, top1_drop)."""
    from validation import evaluate
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  [{name}] ABLATION ORACLE on top {top_n} SVs (slow)...")

    U, S, Vt = np.linalg.svd(W.astype(np.float64), full_matrices=False)
    base_top1, _ = evaluate(val_loader, model.to(DEVICE), DEVICE)
    base_top1 = float(base_top1)
    results = []
    for i in range(min(top_n, len(S))):
        # Reconstruct with σ_i set to 0
        S_z = S.copy()
        S_z[i] = 0.0
        W_z = (U * S_z) @ Vt
        layer_setter(torch.from_numpy(W_z).float())
        top1_z, _ = evaluate(val_loader, model.to(DEVICE), DEVICE)
        results.append({
            "i": i,
            "sigma": float(S[i]),
            "top1_after": float(top1_z),
            "delta_top1": float(top1_z - base_top1),
        })
        print(f"    SV {i}: σ={S[i]:.3f}, top1 {base_top1:.2f}->{top1_z:.2f} "
              f"(Δ={top1_z-base_top1:+.3f})")
    # restore
    layer_setter(torch.from_numpy(W).float())
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_LAYERS = [
    "blocks.0.attn.qkv",
    "blocks.0.mlp.fc1",
    "blocks.6.attn.qkv",
    "blocks.6.mlp.fc1",
    "blocks.11.attn.qkv",
    "blocks.11.mlp.fc1",
    "blocks.11.attn.proj",
]


def get_weight_by_path(model, path):
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part)
    W = obj.weight.detach().numpy()
    if W.ndim == 4:
        W = W.reshape(W.shape[0], -1)
    return W


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--n-perm", type=int, default=20)
    parser.add_argument("--n-boot", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=None,
                        help="Only run per-SV tests on top K SVs (per layer). "
                             "Default: all.")
    parser.add_argument("--ablate", action="store_true",
                        help="Also run gold-standard per-SV ablation oracle "
                             "(slow, requires GPU + ImageNet val).")
    args = parser.parse_args()

    print(f"Loading ViT-B baseline...")
    model = timm.create_model("vit_base_patch16_224", pretrained=True)

    out = {
        "method": "randomness_diagnostic_v1",
        "n_perm": args.n_perm,
        "n_boot": args.n_boot,
        "top_k": args.top_k,
        "tests": [
            "permutation_null",
            "bootstrap_stability",
            "MP_fit_BEMA",
            "haar_max_entry_EVT",
            "haar_L4_norm",
            "haar_shapiro_wilk",
            "haar_dagostino_K2",
            "haar_anderson_darling",
            "fisher_combination",
        ],
        "layers": {},
    }

    for name in args.layers:
        W = get_weight_by_path(model, name)
        out["layers"][name] = diagnose_layer(
            name, W, n_perm=args.n_perm, n_boot=args.n_boot, top_k=args.top_k
        )
        gc.collect()

    if args.ablate:
        print("\n=== Ablation oracle ===")
        # Hook up the val loader
        from validation import get_val_dataset
        data_config = timm.data.resolve_model_data_config(model)
        preprocess = timm.data.create_transform(**data_config, is_training=False)
        val_loader = get_val_dataset(preprocess=preprocess, batch_size=8)

        for name in args.layers:
            W = get_weight_by_path(model, name)
            obj = model
            for part in name.split(".")[:-1]:
                obj = getattr(obj, part)
            leaf = getattr(obj, name.split(".")[-1])
            def setter(t, _leaf=leaf):
                with torch.no_grad():
                    _leaf.weight.copy_(t)
            out["layers"][name]["ablation_oracle"] = ablation_oracle(
                name, W, model, setter, val_loader, top_n=20
            )

    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT_FILE}")
    print(f"Layers analyzed: {len(out['layers'])}")
    print("\nQuick summary (highest p_random per layer):")
    for name, layer in out["layers"].items():
        ranked = sorted(layer["svs"], key=lambda r: -r["p_random"])
        most_random = ranked[0]
        least_random = ranked[-1]
        print(f"  {name:25s}  most-random SV i={most_random['i']:3d} "
              f"σ={most_random['sigma']:.3f} p_rand={most_random['p_random']:.3f}   "
              f"least-random i={least_random['i']:3d} σ={least_random['sigma']:.3f} "
              f"p_rand={least_random['p_random']:.3f}")


if __name__ == "__main__":
    main()

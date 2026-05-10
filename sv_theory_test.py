"""
Three theoretically-motivated SV pruning methods, compared against no SV pruning.

Method 1: "Separate U/V scaling"
    - θ_U(σ) = c/√M * max(floor, (1 - σ/σ_+)^power)   for left singular vectors
    - θ_V(σ) = c/√N * max(floor, (1 - σ/σ_+)^power)   for right singular vectors
    - Theoretically correct: entries of random unit vector in R^n are ~1/√n

Method 2: "Haar test"
    - For each bulk singular vector (σ < σ_+), test whether each entry is
      consistent with the Haar distribution on the unit sphere.
    - Under Haar: entries ~ N(0, 1/n) approximately. Prune entries with
      |entry| < z_thresh / √n where z_thresh is a z-score cutoff.
    - Entries that look "random" (small z-score) get pruned; entries that are
      unusually large (informative) are kept.

Method 3: "sqrt(NM) power=3" (previous best baseline, for reference)

All followed by magnitude pruning to cycle-8 equivalent (~67% kept).
"""

import sys, os, json, math, time, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import timm
import numpy as np
from scipy import stats

from pruning import count_nonzero_params, count_total_params, replace_layers, compute_layer_metrics_once
from SplittableLayers import SplittableConv, SplittableLinear
from validation import evaluate, get_val_dataset
from RMT import bema_inside

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ALPHA, BETA, GOF = 0.25, 0.8, 1
HP_B = 1.5
RHO = 0.06
TARGET_CYCLE = 8


def compute_splus(W_np):
    M, N = W_np.shape
    p = min(M, N)
    n = max(M, N)
    if M <= N:
        gram = W_np @ W_np.T / N
    else:
        gram = W_np.T @ W_np / N
    eigenvals = np.sort(np.linalg.eigvalsh(gram))
    sigma_sq, lamda_plus, l2 = bema_inside(p, n, eigenvals, ALPHA, 0.8)
    return math.sqrt(N * lamda_plus)


def sv_prune_separate_uv(W_np, splus, c_scale, power, floor):
    """Method 1: Separate thresholds for U (c/√M) and V (c/√N)."""
    U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
    M, N = W_np.shape

    theta_U_base = c_scale / math.sqrt(M)
    theta_V_base = c_scale / math.sqrt(N)

    for i in range(len(S)):
        ratio = S[i] / splus
        if ratio < 1.0:
            scale_factor = max(floor, (1.0 - ratio) ** power)
        else:
            scale_factor = floor

        thresh_U = theta_U_base * scale_factor
        thresh_V = theta_V_base * scale_factor

        U[:, i] = np.where(np.abs(U[:, i]) < thresh_U, 0, U[:, i])
        Vt[i, :] = np.where(np.abs(Vt[i, :]) < thresh_V, 0, Vt[i, :])

    return U @ np.diag(S) @ Vt


def sv_prune_haar(W_np, splus, z_thresh):
    """
    Method 2: Haar distribution test.
    For bulk singular vectors (σ < σ_+), entries of a Haar-distributed
    unit vector in R^n are approximately N(0, 1/n).
    Prune entries whose absolute value is below z_thresh standard deviations
    of this distribution (i.e., entries consistent with being random).
    For spike vectors (σ > σ_+), don't prune.
    """
    U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
    M, N = W_np.shape

    # Standard deviations under Haar
    std_U = 1.0 / math.sqrt(M)  # entries of unit vector in R^M
    std_V = 1.0 / math.sqrt(N)  # entries of unit vector in R^N

    for i in range(len(S)):
        if S[i] >= splus:
            continue  # spike — don't touch

        # Bulk: test each entry against Haar distribution
        # Prune if |entry| < z_thresh * std (looks random)
        thresh_U = z_thresh * std_U
        thresh_V = z_thresh * std_V

        U[:, i] = np.where(np.abs(U[:, i]) < thresh_U, 0, U[:, i])
        Vt[i, :] = np.where(np.abs(Vt[i, :]) < thresh_V, 0, Vt[i, :])

    return U @ np.diag(S) @ Vt


def sv_prune_sqrt_nm(W_np, splus, theta_base, power):
    """Method 3: Previous best — sqrt(NM) scaling, power=3."""
    U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
    M, N = W_np.shape
    theta_scaled = theta_base * math.sqrt(N * M)

    for i in range(len(S)):
        ratio = S[i] / splus
        if ratio < 1.0:
            dynamic_thresh = theta_scaled * max(1.0/750.0, (1.0 - ratio) ** power)
        else:
            dynamic_thresh = theta_scaled / 750.0

        U[:, i] = np.where(np.abs(U[:, i]) < dynamic_thresh, 0, U[:, i])
        Vt[i, :] = np.where(np.abs(Vt[i, :]) < dynamic_thresh, 0, Vt[i, :])

    return U @ np.diag(S) @ Vt


def magnitude_prune(model, cached_metrics):
    """Apply magnitude pruning surrogate for TARGET_CYCLE cycles."""
    for name, layer in model.named_modules():
        if not isinstance(layer, (SplittableConv, SplittableLinear)):
            continue
        metrics = cached_metrics.get(name)
        if metrics is None:
            continue

        mu_l = metrics['LinfError']
        gamma_l = metrics['percentage_less_than_splus'] / 100.0

        survival = 1.0
        for t in range(1, TARGET_CYCLE + 1):
            prune_frac = ((1 - mu_l) * gamma_l) ** (HP_B / t) * RHO
            prune_frac = min(prune_frac, 1.0)
            survival *= (1.0 - prune_frac)

        all_abs = []
        for sub in layer.modules():
            if isinstance(sub, (torch.nn.Conv2d, torch.nn.Linear)):
                w = sub.weight.data.abs()
                nz = w[w > 0]
                if len(nz) > 0:
                    all_abs.append(nz)

        if all_abs:
            all_abs = torch.cat(all_abs)
            n_keep = max(int(len(all_abs) * survival), 0)
            n_prune = len(all_abs) - n_keep
            if n_prune > 0:
                sorted_vals, _ = torch.sort(all_abs)
                threshold = sorted_vals[n_prune - 1].item()
                with torch.no_grad():
                    for sub in layer.modules():
                        if isinstance(sub, (torch.nn.Conv2d, torch.nn.Linear)):
                            mask = sub.weight.data.abs() > threshold
                            sub.weight.data *= mask.float()


def run_test(model_state, cached_metrics, val_loader, total_params, sv_func, label):
    """Run one test: SV prune -> magnitude prune -> evaluate."""
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    replace_layers(model, ALPHA, BETA, GOF, depth=0)
    model.load_state_dict(model_state)
    model.eval()

    t0 = time.time()

    # SV pruning (on CPU)
    if sv_func is not None:
        for name, layer in model.named_modules():
            if not isinstance(layer, (SplittableConv, SplittableLinear)):
                continue
            W = layer.get_matrix()
            splus = compute_splus(W)
            W_pruned = sv_func(W, splus)
            layer.set_params("layer1", torch.from_numpy(W_pruned).float(),
                           bias=None, change_bias=False)

    model.to(DEVICE)

    # Magnitude pruning
    magnitude_prune(model, cached_metrics)

    num_nonzero = count_nonzero_params(model)
    pct_kept = 100 * num_nonzero / total_params

    top1, _ = evaluate(val_loader, model, DEVICE)
    top1 = top1.cpu().item() if isinstance(top1, torch.Tensor) else top1
    dt = time.time() - t0

    del model
    torch.cuda.empty_cache()
    return top1, pct_kept, dt


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Three SV pruning methods + baseline\n")

    # Setup
    model = timm.create_model("vit_base_patch16_224", pretrained=True).to(DEVICE)
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)
    val_loader = get_val_dataset(preprocess=preprocess)

    replace_layers(model, ALPHA, BETA, GOF, depth=0)
    import pruning
    pruning._layer_metrics_cache = None
    cached_metrics = compute_layer_metrics_once(model)

    model.to(DEVICE)
    total_params = count_total_params(model)
    model_state = copy.deepcopy(model.state_dict())
    del model
    torch.cuda.empty_cache()

    # Define all test configs
    configs = []

    # Baseline
    configs.append(("No SV pruning (baseline)", None))

    # Method 1: Separate U/V scaling with different c and power values
    for c in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for power in [2, 3, 5]:
            floor = 0.01
            label = f"Sep U/V: c={c}, pow={power}"
            configs.append((label, lambda W, sp, c=c, p=power, f=floor:
                          sv_prune_separate_uv(W, sp, c, p, f)))

    # Method 2: Haar test with different z-score thresholds
    for z in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        label = f"Haar z={z}"
        configs.append((label, lambda W, sp, z=z: sv_prune_haar(W, sp, z)))

    # Method 3: sqrt(NM) power=3 (previous best baseline)
    configs.append(("sqrt(NM) pow=3 (prev best)",
                   lambda W, sp: sv_prune_sqrt_nm(W, sp, 0.00001125, 3)))

    print(f"Total configs: {len(configs)}")
    print(f"Estimated time: {len(configs) * 2:.0f} min\n")

    print(f"{'Method':<35}  {'Top-1%':>7}  {'Kept%':>7}  {'vs base':>8}  {'Time':>6}")
    print("-" * 70)

    results = []
    baseline_top1 = None

    for label, sv_func in configs:
        top1, pct_kept, dt = run_test(model_state, cached_metrics, val_loader, total_params,
                                       sv_func, label)
        if baseline_top1 is None:
            baseline_top1 = top1
        diff = top1 - baseline_top1
        marker = " ***" if diff > 0.1 else ""
        print(f"{label:<35}  {top1:7.2f}  {pct_kept:7.1f}  {diff:+8.2f}  {dt:5.0f}s{marker}")
        results.append({'method': label, 'top1': top1, 'pct_kept': pct_kept, 'diff': diff})

    # Print sorted summary
    print(f"\n{'='*70}")
    print("SORTED BY ACCURACY (best first)")
    print(f"{'='*70}")
    results_sorted = sorted(results, key=lambda x: x['top1'], reverse=True)
    for r in results_sorted[:15]:
        print(f"  {r['method']:<35}  {r['top1']:7.2f}%  ({r['diff']:+.2f}pp)")

    with open(os.path.join(os.path.dirname(__file__), 'sv_theory_test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to sv_theory_test_results.json")

"""
Grid search over SV pruning hyperparameters with power=3, sqrt(NM) fixed.

Formula:
    θ_sv(σ) = θ_base * sqrt(N*M) * max(floor, (1 - σ/σ_+)^3)

Two hyperparameters:
    θ_base: base threshold scaling (no rho in it)
    floor:  minimum pruning threshold as fraction of θ_base*sqrt(NM)

Grid:
    θ_base: [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
    floor:  [0, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05]

All at cycle-8 equivalent compression (~67% kept).
"""

import sys, os, json, math, time, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import timm
import numpy as np
import itertools

from pruning import count_nonzero_params, count_total_params, replace_layers, compute_layer_metrics_once
from SplittableLayers import SplittableConv, SplittableLinear
from validation import evaluate, get_val_dataset
from RMT import bema_inside

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ALPHA, BETA, GOF = 0.25, 0.8, 1
HP_B = 1.5
RHO = 0.06
TARGET_CYCLE = 8
POWER = 3


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


def sv_prune_layer(W_np, splus, theta_base, floor_val):
    """SV prune with power=3, sqrt(NM), given theta_base and floor."""
    U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
    M, N = W_np.shape
    theta_scaled = theta_base * math.sqrt(N * M)

    for i in range(len(S)):
        ratio = S[i] / splus
        if ratio < 1.0:
            dynamic_thresh = theta_scaled * max(floor_val, (1.0 - ratio) ** POWER)
        else:
            dynamic_thresh = theta_scaled * floor_val

        U[:, i] = np.where(np.abs(U[:, i]) < dynamic_thresh, 0, U[:, i])
        Vt[i, :] = np.where(np.abs(Vt[i, :]) < dynamic_thresh, 0, Vt[i, :])

    # Floor prune all
    if floor_val > 0:
        floor_thresh = theta_scaled * floor_val
        U = np.where(np.abs(U) < floor_thresh, 0, U)
        Vt = np.where(np.abs(Vt) < floor_thresh, 0, Vt)

    return U @ np.diag(S) @ Vt


def run_single(model_state, cached_metrics, val_loader, total_params, theta_base, floor_val):
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    replace_layers(model, ALPHA, BETA, GOF, depth=0)
    model.load_state_dict(model_state)
    model.eval()

    # SV pruning
    if theta_base > 0:
        for name, layer in model.named_modules():
            if not isinstance(layer, (SplittableConv, SplittableLinear)):
                continue
            W = layer.get_matrix()
            splus = compute_splus(W)
            W_pruned = sv_prune_layer(W, splus, theta_base, floor_val)
            layer.set_params("layer1", torch.from_numpy(W_pruned).float(),
                           bias=None, change_bias=False)

    model.to(DEVICE)

    # Magnitude pruning (surrogate for TARGET_CYCLE cycles)
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

    num_nonzero = count_nonzero_params(model)
    pct_kept = 100 * num_nonzero / total_params

    top1, top5 = evaluate(val_loader, model, DEVICE)
    top1 = top1.cpu().item() if isinstance(top1, torch.Tensor) else top1

    del model
    torch.cuda.empty_cache()
    return top1, pct_kept


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"SV HP grid: power={POWER}, sqrt(NM), cycle-{TARGET_CYCLE} compression\n")

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

    # Baseline (no SV)
    t0 = time.time()
    top1_base, pct_base = run_single(model_state, cached_metrics, val_loader, total_params, 0, 0)
    print(f"Baseline (no SV): Top-1={top1_base:.2f}%, Kept={pct_base:.1f}% ({time.time()-t0:.0f}s)\n")

    # Grid
    theta_vals = [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
    floor_vals = [0, 1e-4, 1e-3, 5e-3, 0.01, 0.05]

    grid = list(itertools.product(theta_vals, floor_vals))
    print(f"Grid: {len(theta_vals)} theta x {len(floor_vals)} floor = {len(grid)} points")
    print(f"Estimated time: {len(grid) * 120 / 60:.0f} min\n")

    print(f"{'theta':>10}  {'floor':>8}  {'Top-1%':>7}  {'Kept%':>7}  {'vs base':>8}  {'Time':>6}")
    print("-" * 55)

    results = []
    best_score = top1_base
    best_config = None

    for idx, (theta, floor) in enumerate(grid):
        t0 = time.time()
        top1, pct_kept = run_single(model_state, cached_metrics, val_loader, total_params, theta, floor)
        dt = time.time() - t0
        diff = top1 - top1_base
        marker = " ***" if top1 > best_score else ""
        if top1 > best_score:
            best_score = top1
            best_config = (theta, floor)
        print(f"{theta:10.1e}  {floor:8.4f}  {top1:7.2f}  {pct_kept:7.1f}  {diff:+8.2f}  {dt:5.0f}s{marker}")
        results.append({
            'theta_base': theta, 'floor': floor,
            'top1': top1, 'pct_kept': pct_kept, 'diff': diff,
        })

    print(f"\nBaseline: {top1_base:.2f}%")
    if best_config:
        print(f"Best: theta={best_config[0]:.1e}, floor={best_config[1]:.4f} -> {best_score:.2f}% (+{best_score-top1_base:.2f}pp)")

    with open(os.path.join(os.path.dirname(__file__), 'sv_hp_grid_results.json'), 'w') as f:
        json.dump({'baseline': top1_base, 'results': results}, f, indent=2)
    print("Saved to sv_hp_grid_results.json")

"""
Grid search over the power in the SV pruning formula:
    θ_sv(σ) = θ_base * max(1/750, (1 - σ/σ_+)^power)

Test different powers: 1, 2, 3, 5, 10, 15, 20, 30
For each: SV prune singular vectors -> reconstruct -> magnitude prune -> evaluate.
All at cycle-8 equivalent compression.
"""

import sys, os, json, math, time, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import timm
import numpy as np

from pruning import count_nonzero_params, count_total_params, replace_layers, compute_layer_metrics_once
from SplittableLayers import SplittableConv, SplittableLinear
from validation import evaluate, get_val_dataset
from RMT import bema_inside

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ALPHA, BETA, GOF = 0.25, 0.8, 1
HP_B = 1.5
RHO = 0.06
TARGET_CYCLE = 8
THETA_BASE = 0.00001125  # from the paper: 0.00001125 * sqrt(N*M) or * N*M


def sv_prune_layer(W_np, splus, power, theta_base_scaled):
    """
    Prune singular vectors using the paper's formula with variable power.
    θ_sv(σ) = theta * max(1/750, (1 - σ/σ_+)^power)  for σ < σ_+
    θ_sv(σ) = theta / 750                               for σ >= σ_+
    """
    U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
    k = len(S)

    for i in range(k):
        ratio = S[i] / splus
        if ratio < 1.0:
            dynamic_thresh = theta_base_scaled * max(1.0/750.0, (1.0 - ratio) ** power)
        else:
            dynamic_thresh = theta_base_scaled / 750.0

        U[:, i] = np.where(np.abs(U[:, i]) < dynamic_thresh, 0, U[:, i])
        Vt[i, :] = np.where(np.abs(Vt[i, :]) < dynamic_thresh, 0, Vt[i, :])

    # Also floor prune at theta/750 for all
    floor_thresh = theta_base_scaled / 750.0
    U = np.where(np.abs(U) < floor_thresh, 0, U)
    Vt = np.where(np.abs(Vt) < floor_thresh, 0, Vt)

    # Reconstruct
    return U @ np.diag(S) @ Vt


def compute_splus(W_np):
    """Compute σ_+ for a weight matrix."""
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


def run_single(model_state, cached_metrics, val_loader, total_params, power, theta_scale):
    """
    1. Load fresh model
    2. SV prune with given power (if power > 0)
    3. Magnitude prune to cycle-8 equivalent
    4. Evaluate
    """
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    replace_layers(model, ALPHA, BETA, GOF, depth=0)
    model.load_state_dict(model_state)
    model.eval()

    # Step 1: SV pruning
    if power > 0:
        for name, layer in model.named_modules():
            if not isinstance(layer, (SplittableConv, SplittableLinear)):
                continue

            W = layer.get_matrix()
            M_dim, N_dim = W.shape
            splus = compute_splus(W)

            # Scale theta by sqrt(N*M) as user requested
            theta_scaled = THETA_BASE * theta_scale(N_dim, M_dim)

            W_pruned = sv_prune_layer(W, splus, power, theta_scaled)
            layer.set_params("layer1", torch.from_numpy(W_pruned).float(),
                           bias=None, change_bias=False)

    model.to(DEVICE)

    # Step 2: Magnitude pruning (surrogate)
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
    print(f"SV power grid search — cycle-{TARGET_CYCLE} compression\n")

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

    # Test configurations
    # theta_scale: function(N, M) -> scaling factor
    sqrt_scale = lambda N, M: math.sqrt(N * M)
    nm_scale = lambda N, M: N * M

    configs = [
        (0,  'none',  'No SV (baseline)'),
        (1,  'sqrt',  'power=1, sqrt(NM)'),
        (2,  'sqrt',  'power=2, sqrt(NM)'),
        (3,  'sqrt',  'power=3, sqrt(NM)'),
        (5,  'sqrt',  'power=5, sqrt(NM)'),
        (10, 'sqrt',  'power=10, sqrt(NM)'),
        (20, 'sqrt',  'power=20, sqrt(NM)'),
        (30, 'sqrt',  'power=30, sqrt(NM)'),
        (1,  'nm',    'power=1, N*M'),
        (2,  'nm',    'power=2, N*M'),
        (3,  'nm',    'power=3, N*M'),
        (5,  'nm',    'power=5, N*M'),
        (10, 'nm',    'power=10, N*M'),
    ]

    print(f"{'Config':<25}  {'Top-1%':>7}  {'Kept%':>7}  {'Time':>6}")
    print("-" * 55)

    results = []
    for power, scale_name, label in configs:
        theta_scale = sqrt_scale if scale_name == 'sqrt' else nm_scale
        t0 = time.time()
        top1, pct_kept = run_single(model_state, cached_metrics, val_loader, total_params,
                                     power, theta_scale)
        dt = time.time() - t0
        print(f"{label:<25}  {top1:7.2f}  {pct_kept:7.1f}  {dt:5.0f}s")
        results.append({'power': power, 'scale': scale_name, 'label': label,
                       'top1': top1, 'pct_kept': pct_kept})

    with open(os.path.join(os.path.dirname(__file__), 'sv_power_grid_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to sv_power_grid_results.json")

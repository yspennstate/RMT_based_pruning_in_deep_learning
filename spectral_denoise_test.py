"""
Spectral denoising + magnitude pruning.
Theory: MP fit estimates noise variance per layer. Optimal singular value
shrinkage removes the random component in the spectral domain. Then magnitude
pruning cleans up remaining small entries.

Compare:
  1. No denoising (magnitude pruning only) — baseline
  2. Hard bulk zeroing (zero all σ ≤ σ_+)
  3. Soft shrinkage (Gavish-Donoho style optimal shrinkage)
  4. Partial shrinkage (shrink bulk by factor α, debias spikes)

All at the same target compression (cycle-8 equivalent ~67% kept).
"""

import sys, os, json, math, time, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import timm
import numpy as np

from pruning import count_nonzero_params, count_total_params, replace_layers, compute_layer_metrics_once
from SplittableLayers import SplittableConv, SplittableLinear
from validation import evaluate, get_val_dataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ALPHA, BETA, GOF = 0.25, 0.8, 1
HP_B = 1.5
RHO = 0.06
TARGET_CYCLE = 8


def compute_mp_params(layer):
    """Compute MP parameters for a layer: sigma_sq, splus, gamma."""
    W = layer.get_matrix()
    M, N = W.shape
    p = min(M, N)
    n = max(M, N)
    gamma = p / n

    if M <= N:
        gram = W @ W.T / N
    else:
        gram = W.T @ W / N

    eigenvals = np.linalg.eigvalsh(gram)
    eigenvals = np.sort(eigenvals)

    from RMT import bema_inside
    sigma_sq, lamda_plus, l2 = bema_inside(p, n, eigenvals, ALPHA, 0.8)
    splus = math.sqrt(N * lamda_plus)

    return {
        'sigma_sq': sigma_sq,
        'splus': splus,
        'gamma': gamma,
        'M': M,
        'N': N,
    }


def denoise_layer(layer, method, mp_params):
    """
    Apply spectral denoising to a layer's weight matrix.
    Returns the denoised weight matrix as a torch tensor.
    """
    W = layer.get_matrix()  # numpy array
    M, N = W.shape
    sigma_sq = mp_params['sigma_sq']
    splus = mp_params['splus']
    gamma = mp_params['gamma']

    U, S, Vt = np.linalg.svd(W, full_matrices=False)

    if method == 'none':
        return  # no denoising

    elif method == 'hard_bulk_zero':
        # Zero all singular values below σ_+
        S_new = np.where(S > splus, S, 0.0)

    elif method == 'soft_shrink':
        # Optimal shrinkage (Gavish-Donoho style)
        # For spikes (σ > σ_+): debias using σ_new = sqrt(σ² - σ²_noise*(1+γ))
        # where σ²_noise = N * sigma_sq (the noise singular value scale)
        # For bulk (σ ≤ σ_+): set to zero
        noise_sv_sq = N * sigma_sq * (1 + gamma)  # expected noise contribution to σ²
        S_new = np.zeros_like(S)
        for i in range(len(S)):
            if S[i] > splus:
                debiased_sq = S[i]**2 - noise_sv_sq
                if debiased_sq > 0:
                    S_new[i] = math.sqrt(debiased_sq)
                else:
                    S_new[i] = 0.0
            else:
                S_new[i] = 0.0

    elif method.startswith('partial_shrink_'):
        # Shrink bulk by factor α, debias spikes
        alpha = float(method.split('_')[-1])
        noise_sv_sq = N * sigma_sq * (1 + gamma)
        S_new = np.zeros_like(S)
        for i in range(len(S)):
            if S[i] > splus:
                # Debias the spike
                debiased_sq = S[i]**2 - noise_sv_sq
                if debiased_sq > 0:
                    S_new[i] = math.sqrt(debiased_sq)
                else:
                    S_new[i] = S[i] * alpha
            else:
                # Shrink bulk
                S_new[i] = S[i] * alpha

    elif method.startswith('bulk_shrink_'):
        # Only shrink bulk singular values by factor α, leave spikes untouched
        alpha = float(method.split('_')[-1])
        S_new = S.copy()
        for i in range(len(S)):
            if S[i] <= splus:
                S_new[i] = S[i] * alpha

    elif method == 'spike_debias_only':
        # Only debias spikes, leave bulk untouched
        noise_sv_sq = N * sigma_sq * (1 + gamma)
        S_new = S.copy()
        for i in range(len(S)):
            if S[i] > splus:
                debiased_sq = S[i]**2 - noise_sv_sq
                if debiased_sq > 0:
                    S_new[i] = math.sqrt(debiased_sq)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Reconstruct
    W_new = (U * S_new[None, :]) @ Vt if U.shape[0] >= U.shape[1] else U @ (S_new[:, None] * Vt)
    # Actually for SVD: W = U @ diag(S) @ Vt, and U is M x k, Vt is k x N
    W_new = U @ np.diag(S_new) @ Vt

    return torch.from_numpy(W_new).float()


def run_single(model_state, cached_metrics, val_loader, total_params, denoise_method):
    """
    1. Load fresh model
    2. Denoise each layer spectrally
    3. Magnitude prune to cycle-8 equivalent compression
    4. Evaluate
    """
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    replace_layers(model, ALPHA, BETA, GOF, depth=0)
    model.load_state_dict(model_state)
    model.eval()

    # Step 1: Spectral denoising (on CPU, layer by layer)
    if denoise_method != 'none':
        for name, layer in model.named_modules():
            if not isinstance(layer, (SplittableConv, SplittableLinear)):
                continue
            mp_params = compute_mp_params(layer)
            W_denoised = denoise_layer(layer, denoise_method, mp_params)
            if W_denoised is not None:
                # Set the denoised weights back
                layer.set_params("layer1", W_denoised, bias=None, change_bias=False)

    model.to(DEVICE)

    # Step 2: Magnitude pruning (surrogate for TARGET_CYCLE cycles)
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
    print(f"Spectral denoising + magnitude pruning test")
    print(f"Target: cycle-{TARGET_CYCLE} equivalent compression\n")

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

    top1_base, _ = evaluate(val_loader, model, DEVICE)
    top1_base = top1_base.cpu().item() if isinstance(top1_base, torch.Tensor) else top1_base
    print(f"Baseline (unpruned): Top-1={top1_base:.2f}%\n")
    del model
    torch.cuda.empty_cache()

    # Methods to test
    methods = [
        ('none', 'No denoising (baseline)'),
        ('spike_debias_only', 'Debias spikes only'),
        ('bulk_shrink_0.9', 'Shrink bulk by 0.9'),
        ('bulk_shrink_0.7', 'Shrink bulk by 0.7'),
        ('bulk_shrink_0.5', 'Shrink bulk by 0.5'),
        ('bulk_shrink_0.3', 'Shrink bulk by 0.3'),
        ('bulk_shrink_0.0', 'Shrink bulk to 0 (= hard zero)'),
        ('partial_shrink_0.9', 'Debias spikes + shrink bulk 0.9'),
        ('partial_shrink_0.7', 'Debias spikes + shrink bulk 0.7'),
        ('partial_shrink_0.5', 'Debias spikes + shrink bulk 0.5'),
        ('partial_shrink_0.3', 'Debias spikes + shrink bulk 0.3'),
        ('soft_shrink', 'Full optimal shrinkage (debias + zero bulk)'),
    ]

    print(f"{'Method':<40}  {'Top-1%':>7}  {'Kept%':>7}  {'Time':>6}")
    print("-" * 65)

    results = []
    for method_key, method_name in methods:
        t0 = time.time()
        top1, pct_kept = run_single(model_state, cached_metrics, val_loader, total_params, method_key)
        dt = time.time() - t0
        print(f"{method_name:<40}  {top1:7.2f}  {pct_kept:7.1f}  {dt:5.0f}s")
        results.append({
            'method': method_key,
            'name': method_name,
            'top1': top1,
            'pct_kept': pct_kept,
        })

    with open(os.path.join(os.path.dirname(__file__), 'spectral_denoise_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to spectral_denoise_results.json")

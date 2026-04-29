"""
Grid search over SV pruning threshold.
For each threshold: SV prune -> coefficient prune to cycle-8 equivalent -> evaluate.
Compare against no-SV baseline at the same compression.
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
TARGET_CYCLE = 8  # Simulate 8 cycles of pruning in one shot


def run_single(model_state, cached_metrics, val_loader, total_params, sv_threshold_mult):
    """
    1. Load fresh model from cached state
    2. If sv_threshold_mult > 0: apply SV pruning with that threshold
    3. Apply coefficient pruning (surrogate: cumulative survival over TARGET_CYCLE cycles)
    4. Evaluate
    """
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    replace_layers(model, ALPHA, BETA, GOF, depth=0)
    model.load_state_dict(model_state)
    model.to(DEVICE)
    model.eval()

    # Step 1: SV pruning (if threshold > 0)
    sv_zeros_added = 0
    if sv_threshold_mult > 0:
        for name, layer in model.named_modules():
            if not isinstance(layer, (SplittableConv, SplittableLinear)):
                continue
            W = layer.get_matrix()
            M_dim, N_dim = W.shape
            sv_threshold = sv_threshold_mult  # Direct threshold value
            # Run split (SV pruning via SVD)
            result, splus, LinfError, pct = layer.split(1, sv_threshold)

    # Move back to GPU after SV pruning (split() uses numpy/CPU)
    model.to(DEVICE)

    # Step 2: Coefficient pruning — surrogate for TARGET_CYCLE cycles
    for name, layer in model.named_modules():
        if not isinstance(layer, (SplittableConv, SplittableLinear)):
            continue
        metrics = cached_metrics.get(name)
        if metrics is None:
            continue

        mu_l = metrics['LinfError']
        gamma_l = metrics['percentage_less_than_splus'] / 100.0

        # Cumulative survival over TARGET_CYCLE cycles
        survival = 1.0
        for t in range(1, TARGET_CYCLE + 1):
            prune_frac = ((1 - mu_l) * gamma_l) ** (HP_B / t) * RHO
            prune_frac = min(prune_frac, 1.0)
            survival *= (1.0 - prune_frac)

        # Collect nonzero weights and prune smallest
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
    print(f"SV threshold grid search — coefficient pruning to cycle {TARGET_CYCLE} equivalent")
    print(f"Parameters: b={HP_B}, rho={RHO}\n")

    # Setup
    model = timm.create_model("vit_base_patch16_224", pretrained=True).to(DEVICE)
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)
    val_loader = get_val_dataset(preprocess=preprocess)

    replace_layers(model, ALPHA, BETA, GOF, depth=0)
    cached_metrics = compute_layer_metrics_once(model)
    model.to(DEVICE)
    total_params = count_total_params(model)
    model_state = copy.deepcopy(model.state_dict())

    # Baseline
    top1_base, _ = evaluate(val_loader, model, DEVICE)
    top1_base = top1_base.cpu().item() if isinstance(top1_base, torch.Tensor) else top1_base
    print(f"Baseline (unpruned): Top-1={top1_base:.2f}%\n")
    del model
    torch.cuda.empty_cache()

    # Grid of SV thresholds
    # The original paper uses: 750 * 0.000000015 * rho * N * M
    # For a 768x768 layer: 750 * 1.5e-8 * 0.06 * 768 * 768 ≈ 0.40
    # For sqrt: 750 * 1.5e-8 * 0.06 * 768 ≈ 0.00052
    # Let's sweep from very small to large
    sv_thresholds = [0] + [round(x, 6) for x in [
        0.0001, 0.0005, 0.001, 0.002, 0.005,
        0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0
    ]]

    print(f"{'SV thresh':>10}  {'Top-1%':>7}  {'Kept%':>7}  {'Time':>6}")
    print("-" * 40)

    results = []
    for thresh in sv_thresholds:
        t0 = time.time()
        top1, pct_kept = run_single(model_state, cached_metrics, val_loader, total_params, thresh)
        dt = time.time() - t0
        label = "no SV" if thresh == 0 else f"{thresh}"
        print(f"{label:>10}  {top1:7.2f}  {pct_kept:7.1f}  {dt:5.0f}s")
        results.append({'sv_threshold': thresh, 'top1': top1, 'pct_kept': pct_kept})

    # Save
    with open(os.path.join(os.path.dirname(__file__), 'sv_threshold_grid_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to sv_threshold_grid_results.json")

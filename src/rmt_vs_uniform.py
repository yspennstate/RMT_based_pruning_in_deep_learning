"""
Head-to-head: Full RMT method (b=1.5, SV pruning, N*M) vs simple uniform (b=0, no SV).
Both use iterative 19-cycle pruning with cached metrics from unpruned model.
"""

import sys, os, json, math, time, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import timm

from pruning import count_nonzero_params, count_total_params, replace_layers, prune_model, compute_layer_metrics_once
from SplittableLayers import SplittableConv, SplittableLinear
from validation import evaluate, get_val_dataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ALPHA, BETA, GOF = 0.25, 0.8, 1
RHO = 0.06
N_CYCLES = 19


def run_test(label, hp_b, enable_sv, model_state, cached_metrics, val_loader, total_params):
    """Run 19-cycle iterative pruning."""

    print(f"\n{'='*60}")
    print(f"{label}: b={hp_b}, SV={'ON (N*M)' if enable_sv else 'OFF'}, rho={RHO}, cycles={N_CYCLES}")
    print(f"{'='*60}")

    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    replace_layers(model, ALPHA, BETA, GOF, depth=0)
    model.load_state_dict(model_state)
    model.to(DEVICE)
    model.eval()

    # Baseline
    top1_base, _ = evaluate(val_loader, model, DEVICE)
    top1_base = top1_base.cpu().item() if isinstance(top1_base, torch.Tensor) else top1_base
    print(f"Baseline: Top-1={top1_base:.2f}%")

    results = [{'cycle': 0, 'top1': top1_base, 'pct_kept': 100.0}]

    for cycle in range(1, N_CYCLES + 1):
        t0 = time.time()

        # SV pruning on even cycles (full paper method)
        if enable_sv and cycle % 2 == 0:
            for name, layer in model.named_modules():
                if not isinstance(layer, (SplittableConv, SplittableLinear)):
                    continue

                # Save zero mask BEFORE SVD reconstruction
                zero_masks = {}
                for sub_name, sub in layer.named_modules():
                    if isinstance(sub, (torch.nn.Conv2d, torch.nn.Linear)):
                        zero_masks[sub_name] = (sub.weight.data == 0)

                W = layer.get_matrix()
                M_dim, N_dim = W.shape
                # Original paper threshold: scale * 750 * 1.5e-8 * rho * N * M
                scale = 1 + cycle * (1 - 1) / N_CYCLES  # scale=1 always
                sv_threshold = scale * 750 * 0.000000015 * RHO * N_dim * M_dim
                result, splus, LinfError, pct = layer.split(1, sv_threshold)

                # Re-apply zero mask so SVD doesn't resurrect pruned weights
                with torch.no_grad():
                    for sub_name, sub in layer.named_modules():
                        if isinstance(sub, (torch.nn.Conv2d, torch.nn.Linear)):
                            if sub_name in zero_masks:
                                sub.weight.data[zero_masks[sub_name]] = 0.0

            # Move back to GPU after split() (uses numpy/CPU)
            model.to(DEVICE)

        # Coefficient pruning with cached metrics
        prune_model(model, RHO, cycle, N_CYCLES, DEVICE,
                    hp_b=hp_b, enable_sv_pruning=False, cached_metrics=cached_metrics)

        num_nonzero = count_nonzero_params(model)
        pct_kept = 100 * num_nonzero / total_params
        top1, _ = evaluate(val_loader, model, DEVICE)
        top1 = top1.cpu().item() if isinstance(top1, torch.Tensor) else top1
        dt = time.time() - t0

        results.append({'cycle': cycle, 'top1': top1, 'pct_kept': pct_kept})
        print(f"  Cycle {cycle}/{N_CYCLES}: Top1={top1:.2f}% Kept={pct_kept:.1f}% ({dt:.0f}s)")

    del model
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"RMT (b=1.5 + SV pruning) vs Uniform (b=0, no SV)")

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

    # Test 1: Full RMT method (b=1.5, SV pruning with N*M)
    results_rmt = run_test("FULL RMT", hp_b=1.5, enable_sv=True,
                           model_state=model_state, cached_metrics=cached_metrics,
                           val_loader=val_loader, total_params=total_params)

    # Test 2: Uniform baseline (b=0, no SV)
    results_uniform = run_test("UNIFORM", hp_b=0.001, enable_sv=False,
                               model_state=model_state, cached_metrics=cached_metrics,
                               val_loader=val_loader, total_params=total_params)
    # Note: b=0 would cause 0^0=1 issues, use b=0.001 as effectively zero

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: RMT (b=1.5 + SV) vs Uniform (b~0, no SV)")
    print(f"{'='*60}")
    print(f"{'Cycle':>5}  {'RMT Top1':>9}  {'RMT Kept':>9}  {'Unif Top1':>10}  {'Unif Kept':>10}  {'Diff':>6}")
    print("-" * 60)
    for r1, r2 in zip(results_rmt, results_uniform):
        diff = r1['top1'] - r2['top1']
        print(f"{r1['cycle']:5d}  {r1['top1']:9.2f}  {r1['pct_kept']:9.1f}  {r2['top1']:10.2f}  {r2['pct_kept']:10.1f}  {diff:+6.2f}")

    # Save
    output = {
        'rmt_b15_sv': results_rmt,
        'uniform_b0_nosv': results_uniform,
    }
    with open(os.path.join(os.path.dirname(__file__), 'rmt_vs_uniform_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved to rmt_vs_uniform_results.json")

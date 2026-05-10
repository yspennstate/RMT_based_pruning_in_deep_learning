"""
Test: does singular vector pruning help?
Compare accuracy with and without SV pruning, using original paper parameters.
Uses CACHED metrics from unpruned model (matching the paper's intent).
SV pruning threshold uses sqrt(N*M) instead of N*M.
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

# Original paper parameters
HP_B = 1.5
RHO = 0.06
N_CYCLES = 19  # match the paper


def run_pruning_test(enable_sv_pruning, use_sqrt_nm, n_cycles=N_CYCLES):
    """Run n_cycles of actual iterative pruning with cached metrics."""

    # Reset the global cache so each run starts fresh
    import pruning
    pruning._layer_metrics_cache = None

    print(f"\n{'='*60}")
    print(f"SV pruning: {'ON' if enable_sv_pruning else 'OFF'}, sqrt(NM): {'YES' if use_sqrt_nm else 'NO (N*M)'}")
    print(f"b={HP_B}, rho={RHO}, cycles={n_cycles}")
    print(f"{'='*60}")

    # Load fresh model
    model = timm.create_model("vit_base_patch16_224", pretrained=True).to(DEVICE)
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)
    val_loader = get_val_dataset(preprocess=preprocess)

    replace_layers(model, ALPHA, BETA, GOF, depth=0)

    # Compute and cache metrics on the UNPRUNED model
    cached_metrics = compute_layer_metrics_once(model)

    model.to(DEVICE)
    total_params = count_total_params(model)

    # Baseline accuracy
    top1_base, top5_base = evaluate(val_loader, model, DEVICE)
    top1_base = top1_base.cpu().item() if isinstance(top1_base, torch.Tensor) else top1_base
    print(f"Baseline: Top-1={top1_base:.2f}%, Total params={total_params}")

    results = [{
        'cycle': 0,
        'top1': top1_base,
        'pct_kept': 100.0,
        'nonzero': total_params,
    }]

    # Save the zero-mask before SV pruning to preserve previously pruned entries.
    for cycle in range(1, n_cycles + 1):
        t0 = time.time()

        # SV pruning on even cycles — done BEFORE coefficient pruning
        if enable_sv_pruning and cycle % 2 == 0:
            for name, layer in model.named_modules():
                if not isinstance(layer, (SplittableConv, SplittableLinear)):
                    continue
                # Save the zero mask BEFORE SVD reconstruction
                zero_masks = {}
                for sub_name, sub in layer.named_modules():
                    if isinstance(sub, (torch.nn.Conv2d, torch.nn.Linear)):
                        zero_masks[sub_name] = (sub.weight.data == 0)

                W = layer.get_matrix()
                M_dim, N_dim = W.shape
                if use_sqrt_nm:
                    sv_threshold = 750 * 0.000000015 * RHO * math.sqrt(N_dim * M_dim)
                else:
                    sv_threshold = 750 * 0.000000015 * RHO * N_dim * M_dim
                # Run split with the threshold (does SV pruning via SVD reconstruction)
                result, splus, LinfError, pct = layer.split(1, sv_threshold)

                # Re-apply the zero mask so SVD reconstruction doesn't undo coefficient pruning
                with torch.no_grad():
                    for sub_name, sub in layer.named_modules():
                        if isinstance(sub, (torch.nn.Conv2d, torch.nn.Linear)):
                            if sub_name in zero_masks:
                                sub.weight.data[zero_masks[sub_name]] = 0.0

        # Coefficient pruning with CACHED metrics from unpruned model
        prune_model(model, RHO, cycle, n_cycles, DEVICE,
                    hp_b=HP_B, enable_sv_pruning=False, cached_metrics=cached_metrics)

        num_nonzero = count_nonzero_params(model)
        pct_kept = 100 * num_nonzero / total_params

        top1, top5 = evaluate(val_loader, model, DEVICE)
        top1 = top1.cpu().item() if isinstance(top1, torch.Tensor) else top1
        dt = time.time() - t0

        results.append({
            'cycle': cycle,
            'top1': top1,
            'pct_kept': pct_kept,
            'nonzero': num_nonzero,
        })

        print(f"  Cycle {cycle}/{n_cycles}: Top1={top1:.2f}% Kept={pct_kept:.1f}% ({dt:.0f}s)")

    del model
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Testing singular vector pruning effect")
    print(f"Parameters: b={HP_B}, rho={RHO}, cycles={N_CYCLES}")

    # Test 1: No SV pruning (coefficient pruning only) — load if already done
    results_path = os.path.join(os.path.dirname(__file__), 'sv_pruning_test_results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            prev = json.load(f)
        if 'no_sv_pruning' in prev and len(prev['no_sv_pruning']) > 1:
            print("Loaded previous no-SV results, skipping to SV test...")
            results_no_sv = prev['no_sv_pruning']
        else:
            results_no_sv = run_pruning_test(enable_sv_pruning=False, use_sqrt_nm=False)
    else:
        results_no_sv = run_pruning_test(enable_sv_pruning=False, use_sqrt_nm=False)

    # Test 2: With SV pruning, using sqrt(N*M) threshold
    results_sv_sqrt = run_pruning_test(enable_sv_pruning=True, use_sqrt_nm=True)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Cycle':>5}  {'No SV Top1':>10}  {'No SV Kept':>10}  {'SV+sqrt Top1':>12}  {'SV+sqrt Kept':>12}")
    print("-" * 60)
    for r1, r2 in zip(results_no_sv, results_sv_sqrt):
        print(f"{r1['cycle']:5d}  {r1['top1']:10.2f}  {r1['pct_kept']:10.1f}  {r2['top1']:12.2f}  {r2['pct_kept']:12.1f}")

    # Save results
    output = {
        'no_sv_pruning': results_no_sv,
        'sv_pruning_sqrt': results_sv_sqrt,
        'params': {'hp_b': HP_B, 'rho': RHO, 'n_cycles': N_CYCLES},
    }
    with open(os.path.join(os.path.dirname(__file__), 'sv_pruning_test_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to sv_pruning_test_results.json")

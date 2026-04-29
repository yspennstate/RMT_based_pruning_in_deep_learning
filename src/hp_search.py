"""
Hyperparameter search for RMT pruning using the ORIGINAL codebase.
Only 3 hyperparameters are tuned:
  - rho (target_reduction): prune rate per cycle
  - hp_a: linear coefficient in threshold (original: 4.0)
  - hp_b: exponent numerator in b/i schedule (original: 1.5)

SV pruning is disabled during search for speed.
Uses the original SplittableLayers, RMT.py, etc. unchanged.
"""

import sys
import os
import json
import math
import time
import copy

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.optim as optim
import timm
import optuna

from pruning import (
    count_nonzero_params,
    count_total_params,
    prune_model,
    replace_layers,
    compute_layer_metrics_once,
)
from SplittableLayers import SplittableConv, SplittableLinear
from training import fine_tune_model
from validation import evaluate, get_val_dataset

# ============================================================
# Config
# ============================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "hp_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Fixed params
ALPHA = 0.25
BETA = 0.8
GOF = 1
N_SEARCH_CYCLES = 10       # cycles per trial during search
LEARNING_RATE = 0.05e-6
L1_LAMBDA = 0.0000005
WEIGHT_DECAY = 0.0000002
FINE_TUNE_EPOCHS = 500      # reduced for search speed

# ============================================================
# Cached model + val loader + layer metrics
# ============================================================
_cached_model_state = None
_cached_val_loader = None
_cached_layer_metrics = None

def get_base_model_and_loader():
    global _cached_model_state, _cached_val_loader, _cached_layer_metrics

    if _cached_model_state is None:
        print("Loading ViT-Base pretrained model...")
        model = timm.create_model("vit_base_patch16_224", pretrained=True).to(DEVICE)

        # Use timm's own transform (matches the pretrained model's expected preprocessing)
        if _cached_val_loader is None:
            data_config = timm.data.resolve_model_data_config(model)
            preprocess = timm.data.create_transform(**data_config, is_training=False)
            _cached_val_loader = get_val_dataset(preprocess=preprocess)
            print(f"Loaded validation dataset (timm transform)")

        replace_layers(model, ALPHA, BETA, GOF, depth=0)

        # Compute RMT metrics once on the unpruned model
        # (layer.split() may move some weights to CPU via numpy)
        _cached_layer_metrics = compute_layer_metrics_once(model)

        # Ensure everything is back on device after SVD computations
        model.to(DEVICE)
        _cached_model_state = copy.deepcopy(model.state_dict())
        # Get baseline accuracy
        top1, top5 = evaluate(_cached_val_loader, model, DEVICE)
        top1 = top1.cpu().item() if isinstance(top1, torch.Tensor) else top1
        top5 = top5.cpu().item() if isinstance(top5, torch.Tensor) else top5
        print(f"Baseline: Top-1={top1:.2f}% Top-5={top5:.2f}%")
        get_base_model_and_loader.baseline_top1 = top1
        get_base_model_and_loader.baseline_top5 = top5
        del model
        torch.cuda.empty_cache()

    return _cached_model_state, _cached_val_loader, _cached_layer_metrics

# ============================================================
# Run one trial
# ============================================================
def run_trial(hp_a, hp_b, rho, n_cycles=N_SEARCH_CYCLES):
    """Run pruning with given hyperparams. Returns results dict.

    Single-step pruning: compute the cumulative survival fraction across all
    N cycles analytically, then prune each layer once to that target.
    Mathematically equivalent to the iterative version because each cycle
    removes the smallest remaining weights (order-independent).
    """

    model_state, val_loader, layer_metrics = get_base_model_and_loader()
    base_top1 = get_base_model_and_loader.baseline_top1

    # Fresh model from cached state
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    replace_layers(model, ALPHA, BETA, GOF, depth=0)
    model.load_state_dict(model_state)
    model.to(DEVICE)
    model.eval()

    total_params = count_total_params(model)

    results = {
        'hp_a': hp_a, 'hp_b': hp_b, 'rho': rho,
        'n_cycles': n_cycles, 'total_params': total_params,
        'base_top1': base_top1,
        'cycles': []
    }

    t0 = time.time()

    # --- Single-step pruning: compute cumulative survival per layer ---
    splittable_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (SplittableConv, SplittableLinear))
    ]

    for name, layer in splittable_layers:
        metrics = layer_metrics.get(name)
        if metrics is None:
            continue

        LinfError = metrics['LinfError']
        percentage_less_than_splus = metrics['percentage_less_than_splus']

        # Compute cumulative survival fraction across all cycles
        survival = 1.0
        for i in range(1, n_cycles + 1):
            target_reduction = rho
            randomness_score = (
                (1 - LinfError) ** (hp_b / i)
                * (percentage_less_than_splus / 100) ** (hp_b / i)
            )
            prune_frac = min(randomness_score * target_reduction, 1.0)
            survival *= (1.0 - prune_frac)

        # Number of weights to keep
        all_abs_weights = []
        for submodule in layer.modules():
            if isinstance(submodule, (torch.nn.Conv2d, torch.nn.Linear)):
                w = submodule.weight.data.abs()
                nonzero_vals = w[w > 0]
                if len(nonzero_vals) > 0:
                    all_abs_weights.append(nonzero_vals)

        if all_abs_weights:
            all_abs_weights = torch.cat(all_abs_weights)
            n_keep = max(int(len(all_abs_weights) * survival), 0)
            n_prune = len(all_abs_weights) - n_keep

            if n_prune > 0:
                sorted_vals, _ = torch.sort(all_abs_weights)
                threshold = sorted_vals[n_prune - 1].item()
                with torch.no_grad():
                    for submodule in layer.modules():
                        if isinstance(submodule, (torch.nn.Conv2d, torch.nn.Linear)):
                            mask = submodule.weight.data.abs() > threshold
                            submodule.weight.data *= mask.float()

    # Single evaluation
    num_nonzero = count_nonzero_params(model)
    pct_kept = 100 * num_nonzero / total_params
    top1, top5 = evaluate(val_loader, model, DEVICE)
    top1 = top1.cpu().item() if isinstance(top1, torch.Tensor) else top1
    top5 = top5.cpu().item() if isinstance(top5, torch.Tensor) else top5

    elapsed = time.time() - t0
    print(f"  Pruned in 1 step: Top-1={top1:.2f}% Kept={pct_kept:.1f}% [{elapsed:.0f}s]")

    results['cycles'].append({
        'cycle': n_cycles, 'top1': top1, 'top5': top5,
        'nonzero': num_nonzero, 'pct_kept': pct_kept,
        'elapsed': round(elapsed, 1)
    })

    final = results['cycles'][-1]
    results['final_top1'] = final['top1']
    results['final_pct_kept'] = final['pct_kept']
    results['accuracy_drop'] = base_top1 - final['top1']

    return results

# ============================================================
# Optuna objective
# ============================================================
_search_bounds = {'a': (1.0, 8.0), 'b': (0.5, 5.0), 'rho': (0.03, 0.20)}

def objective(trial):
    hp_a = trial.suggest_float('a', *_search_bounds['a'])
    hp_b = trial.suggest_float('b', *_search_bounds['b'])
    rho = trial.suggest_float('rho', *_search_bounds['rho'])

    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: a={hp_a:.3f} b={hp_b:.3f} rho={rho:.4f}")
    print(f"{'='*60}")

    results = run_trial(hp_a=hp_a, hp_b=hp_b, rho=rho)

    # Score: geometric mean of accuracy retained and pruning fraction
    base_top1 = results['base_top1']
    final_top1 = results['final_top1']
    pct_kept = results['final_pct_kept']

    accuracy_retained = final_top1 / base_top1 if base_top1 > 0 else 0
    pruning_fraction = 1.0 - pct_kept / 100.0

    if accuracy_retained <= 0 or pruning_fraction <= 0:
        score = 0.0
    else:
        score = math.sqrt(accuracy_retained * pruning_fraction)

    results['score'] = score
    print(f"\n>>> Trial {trial.number}: a={hp_a:.3f} b={hp_b:.3f} rho={rho:.4f}")
    print(f"    Top1={final_top1:.2f}% Kept={pct_kept:.1f}% Score={score:.4f}")

    # Save (use refine_ prefix if narrowed search)
    prefix = "refine" if _search_bounds['a'][0] > 1.0 else "trial"
    trial_path = os.path.join(RESULTS_DIR, f"{prefix}_{trial.number:03d}.json")
    with open(trial_path, 'w') as f:
        json.dump(results, f, indent=2)

    return score

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        # Full run with best params
        hp_a = float(sys.argv[2])
        hp_b = float(sys.argv[3])
        rho = float(sys.argv[4])
        n_cycles = int(sys.argv[5]) if len(sys.argv) > 5 else 19
        print(f"Full run: a={hp_a} b={hp_b} rho={rho} cycles={n_cycles}")
        results = run_trial(hp_a=hp_a, hp_b=hp_b, rho=rho, n_cycles=n_cycles)
        with open(os.path.join(RESULTS_DIR, "full_run.json"), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDone: Top1={results['final_top1']:.2f}% Kept={results['final_pct_kept']:.1f}%")
    elif len(sys.argv) > 1 and sys.argv[1] == '--refine':
        # Refined search over narrower interval
        n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        _search_bounds['a'] = (5.5, 8.0)
        _search_bounds['b'] = (2.0, 3.5)
        _search_bounds['rho'] = (0.06, 0.10)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=123),
            study_name='rmt_pruning_hp_refine'
        )

        print("REFINED Optuna HP search (narrowed ranges)")
        print(f"  a: {_search_bounds['a']}, b: {_search_bounds['b']}, rho: {_search_bounds['rho']}")
        print(f"  {n_trials} trials x {N_SEARCH_CYCLES} cycles")
        print(f"  Objective: sqrt(accuracy_retained * pruning_fraction)\n")

        get_base_model_and_loader()
        study.optimize(objective, n_trials=n_trials)

        print("\n" + "=" * 60)
        print("REFINED SEARCH COMPLETE")
        print("=" * 60)
        print(f"Best score: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        bp = study.best_params
        print(f"\nTo run full pruning:")
        print(f"  python hp_search.py --full {bp['a']:.4f} {bp['b']:.4f} {bp['rho']:.4f} 19")

        summary = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'all_trials': [
                {'number': t.number, 'params': t.params, 'value': t.value}
                for t in study.trials
            ]
        }
        with open(os.path.join(RESULTS_DIR, "optuna_refine_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

        # Also save individual trials with refine_ prefix
        for t in study.trials:
            trial_path = os.path.join(RESULTS_DIR, f"refine_{t.number:03d}.json")
            if not os.path.exists(trial_path):
                # Already saved during objective
                pass
    else:
        # Optuna search
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name='rmt_pruning_hp'
        )

        print("Optuna HP search for RMT pruning (original codebase)")
        print("Tuning: a (threshold coeff), b (exponent in b/i), rho (prune rate)")
        print("SV pruning: disabled during search")
        print(f"Search: 20 trials x {N_SEARCH_CYCLES} cycles")
        print(f"Objective: sqrt(accuracy_retained * pruning_fraction)\n")

        # Pre-load model and data
        get_base_model_and_loader()

        study.optimize(objective, n_trials=20)

        # Summary
        print("\n" + "=" * 60)
        print("SEARCH COMPLETE")
        print("=" * 60)
        print(f"Best score: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        bp = study.best_params
        print(f"\nTo run full pruning:")
        print(f"  python hp_search.py --full {bp['a']:.4f} {bp['b']:.4f} {bp['rho']:.4f} 19")

        summary = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'all_trials': [
                {'number': t.number, 'params': t.params, 'value': t.value}
                for t in study.trials
            ]
        }
        with open(os.path.join(RESULTS_DIR, "optuna_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

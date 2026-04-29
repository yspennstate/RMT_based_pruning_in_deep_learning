"""
Pruning hyperparameter grid search for RMT pruning hyperparameters.
Dense grid over the best region found by Optuna, plus a wider Optuna
exploration. Designed for long unattended runs (~10 hours on 1× A40).

Results saved incrementally to hp_results/grid_*.json so nothing is lost
if the process is interrupted.
"""

import sys
import os
import json
import math
import time
import itertools
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import timm
import optuna

from pruning import (
    count_nonzero_params,
    count_total_params,
    replace_layers,
    compute_layer_metrics_once,
)
from SplittableLayers import SplittableConv, SplittableLinear
from validation import evaluate, get_val_dataset

# ============================================================
# Config
# ============================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "hp_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA = 0.25
BETA = 0.8
GOF = 1
N_SEARCH_CYCLES = 10

MAX_RUNTIME_HOURS = 10.0
MAX_RUNTIME_SECS = MAX_RUNTIME_HOURS * 3600

# ============================================================
# Cached model + val loader + layer metrics
# ============================================================
_cached_model_state = None
_cached_val_loader = None
_cached_layer_metrics = None
_baseline_top1 = None

def setup():
    global _cached_model_state, _cached_val_loader, _cached_layer_metrics, _baseline_top1

    print("Loading ViT-Base pretrained model...")
    model = timm.create_model("vit_base_patch16_224", pretrained=True).to(DEVICE)

    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)
    _cached_val_loader = get_val_dataset(preprocess=preprocess)
    print(f"Loaded validation dataset")

    replace_layers(model, ALPHA, BETA, GOF, depth=0)
    _cached_layer_metrics = compute_layer_metrics_once(model)
    model.to(DEVICE)
    _cached_model_state = copy.deepcopy(model.state_dict())

    top1, top5 = evaluate(_cached_val_loader, model, DEVICE)
    _baseline_top1 = top1.cpu().item() if isinstance(top1, torch.Tensor) else top1
    print(f"Baseline: Top-1={_baseline_top1:.2f}%")
    del model
    torch.cuda.empty_cache()


def run_single(hp_a, hp_b, rho, n_cycles=N_SEARCH_CYCLES):
    """Run pruning with given hyperparams. Returns results dict."""
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    replace_layers(model, ALPHA, BETA, GOF, depth=0)
    model.load_state_dict(_cached_model_state)
    model.to(DEVICE)
    model.eval()

    total_params = count_total_params(model)

    # Collect splittable layers
    splittable_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (SplittableConv, SplittableLinear))
    ]

    for name, layer in splittable_layers:
        metrics = _cached_layer_metrics.get(name)
        if metrics is None:
            continue

        LinfError = metrics['LinfError']
        percentage_less_than_splus = metrics['percentage_less_than_splus']

        # Cumulative survival fraction across all cycles
        survival = 1.0
        for i in range(1, n_cycles + 1):
            target_reduction = rho
            randomness_score = (
                (1 - LinfError) ** (hp_b / i)
                * (percentage_less_than_splus / 100) ** (hp_b / i)
            )
            prune_frac = min(randomness_score * target_reduction, 1.0)
            survival *= (1.0 - prune_frac)

        # Prune by threshold
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

    num_nonzero = count_nonzero_params(model)
    pct_kept = 100 * num_nonzero / total_params
    top1, top5 = evaluate(_cached_val_loader, model, DEVICE)
    top1 = top1.cpu().item() if isinstance(top1, torch.Tensor) else top1
    top5 = top5.cpu().item() if isinstance(top5, torch.Tensor) else top5

    del model
    torch.cuda.empty_cache()

    accuracy_retained = top1 / _baseline_top1 if _baseline_top1 > 0 else 0
    pruning_fraction = 1.0 - pct_kept / 100.0
    if accuracy_retained <= 0 or pruning_fraction <= 0:
        score = 0.0
    else:
        score = math.sqrt(accuracy_retained * pruning_fraction)

    return {
        'hp_a': hp_a, 'hp_b': hp_b, 'rho': rho,
        'n_cycles': n_cycles, 'total_params': total_params,
        'base_top1': _baseline_top1,
        'final_top1': top1, 'final_top5': top5,
        'nonzero': num_nonzero, 'pct_kept': pct_kept,
        'accuracy_drop': _baseline_top1 - top1,
        'pruning_fraction': pruning_fraction,
        'score': score,
    }


def save_result(result, prefix, index):
    path = os.path.join(RESULTS_DIR, f"{prefix}_{index:04d}.json")
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    return path


# ============================================================
# Phase 1: Dense grid search over best region
# ============================================================
def phase1_grid_search(start_time):
    """Dense grid over the sweet spot found by Optuna."""
    print("\n" + "=" * 70)
    print("PHASE 1: Dense grid search")
    print("=" * 70)

    # Grid ranges based on top results:
    #   a: best at 5.5-7.5  -> grid 5.0 to 8.0, step 0.25 (13 pts)
    #   b: best at 2.0-3.1  -> grid 1.8 to 3.4, step 0.15 (11 pts)
    #   rho: best at 0.07-0.09 -> grid 0.06 to 0.10, step 0.005 (9 pts)
    a_vals = [round(5.0 + i * 0.25, 2) for i in range(13)]   # 5.0 to 8.0
    b_vals = [round(1.8 + i * 0.15, 2) for i in range(11)]   # 1.8 to 3.3
    rho_vals = [round(0.06 + i * 0.005, 4) for i in range(9)] # 0.06 to 0.10

    grid = list(itertools.product(a_vals, b_vals, rho_vals))
    total = len(grid)
    print(f"  a: {a_vals[0]} to {a_vals[-1]} ({len(a_vals)} pts)")
    print(f"  b: {b_vals[0]} to {b_vals[-1]} ({len(b_vals)} pts)")
    print(f"  rho: {rho_vals[0]} to {rho_vals[-1]} ({len(rho_vals)} pts)")
    print(f"  Total grid points: {total}")
    print(f"  Est. time: {total * 90 / 3600:.1f} hours\n")

    best_score = 0
    best_result = None
    results_all = []

    for idx, (a, b, rho) in enumerate(grid):
        elapsed = time.time() - start_time
        remaining = MAX_RUNTIME_SECS - elapsed
        if remaining < 120:  # stop with 2 min buffer
            print(f"\n  Time limit approaching ({elapsed/3600:.1f}h elapsed). Stopping grid.")
            break

        t0 = time.time()
        result = run_single(hp_a=a, hp_b=b, rho=rho)
        dt = time.time() - t0

        save_result(result, "grid", idx)
        results_all.append(result)

        marker = ""
        if result['score'] > best_score:
            best_score = result['score']
            best_result = result
            marker = " *** NEW BEST ***"

        print(f"  [{idx+1}/{total}] a={a:.2f} b={b:.2f} rho={rho:.4f} "
              f"-> Top1={result['final_top1']:.2f}% Kept={result['pct_kept']:.1f}% "
              f"Score={result['score']:.4f} ({dt:.0f}s){marker}")

    # Save grid summary
    if best_result:
        summary = {
            'phase': 'grid',
            'completed': len(results_all),
            'total_planned': total,
            'best_score': best_score,
            'best_params': {
                'a': best_result['hp_a'],
                'b': best_result['hp_b'],
                'rho': best_result['rho'],
            },
            'best_top1': best_result['final_top1'],
            'best_pct_kept': best_result['pct_kept'],
        }
        with open(os.path.join(RESULTS_DIR, "grid_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Grid best: a={best_result['hp_a']:.3f} b={best_result['hp_b']:.3f} "
              f"rho={best_result['rho']:.4f} Score={best_score:.4f} "
              f"Top1={best_result['final_top1']:.2f}% Kept={best_result['pct_kept']:.1f}%")

    return results_all, best_result


# ============================================================
# Phase 2: Optuna refinement around grid best
# ============================================================
def phase2_optuna_refine(start_time, grid_best):
    """Use remaining time for Optuna TPE around the grid best."""
    elapsed = time.time() - start_time
    remaining = MAX_RUNTIME_SECS - elapsed
    if remaining < 300:
        print("\nNo time left for Phase 2.")
        return

    est_trials = int(remaining / 90)  # ~90s per trial
    print(f"\n{'=' * 70}")
    print(f"PHASE 2: Optuna refinement ({est_trials} trials, {remaining/3600:.1f}h remaining)")
    print("=" * 70)

    # Narrow ranges around grid best
    if grid_best:
        a_center = grid_best['hp_a']
        b_center = grid_best['hp_b']
        rho_center = grid_best['rho']
    else:
        a_center, b_center, rho_center = 6.6, 2.1, 0.078

    a_lo = max(a_center - 0.5, 4.5)
    a_hi = min(a_center + 0.5, 8.5)
    b_lo = max(b_center - 0.3, 1.5)
    b_hi = min(b_center + 0.3, 4.0)
    rho_lo = max(rho_center - 0.01, 0.04)
    rho_hi = min(rho_center + 0.01, 0.12)

    print(f"  a: [{a_lo:.2f}, {a_hi:.2f}]")
    print(f"  b: [{b_lo:.2f}, {b_hi:.2f}]")
    print(f"  rho: [{rho_lo:.4f}, {rho_hi:.4f}]\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=999),
        study_name='long unattended_refine'
    )

    best_score = 0
    trial_idx = 0

    def objective(trial):
        nonlocal best_score, trial_idx

        elapsed = time.time() - start_time
        if elapsed > MAX_RUNTIME_SECS - 120:
            raise optuna.exceptions.OptunaError("Time limit reached")

        hp_a = trial.suggest_float('a', a_lo, a_hi)
        hp_b = trial.suggest_float('b', b_lo, b_hi)
        rho = trial.suggest_float('rho', rho_lo, rho_hi)

        t0 = time.time()
        result = run_single(hp_a=hp_a, hp_b=hp_b, rho=rho)
        dt = time.time() - t0

        save_result(result, "optuna2", trial_idx)
        trial_idx += 1

        marker = ""
        if result['score'] > best_score:
            best_score = result['score']
            marker = " *** NEW BEST ***"

        print(f"  [Trial {trial.number}] a={hp_a:.3f} b={hp_b:.3f} rho={rho:.4f} "
              f"-> Top1={result['final_top1']:.2f}% Kept={result['pct_kept']:.1f}% "
              f"Score={result['score']:.4f} ({dt:.0f}s){marker}")

        return result['score']

    try:
        study.optimize(objective, n_trials=est_trials, timeout=remaining - 60)
    except Exception as e:
        print(f"\n  Phase 2 stopped: {e}")

    # Save Optuna phase 2 summary
    if len(study.trials) > 0:
        summary = {
            'phase': 'optuna2',
            'completed': len(study.trials),
            'best_score': study.best_value,
            'best_params': study.best_params,
        }
        with open(os.path.join(RESULTS_DIR, "optuna2_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Optuna2 best: {study.best_params} Score={study.best_value:.4f}")


# ============================================================
# Phase 3: Final summary
# ============================================================
def final_summary():
    """Read ALL results and produce a master summary."""
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print("=" * 70)

    all_results = []
    for pattern in ["trial_*.json", "refine_*.json", "grid_*.json", "optuna2_*.json"]:
        import glob
        for f in glob.glob(os.path.join(RESULTS_DIR, pattern)):
            try:
                d = json.load(open(f))
                d['source'] = os.path.basename(f)
                all_results.append(d)
            except:
                pass

    if not all_results:
        print("  No results found!")
        return

    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)

    print(f"\n  Total trials across all phases: {len(all_results)}")
    print(f"\n  Top 20 results:")
    print(f"  {'Rank':>4} {'a':>6} {'b':>6} {'rho':>8} {'Top1%':>7} {'Kept%':>7} {'Score':>7} {'Source'}")
    print(f"  {'-'*65}")

    for i, d in enumerate(all_results[:20]):
        print(f"  {i+1:>4} {d['hp_a']:>6.3f} {d['hp_b']:>6.3f} {d['rho']:>8.4f} "
              f"{d['final_top1']:>7.2f} {d['pct_kept']:>7.1f} {d['score']:>7.4f} {d.get('source','')}")

    # Save master summary
    master = {
        'total_trials': len(all_results),
        'best': {
            'params': {'a': all_results[0]['hp_a'], 'b': all_results[0]['hp_b'], 'rho': all_results[0]['rho']},
            'score': all_results[0]['score'],
            'top1': all_results[0]['final_top1'],
            'pct_kept': all_results[0]['pct_kept'],
            'source': all_results[0].get('source', ''),
        },
        'top20': [
            {
                'a': d['hp_a'], 'b': d['hp_b'], 'rho': d['rho'],
                'score': d['score'], 'top1': d['final_top1'],
                'pct_kept': d['pct_kept'], 'source': d.get('source', ''),
            }
            for d in all_results[:20]
        ]
    }
    with open(os.path.join(RESULTS_DIR, "long unattended_master_summary.json"), 'w') as f:
        json.dump(master, f, indent=2)

    best = all_results[0]
    print(f"\n  BEST OVERALL: a={best['hp_a']:.4f} b={best['hp_b']:.4f} rho={best['rho']:.5f}")
    print(f"  Top-1={best['final_top1']:.2f}% Kept={best['pct_kept']:.1f}% Score={best['score']:.4f}")
    print(f"\n  To run full 19-cycle pruning with best params:")
    print(f"  python hp_search.py --full {best['hp_a']:.4f} {best['hp_b']:.4f} {best['rho']:.5f} 19")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    start_time = time.time()
    print(f"Pruning hyperparameter grid search starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Will run for up to {MAX_RUNTIME_HOURS} hours")
    print(f"Device: {DEVICE}")

    setup()

    grid_results, grid_best = phase1_grid_search(start_time)
    phase2_optuna_refine(start_time, grid_best)
    final_summary()

    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time/3600:.2f} hours")
    print(f"Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")

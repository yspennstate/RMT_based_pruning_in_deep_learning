"""
Hyperparameter search for RMT pruning — matches paper equation (5) exactly.

Paper formula:
    zeta_l(t) = ((1 - mu_l) * gamma_l)^(b/t) * rho * num_nonzero_l

Two hyperparameters: b, rho.
Pruning: remove the zeta_l(t) smallest nonzero weights per layer.
SV pruning disabled during search for speed.
"""

import sys, os, json, math, time, copy, itertools, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import timm

from pruning import count_nonzero_params, count_total_params, replace_layers, compute_layer_metrics_once
from SplittableLayers import SplittableConv, SplittableLinear
from validation import evaluate, get_val_dataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "hp_results_v2")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA, BETA, GOF = 0.25, 0.8, 1
N_SEARCH_CYCLES = 10

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
    print("Loaded validation dataset")
    replace_layers(model, ALPHA, BETA, GOF, depth=0)
    _cached_layer_metrics = compute_layer_metrics_once(model)
    model.to(DEVICE)
    _cached_model_state = copy.deepcopy(model.state_dict())
    top1, _ = evaluate(_cached_val_loader, model, DEVICE)
    _baseline_top1 = top1.cpu().item() if isinstance(top1, torch.Tensor) else top1
    print(f"Baseline: Top-1={_baseline_top1:.2f}%")
    del model
    torch.cuda.empty_cache()


def run_single(hp_b, rho, n_cycles=N_SEARCH_CYCLES):
    """
    Single-step surrogate matching paper eq (5):
        zeta_l(t) = ((1-mu_l)*gamma_l)^(b/t) * rho * num_nonzero
    Cumulative survival across cycles, then prune smallest weights.
    """
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    replace_layers(model, ALPHA, BETA, GOF, depth=0)
    model.load_state_dict(_cached_model_state)
    model.to(DEVICE)
    model.eval()

    total_params = count_total_params(model)

    for name, layer in model.named_modules():
        if not isinstance(layer, (SplittableConv, SplittableLinear)):
            continue
        metrics = _cached_layer_metrics.get(name)
        if metrics is None:
            continue

        mu_l = metrics['LinfError']
        gamma_l = metrics['percentage_less_than_splus'] / 100.0

        # Cumulative survival = product of (1 - prune_frac) over cycles
        survival = 1.0
        for t in range(1, n_cycles + 1):
            prune_frac = ((1 - mu_l) * gamma_l) ** (hp_b / t) * rho
            prune_frac = min(prune_frac, 1.0)
            survival *= (1.0 - prune_frac)

        # Remove the smallest (1-survival) fraction of nonzero weights
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
    top1, top5 = evaluate(_cached_val_loader, model, DEVICE)
    top1 = top1.cpu().item() if isinstance(top1, torch.Tensor) else top1
    top5 = top5.cpu().item() if isinstance(top5, torch.Tensor) else top5
    del model
    torch.cuda.empty_cache()

    acc_ret = top1 / _baseline_top1 if _baseline_top1 > 0 else 0
    prune_frac = 1.0 - pct_kept / 100.0
    score = math.sqrt(acc_ret * prune_frac) if acc_ret > 0 and prune_frac > 0 else 0.0

    return {
        'hp_b': hp_b, 'rho': rho, 'n_cycles': n_cycles,
        'total_params': total_params, 'base_top1': _baseline_top1,
        'final_top1': top1, 'final_top5': top5,
        'nonzero': num_nonzero, 'pct_kept': pct_kept,
        'accuracy_drop': _baseline_top1 - top1,
        'score': score,
    }


def grid_search(max_hours=4.0):
    start_time = time.time()
    max_secs = max_hours * 3600

    b_vals = [round(0.5 + i * 0.25, 2) for i in range(19)]    # 0.5 to 5.0
    rho_vals = [round(0.03 + i * 0.01, 2) for i in range(18)] # 0.03 to 0.20

    grid = list(itertools.product(b_vals, rho_vals))
    total = len(grid)
    print(f"Grid: {len(b_vals)} x {len(rho_vals)} = {total} points")
    print(f"  b:   {b_vals[0]} to {b_vals[-1]}")
    print(f"  rho: {rho_vals[0]} to {rho_vals[-1]}")
    print(f"  Max: {max_hours}h\n")

    best_score = 0
    best_result = None
    completed = 0

    # Load any existing results to find best so far and skip completed
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "grid_*.json"))):
        try:
            d = json.load(open(f))
            if d['score'] > best_score:
                best_score = d['score']
                best_result = d
        except Exception:
            pass

    for idx, (b, rho) in enumerate(grid):
        if time.time() - start_time > max_secs - 120:
            print(f"\nTime limit. Stopping at {idx}/{total}.")
            break

        # Skip already-completed grid points
        path = os.path.join(RESULTS_DIR, f"grid_{idx:04d}.json")
        if os.path.exists(path):
            completed += 1
            continue

        t0 = time.time()
        result = run_single(hp_b=b, rho=rho)
        dt = time.time() - t0

        path = os.path.join(RESULTS_DIR, f"grid_{idx:04d}.json")
        with open(path, 'w') as f:
            json.dump(result, f, indent=2)

        marker = ""
        if result['score'] > best_score:
            best_score = result['score']
            best_result = result
            marker = " *** BEST ***"

        print(f"  [{idx+1}/{total}] b={b:.2f} rho={rho:.2f} "
              f"-> Top1={result['final_top1']:.2f}% Kept={result['pct_kept']:.1f}% "
              f"Score={result['score']:.4f} ({dt:.0f}s){marker}")
        completed = idx + 1

    if best_result:
        summary = {
            'completed': completed, 'total_planned': total,
            'best_score': best_score,
            'best_params': {'b': best_result['hp_b'], 'rho': best_result['rho']},
            'best_top1': best_result['final_top1'],
            'best_pct_kept': best_result['pct_kept'],
        }
        with open(os.path.join(RESULTS_DIR, "grid_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n{'='*60}")
        print(f"BEST: b={best_result['hp_b']:.2f} rho={best_result['rho']:.2f}")
        print(f"  Top1={best_result['final_top1']:.2f}% Kept={best_result['pct_kept']:.1f}% Score={best_score:.4f}")
        print(f"{'='*60}")

    print(f"\nRuntime: {(time.time()-start_time)/3600:.2f}h")


if __name__ == "__main__":
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    print(f"HP Search v2 — two parameters (b, rho)")
    print(f"Formula: zeta_l(t) = ((1-mu_l)*gamma_l)^(b/t) * rho * num_nonzero")
    print(f"Device: {DEVICE}\n")
    setup()
    grid_search(max_hours=hours)

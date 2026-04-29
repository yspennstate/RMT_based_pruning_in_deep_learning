"""
Refined z+bulk Haar search — runs AFTER haar_optuna.py finishes.
Loads best z+bulk params from first run and does a tight search around them.
"""

import sys, os, json, math, time, copy, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import timm
import numpy as np
import optuna

from pruning import count_nonzero_params, count_total_params, replace_layers, compute_layer_metrics_once
from SplittableLayers import SplittableConv, SplittableLinear
from validation import evaluate, get_val_dataset
from RMT import bema_inside

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ALPHA, BETA, GOF = 0.25, 0.8, 1
HP_B = 1.5
RHO = 0.06
TARGET_CYCLE = 8
N_TRIALS = 60

LOG_FILE = os.path.join(os.path.dirname(__file__), "haar_refined_log.txt")
PREV_RESULTS = os.path.join(os.path.dirname(__file__), "haar_optuna_results.json")


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


def sv_prune_z_bulk(W_np, splus, z_thresh, bulk_cutoff):
    U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
    M, N = W_np.shape
    std_U = 1.0 / math.sqrt(M)
    std_V = 1.0 / math.sqrt(N)
    thresh_U = z_thresh * std_U
    thresh_V = z_thresh * std_V

    for i in range(len(S)):
        if S[i] / splus >= bulk_cutoff:
            continue
        U[:, i] = np.where(np.abs(U[:, i]) < thresh_U, 0, U[:, i])
        Vt[i, :] = np.where(np.abs(Vt[i, :]) < thresh_V, 0, Vt[i, :])

    return U @ np.diag(S) @ Vt


def magnitude_prune(model, cached_metrics):
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


def run_test(model_state, cached_metrics, val_loader, total_params, z_thresh, bulk_cutoff):
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    replace_layers(model, ALPHA, BETA, GOF, depth=0)
    model.load_state_dict(model_state)
    model.eval()

    t0 = time.time()

    for name, layer in model.named_modules():
        if not isinstance(layer, (SplittableConv, SplittableLinear)):
            continue
        W = layer.get_matrix()
        splus = compute_splus(W)
        W_pruned = sv_prune_z_bulk(W, splus, z_thresh, bulk_cutoff)
        layer.set_params("layer1", torch.from_numpy(W_pruned).float(),
                        bias=None, change_bias=False)

    model.to(DEVICE)
    magnitude_prune(model, cached_metrics)

    num_nonzero = count_nonzero_params(model)
    pct_kept = 100 * num_nonzero / total_params
    top1, _ = evaluate(val_loader, model, DEVICE)
    top1 = top1.cpu().item() if isinstance(top1, torch.Tensor) else top1
    dt = time.time() - t0

    del model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    return top1, pct_kept, dt


if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # Load previous results to find best z+bulk region
    with open(PREV_RESULTS, 'r') as f:
        prev = json.load(f)

    baseline_top1 = prev['baseline_top1']

    # Find best z_bulk trials
    zbulk_trials = [t for t in prev['trials']
                    if t['params'].get('method') == 'z_bulk'
                    and t['top1'] is not None and t['top1'] > 0]
    zbulk_trials.sort(key=lambda t: t['top1'], reverse=True)

    print(f"Previous run: {len(prev['trials'])} trials, baseline={baseline_top1:.2f}%")
    print(f"Top 5 z+bulk from previous run:")
    for t in zbulk_trials[:5]:
        p = t['params']
        print(f"  z={p['zb_z']:.4f}  cut={p['zb_cutoff']:.2f}  -> {t['top1']:.2f}%")

    # Determine refined search range from top results
    top_z = [t['params']['zb_z'] for t in zbulk_trials[:10] if t['top1'] > baseline_top1 - 0.2]
    top_cut = [t['params']['zb_cutoff'] for t in zbulk_trials[:10] if t['top1'] > baseline_top1 - 0.2]

    if top_z:
        z_min = max(0.001, min(top_z) * 0.3)
        z_max = max(top_z) * 3.0
        cut_min = max(0.3, min(top_cut) - 0.15)
        cut_max = min(1.0, max(top_cut) + 0.1)
    else:
        # Fallback if nothing beat baseline
        z_min, z_max = 0.005, 0.5
        cut_min, cut_max = 0.5, 1.0

    print(f"\nRefined search range:")
    print(f"  z:      [{z_min:.4f}, {z_max:.4f}]")
    print(f"  cutoff: [{cut_min:.2f}, {cut_max:.2f}]")
    print(f"  trials: {N_TRIALS}\n")

    with open(LOG_FILE, "w") as f:
        f.write(f"Refined z+bulk search started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Baseline: {baseline_top1:.2f}%\n")
        f.write(f"z range: [{z_min:.4f}, {z_max:.4f}], cutoff range: [{cut_min:.2f}, {cut_max:.2f}]\n\n")

    # Setup model
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)
    val_loader = get_val_dataset(preprocess=preprocess)

    replace_layers(model, ALPHA, BETA, GOF, depth=0)
    import pruning
    pruning._layer_metrics_cache = None
    cached_metrics = compute_layer_metrics_once(model)

    total_params = count_total_params(model)
    model_state = copy.deepcopy(model.cpu().state_dict())
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Optuna — refined search
    sampler = optuna.samplers.TPESampler(seed=123, n_startup_trials=10)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="haar_zbulk_refined",
    )

    # Seed with the top results from previous run
    for t in zbulk_trials[:8]:
        p = t['params']
        study.enqueue_trial({"z": p['zb_z'], "cutoff": p['zb_cutoff']})

    def objective(trial):
        z = trial.suggest_float("z", z_min, z_max, log=True)
        cutoff = trial.suggest_float("cutoff", cut_min, cut_max)
        label = f"z={z:.4f} cut={cutoff:.3f}"

        try:
            top1, pct_kept, dt = run_test(
                model_state, cached_metrics, val_loader, total_params, z, cutoff)
            diff = top1 - baseline_top1
            msg = f"Trial {trial.number:3d}: {label:<30}  {top1:7.2f}%  kept={pct_kept:.1f}%  diff={diff:+.2f}pp  {dt:.0f}s"
            print(msg)
            with open(LOG_FILE, "a") as f:
                f.write(msg + "\n")
            return top1
        except Exception as e:
            msg = f"Trial {trial.number:3d}: {label:<30}  FAILED: {e}"
            print(msg)
            with open(LOG_FILE, "a") as f:
                f.write(msg + "\n")
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(5)
            return float('-inf')

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    # Results
    print(f"\n{'='*70}")
    print(f"REFINED RESULTS - {len(study.trials)} trials (z+bulk only)")
    print(f"{'='*70}")
    print(f"Baseline: {baseline_top1:.2f}%")
    print(f"Best: #{study.best_trial.number}  {study.best_value:.2f}% ({study.best_value - baseline_top1:+.2f}pp)")
    print(f"  z={study.best_params['z']:.4f}  cutoff={study.best_params['cutoff']:.3f}")

    print(f"\nTOP 15:")
    print(f"{'#':>4}  {'z':>8}  {'cutoff':>7}  {'Top-1':>7}  {'diff':>7}")
    print("-" * 42)
    sorted_trials = sorted(study.trials,
                          key=lambda t: t.value if t.value is not None else -999,
                          reverse=True)
    for t in sorted_trials[:15]:
        if t.value is None or t.value == float('-inf'):
            continue
        diff = t.value - baseline_top1
        print(f"{t.number:4d}  {t.params['z']:8.4f}  {t.params['cutoff']:7.3f}  {t.value:7.2f}  {diff:+7.2f}")

    # Save
    all_results = {
        'baseline_top1': baseline_top1,
        'best_params': study.best_params,
        'best_top1': study.best_value,
        'search_range': {'z_min': z_min, 'z_max': z_max, 'cut_min': cut_min, 'cut_max': cut_max},
        'trials': [
            {
                'number': t.number,
                'params': t.params,
                'top1': t.value if t.value is not None else None,
            }
            for t in study.trials
        ]
    }
    results_file = os.path.join(os.path.dirname(__file__), 'haar_refined_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {results_file}")

    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"Best: z={study.best_params['z']:.4f} cutoff={study.best_params['cutoff']:.3f} -> {study.best_value:.2f}%\n")
        f.write(f"Finished {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

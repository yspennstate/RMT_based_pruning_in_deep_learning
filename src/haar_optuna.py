"""
Haar SV pruning — 3 methods, Optuna Bayesian optimization.

Method 1: "Just z" — apply z threshold to ALL singular vectors (spikes + bulk alike)
Method 2: "z + bulk" — only prune bulk SVs (σ < σ_+), skip spikes entirely
Method 3: "z + bulk + graduated" — within bulk, prune more in deep bulk, less near edge
           z_eff = z * (1 - σ/σ_+)^power

Starts with smallest z and explores outward via TPE.
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
N_TRIALS = 80

LOG_FILE = os.path.join(os.path.dirname(__file__), "haar_optuna_log.txt")


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


# ── Method 1: Just z — threshold ALL singular vectors ─────────────────────
def sv_prune_just_z(W_np, splus, z_thresh):
    """Apply Haar z-threshold to every singular vector, spikes included."""
    U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
    M, N = W_np.shape
    std_U = 1.0 / math.sqrt(M)
    std_V = 1.0 / math.sqrt(N)
    thresh_U = z_thresh * std_U
    thresh_V = z_thresh * std_V

    for i in range(len(S)):
        U[:, i] = np.where(np.abs(U[:, i]) < thresh_U, 0, U[:, i])
        Vt[i, :] = np.where(np.abs(Vt[i, :]) < thresh_V, 0, Vt[i, :])

    return U @ np.diag(S) @ Vt


# ── Method 2: z + bulk — only prune below σ_+ ────────────────────────────
def sv_prune_z_bulk(W_np, splus, z_thresh, bulk_cutoff):
    """Apply Haar z-threshold only to bulk SVs (σ/σ_+ < bulk_cutoff). Skip spikes."""
    U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
    M, N = W_np.shape
    std_U = 1.0 / math.sqrt(M)
    std_V = 1.0 / math.sqrt(N)
    thresh_U = z_thresh * std_U
    thresh_V = z_thresh * std_V

    for i in range(len(S)):
        if S[i] / splus >= bulk_cutoff:
            continue  # skip spikes / near-edge
        U[:, i] = np.where(np.abs(U[:, i]) < thresh_U, 0, U[:, i])
        Vt[i, :] = np.where(np.abs(Vt[i, :]) < thresh_V, 0, Vt[i, :])

    return U @ np.diag(S) @ Vt


# ── Method 3: z + bulk + graduated — power-law within bulk ───────────────
def sv_prune_z_graduated(W_np, splus, z_thresh, bulk_cutoff, power):
    """Graduated pruning: z_eff = z * (1 - σ/σ_+)^power. Deep bulk pruned harder."""
    U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
    M, N = W_np.shape
    std_U = 1.0 / math.sqrt(M)
    std_V = 1.0 / math.sqrt(N)

    for i in range(len(S)):
        ratio = S[i] / splus
        if ratio >= bulk_cutoff:
            continue
        z_eff = z_thresh * (1.0 - ratio / bulk_cutoff) ** power
        U[:, i] = np.where(np.abs(U[:, i]) < z_eff * std_U, 0, U[:, i])
        Vt[i, :] = np.where(np.abs(Vt[i, :]) < z_eff * std_V, 0, Vt[i, :])

    return U @ np.diag(S) @ Vt


# ── Magnitude pruning (shared) ────────────────────────────────────────────
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


# ── Evaluation ────────────────────────────────────────────────────────────
def run_test(model_state, cached_metrics, val_loader, total_params, sv_func):
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    replace_layers(model, ALPHA, BETA, GOF, depth=0)
    model.load_state_dict(model_state)
    model.eval()

    t0 = time.time()

    if sv_func is not None:
        for name, layer in model.named_modules():
            if not isinstance(layer, (SplittableConv, SplittableLinear)):
                continue
            W = layer.get_matrix()
            splus = compute_splus(W)
            W_pruned = sv_func(W, splus)
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


def make_objective(model_state, cached_metrics, val_loader, total_params, baseline_top1):
    def objective(trial):
        method = trial.suggest_categorical("method", ["just_z", "z_bulk", "z_graduated"])

        if method == "just_z":
            z = trial.suggest_float("jz_z", 0.005, 5.0, log=True)
            sv_func = lambda W, sp: sv_prune_just_z(W, sp, z)
            label = f"JustZ z={z:.4f}"

        elif method == "z_bulk":
            z = trial.suggest_float("zb_z", 0.005, 5.0, log=True)
            cutoff = trial.suggest_float("zb_cutoff", 0.3, 1.0)
            sv_func = lambda W, sp: sv_prune_z_bulk(W, sp, z, cutoff)
            label = f"Z+Bulk z={z:.4f} cut={cutoff:.2f}"

        else:  # z_graduated
            z = trial.suggest_float("zg_z", 0.01, 10.0, log=True)
            cutoff = trial.suggest_float("zg_cutoff", 0.3, 1.0)
            power = trial.suggest_int("zg_power", 1, 4)
            sv_func = lambda W, sp: sv_prune_z_graduated(W, sp, z, cutoff, power)
            label = f"Grad z={z:.4f} cut={cutoff:.2f} pow={power}"

        try:
            top1, pct_kept, dt = run_test(
                model_state, cached_metrics, val_loader, total_params, sv_func)

            diff = top1 - baseline_top1
            msg = f"Trial {trial.number:3d} [{method:<12}]: {label:<40}  {top1:7.2f}%  kept={pct_kept:.1f}%  diff={diff:+.2f}pp  {dt:.0f}s"
            print(msg)
            with open(LOG_FILE, "a") as f:
                f.write(msg + "\n")
            return top1

        except Exception as e:
            msg = f"Trial {trial.number:3d} [{method:<12}]: {label:<40}  FAILED: {e}"
            print(msg)
            with open(LOG_FILE, "a") as f:
                f.write(msg + "\n")
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(5)
            return float('-inf')

    return objective


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"3 Haar methods - Optuna search - {N_TRIALS} trials\n")
    print(f"  Method 1: Just z  -- threshold ALL SVs (spikes + bulk)")
    print(f"  Method 2: z+bulk  -- only prune bulk (s < s_+), skip spikes")
    print(f"  Method 3: Graduated -- z*(1-s/s_+)^power within bulk\n")

    with open(LOG_FILE, "w") as f:
        f.write(f"3 Haar methods Optuna search started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {DEVICE}\n\n")

    # Setup — keep on CPU
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

    # Run baseline
    print("Running baseline (no SV pruning)...")
    baseline_model = timm.create_model("vit_base_patch16_224", pretrained=False)
    replace_layers(baseline_model, ALPHA, BETA, GOF, depth=0)
    baseline_model.load_state_dict(model_state)
    baseline_model.eval()
    baseline_model.to(DEVICE)
    magnitude_prune(baseline_model, cached_metrics)
    baseline_top1, _ = evaluate(val_loader, baseline_model, DEVICE)
    baseline_top1 = baseline_top1.cpu().item() if isinstance(baseline_top1, torch.Tensor) else baseline_top1
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    msg = f"Baseline (no SV): {baseline_top1:.2f}%"
    print(msg + "\n")
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n\n")

    # Optuna study
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=15)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="haar_3methods",
    )

    # Seed trials — smallest z first for each method
    # Method 1: Just z
    study.enqueue_trial({"method": "just_z", "jz_z": 0.01})
    study.enqueue_trial({"method": "just_z", "jz_z": 0.05})
    study.enqueue_trial({"method": "just_z", "jz_z": 0.1})
    study.enqueue_trial({"method": "just_z", "jz_z": 0.5})

    # Method 2: z + bulk
    study.enqueue_trial({"method": "z_bulk", "zb_z": 0.01, "zb_cutoff": 1.0})
    study.enqueue_trial({"method": "z_bulk", "zb_z": 0.05, "zb_cutoff": 1.0})
    study.enqueue_trial({"method": "z_bulk", "zb_z": 0.1,  "zb_cutoff": 0.5})
    study.enqueue_trial({"method": "z_bulk", "zb_z": 0.5,  "zb_cutoff": 0.5})

    # Method 3: Graduated — smallest z first
    study.enqueue_trial({"method": "z_graduated", "zg_z": 0.1,  "zg_cutoff": 1.0, "zg_power": 1})
    study.enqueue_trial({"method": "z_graduated", "zg_z": 0.5,  "zg_cutoff": 1.0, "zg_power": 2})
    study.enqueue_trial({"method": "z_graduated", "zg_z": 1.0,  "zg_cutoff": 1.0, "zg_power": 3})
    study.enqueue_trial({"method": "z_graduated", "zg_z": 3.0,  "zg_cutoff": 1.0, "zg_power": 3})

    objective = make_objective(model_state, cached_metrics, val_loader, total_params, baseline_top1)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    # ── Results ───────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"RESULTS - {len(study.trials)} trials")
    print(f"{'='*75}")
    print(f"Baseline: {baseline_top1:.2f}%\n")

    # Best per method
    for meth in ["just_z", "z_bulk", "z_graduated"]:
        meth_trials = [t for t in study.trials
                      if t.params.get("method") == meth
                      and t.value is not None and t.value > 0]
        if meth_trials:
            best = max(meth_trials, key=lambda t: t.value)
            diff = best.value - baseline_top1
            p = {k: v for k, v in best.params.items() if k != "method"}
            print(f"  Best {meth:<12}: {best.value:.2f}% ({diff:+.2f}pp)  {p}")

    print(f"\nOverall best: #{study.best_trial.number}  {study.best_value:.2f}% ({study.best_value - baseline_top1:+.2f}pp)")
    print(f"  {study.best_params}")

    print(f"\nTOP 15 TRIALS:")
    print(f"{'#':>4}  {'method':<12}  {'Top-1':>7}  {'diff':>7}  params")
    print("-" * 75)
    sorted_trials = sorted(study.trials,
                          key=lambda t: t.value if t.value is not None else -999,
                          reverse=True)
    for t in sorted_trials[:15]:
        if t.value is None or t.value == float('-inf'):
            continue
        diff = t.value - baseline_top1
        meth = t.params["method"]
        p = {k: v for k, v in t.params.items() if k != "method"}
        print(f"{t.number:4d}  {meth:<12}  {t.value:7.2f}  {diff:+7.2f}  {p}")

    # Save
    all_results = {
        'baseline_top1': baseline_top1,
        'best_params': study.best_params,
        'best_top1': study.best_value,
        'trials': [
            {
                'number': t.number,
                'params': t.params,
                'top1': t.value if t.value is not None else None,
            }
            for t in study.trials
        ]
    }
    results_file = os.path.join(os.path.dirname(__file__), 'haar_optuna_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {results_file}")

    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*75}\n")
        for meth in ["just_z", "z_bulk", "z_graduated"]:
            meth_trials = [t for t in study.trials
                          if t.params.get("method") == meth
                          and t.value is not None and t.value > 0]
            if meth_trials:
                best = max(meth_trials, key=lambda t: t.value)
                f.write(f"Best {meth}: {best.value:.2f}% {best.params}\n")
        f.write(f"Overall best: {study.best_params} -> {study.best_value:.2f}%\n")
        f.write(f"Finished {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

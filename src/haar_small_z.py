"""
Haar test with SMALL z values.
For bulk singular vectors (σ < σ_+), entries of a Haar-distributed unit vector
in R^n are approximately N(0, 1/n). Prune entries with |entry| < z/√n.

z=0.5 was way too aggressive (pruned ~38% of random entries).
Try z = 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4

Also test: only prune deep-bulk vectors (σ/σ_+ < threshold), leave near-edge alone.
"""

import sys, os, json, math, time, copy, gc
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


def sv_prune_haar(W_np, splus, z_thresh, bulk_cutoff=1.0):
    """
    Prune bulk singular vector entries that look random under Haar.
    Only applies to singular values with σ/σ_+ < bulk_cutoff.
    """
    U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
    M, N = W_np.shape

    std_U = 1.0 / math.sqrt(M)
    std_V = 1.0 / math.sqrt(N)

    for i in range(len(S)):
        ratio = S[i] / splus
        if ratio >= bulk_cutoff:
            continue  # skip spikes and near-edge

        thresh_U = z_thresh * std_U
        thresh_V = z_thresh * std_V

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


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Haar test with small z values\n")

    # Setup — keep on CPU as much as possible to save VRAM
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

    configs = []

    # Baseline
    configs.append(("No SV (baseline)", None))

    # Small z values — all bulk
    for z in [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]:
        configs.append((f"Haar z={z}", lambda W, sp, z=z: sv_prune_haar(W, sp, z)))

    # Deep bulk only (σ/σ_+ < 0.5) with various z
    for z in [0.05, 0.1, 0.2, 0.3, 0.5]:
        configs.append((f"Deep bulk z={z} (σ/σ+<0.5)",
                       lambda W, sp, z=z: sv_prune_haar(W, sp, z, bulk_cutoff=0.5)))

    # Graduated: prune more in deep bulk, less near edge
    # z_effective = z * (1 - σ/σ_+)^power for each singular vector
    def haar_graduated(W_np, splus, z_base, power):
        U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
        M, N = W_np.shape
        std_U = 1.0 / math.sqrt(M)
        std_V = 1.0 / math.sqrt(N)
        for i in range(len(S)):
            if S[i] >= splus:
                continue
            ratio = S[i] / splus
            z_eff = z_base * (1.0 - ratio) ** power
            U[:, i] = np.where(np.abs(U[:, i]) < z_eff * std_U, 0, U[:, i])
            Vt[i, :] = np.where(np.abs(Vt[i, :]) < z_eff * std_V, 0, Vt[i, :])
        return U @ np.diag(S) @ Vt

    for z in [0.5, 1.0, 2.0, 3.0, 5.0]:
        for power in [1, 2, 3]:
            configs.append((f"Grad z={z} pow={power}",
                           lambda W, sp, z=z, p=power: haar_graduated(W, sp, z, p)))

    print(f"Total configs: {len(configs)}")
    print(f"{'Method':<30}  {'Top-1%':>7}  {'Kept%':>7}  {'vs base':>8}  {'Time':>6}")
    print("-" * 65)

    results = []
    baseline_top1 = None

    for label, sv_func in configs:
        try:
            top1, pct_kept, dt = run_test(model_state, cached_metrics, val_loader, total_params, sv_func)
            if baseline_top1 is None:
                baseline_top1 = top1
            diff = top1 - baseline_top1
            marker = " ***" if diff > 0.1 else ""
            print(f"{label:<30}  {top1:7.2f}  {pct_kept:7.1f}  {diff:+8.2f}  {dt:5.0f}s{marker}")
            results.append({'method': label, 'top1': top1, 'pct_kept': pct_kept, 'diff': diff})
        except Exception as e:
            print(f"{label:<30}  FAILED: {e}")
            results.append({'method': label, 'top1': -1, 'pct_kept': -1, 'diff': 0, 'error': str(e)})
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(5)

    print(f"\n{'='*65}")
    print("TOP 10 BY ACCURACY")
    print(f"{'='*65}")
    for r in sorted(results, key=lambda x: x['top1'], reverse=True)[:10]:
        print(f"  {r['method']:<30}  {r['top1']:7.2f}%  ({r['diff']:+.2f}pp)")

    with open(os.path.join(os.path.dirname(__file__), 'haar_small_z_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to haar_small_z_results.json")

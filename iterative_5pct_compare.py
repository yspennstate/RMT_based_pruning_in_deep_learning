"""
Compare iterative 5% smallest-magnitude pruning per layer:
  A) prune 5% -> SV prune (Haar z+bulk z=0.14 cut=0.76) -> repeat
  B) prune 5%                                            -> repeat

Logs accuracy + kept% at each iteration for both strategies.
"""

import sys, os, math, time, copy, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import timm
import numpy as np

from pruning import count_nonzero_params, count_total_params, replace_layers
from SplittableLayers import SplittableConv, SplittableLinear
from validation import evaluate, get_val_dataset
from RMT import bema_inside

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ALPHA, BETA, GOF = 0.25, 0.8, 1
SV_Z = 0.1407
SV_CUT = 0.760
PRUNE_FRAC = 0.05
N_ITERS = 12
LOG_FILE = os.path.join(os.path.dirname(__file__), "iter5pct_log.txt")


def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def compute_splus(W_np):
    M, N = W_np.shape
    p = min(M, N)
    n = max(M, N)
    gram = (W_np @ W_np.T / N) if M <= N else (W_np.T @ W_np / N)
    eigenvals = np.sort(np.linalg.eigvalsh(gram))
    sigma_sq, lamda_plus, l2 = bema_inside(p, n, eigenvals, ALPHA, 0.8)
    return math.sqrt(N * lamda_plus)


def sv_prune_z_bulk(W_np, splus, z_thresh, bulk_cutoff):
    U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
    M, N = W_np.shape
    thresh_U = z_thresh / math.sqrt(M)
    thresh_V = z_thresh / math.sqrt(N)
    for i in range(len(S)):
        if S[i] / splus >= bulk_cutoff:
            continue
        U[:, i] = np.where(np.abs(U[:, i]) < thresh_U, 0, U[:, i])
        Vt[i, :] = np.where(np.abs(Vt[i, :]) < thresh_V, 0, Vt[i, :])
    return U @ np.diag(S) @ Vt


def get_subs(layer):
    return [s for s in layer.modules() if isinstance(s, (torch.nn.Conv2d, torch.nn.Linear))]


def prune_smallest_and_update_mask(model, masks, frac):
    """Among currently-kept entries (mask True), zero the smallest `frac`. Update mask."""
    for name, layer in model.named_modules():
        if not isinstance(layer, (SplittableConv, SplittableLinear)):
            continue
        subs = get_subs(layer)
        if not subs:
            continue
        # Collect currently-kept (mask=True) magnitudes
        live_vals = []
        for i, sub in enumerate(subs):
            m = masks[name][i]
            w_abs = sub.weight.data.abs()
            live_vals.append(w_abs[m])
        live_cat = torch.cat(live_vals) if live_vals else None
        if live_cat is None or len(live_cat) == 0:
            continue
        n_prune = int(len(live_cat) * frac)
        if n_prune <= 0:
            continue
        sorted_vals, _ = torch.sort(live_cat)
        threshold = sorted_vals[n_prune - 1].item()
        with torch.no_grad():
            for i, sub in enumerate(subs):
                m = masks[name][i]
                # Newly pruned: live AND |w| <= threshold
                kill = m & (sub.weight.data.abs() <= threshold)
                m &= ~kill  # update mask in-place
                sub.weight.data *= m.float()


def apply_masks(model, masks):
    with torch.no_grad():
        for name, layer in model.named_modules():
            if not isinstance(layer, (SplittableConv, SplittableLinear)):
                continue
            subs = get_subs(layer)
            for i, sub in enumerate(subs):
                sub.weight.data *= masks[name][i].float()


def init_masks(model):
    masks = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, (SplittableConv, SplittableLinear)):
            continue
        subs = get_subs(layer)
        masks[name] = [torch.ones_like(s.weight.data, dtype=torch.bool) for s in subs]
    return masks


def sv_prune_all_layers(model):
    for name, layer in model.named_modules():
        if not isinstance(layer, (SplittableConv, SplittableLinear)):
            continue
        W = layer.get_matrix()
        splus = compute_splus(W)
        W_pruned = sv_prune_z_bulk(W, splus, SV_Z, SV_CUT)
        layer.set_params("layer1", torch.from_numpy(W_pruned).float(),
                         bias=None, change_bias=False)


def build_model(state):
    m = timm.create_model("vit_base_patch16_224", pretrained=False)
    replace_layers(m, ALPHA, BETA, GOF, depth=0)
    m.load_state_dict(state)
    m.eval()
    return m


def eval_model(model, val_loader, total_params):
    model.to(DEVICE)
    nz = count_nonzero_params(model)
    pct = 100 * nz / total_params
    top1, _ = evaluate(val_loader, model, DEVICE)
    top1 = top1.cpu().item() if isinstance(top1, torch.Tensor) else top1
    return top1, pct


if __name__ == "__main__":
    with open(LOG_FILE, "w") as f:
        f.write(f"Iterative 5% pruning compare started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"SV params: z={SV_Z}, cutoff={SV_CUT}\n")
        f.write(f"Per-iteration prune: {PRUNE_FRAC*100:.0f}%, iterations: {N_ITERS}\n\n")
    log(f"Device: {DEVICE}")

    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)
    val_loader = get_val_dataset(preprocess=preprocess)

    replace_layers(model, ALPHA, BETA, GOF, depth=0)
    total_params = count_total_params(model)
    base_state = copy.deepcopy(model.cpu().state_dict())
    del model; gc.collect(); torch.cuda.empty_cache()

    # Baseline
    m = build_model(base_state)
    top1, pct = eval_model(m, val_loader, total_params)
    log(f"Baseline:                                      {top1:6.2f}%  kept={pct:.2f}%")
    del m; gc.collect(); torch.cuda.empty_cache()

    log("\n=== Strategy A: 5% prune + SV (Haar z+bulk), persistent mask ===")
    log(f"{'iter':>4}  {'top1':>7}  {'kept':>7}  {'time':>6}")
    model_a = build_model(base_state)
    masks_a = init_masks(model_a)
    for it in range(1, N_ITERS + 1):
        t0 = time.time()
        model_a.cpu()
        prune_smallest_and_update_mask(model_a, masks_a, PRUNE_FRAC)
        sv_prune_all_layers(model_a)
        apply_masks(model_a, masks_a)  # re-zero what was previously pruned
        top1, pct = eval_model(model_a, val_loader, total_params)
        log(f"{it:4d}  {top1:6.2f}%  {pct:6.2f}%  {time.time()-t0:5.0f}s")
    del model_a, masks_a; gc.collect(); torch.cuda.empty_cache()

    log("\n=== Strategy B: 5% prune only (no SV), persistent mask ===")
    log(f"{'iter':>4}  {'top1':>7}  {'kept':>7}  {'time':>6}")
    model_b = build_model(base_state)
    masks_b = init_masks(model_b)
    for it in range(1, N_ITERS + 1):
        t0 = time.time()
        model_b.cpu()
        prune_smallest_and_update_mask(model_b, masks_b, PRUNE_FRAC)
        top1, pct = eval_model(model_b, val_loader, total_params)
        log(f"{it:4d}  {top1:6.2f}%  {pct:6.2f}%  {time.time()-t0:5.0f}s")

    log(f"\nFinished {time.strftime('%Y-%m-%d %H:%M:%S')}")

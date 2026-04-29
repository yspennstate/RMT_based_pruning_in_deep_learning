"""
Strategy C: 5% iterative magnitude prune + SV prune, with mask that GROWS from BOTH steps.
After magnitude prune AND after SV reconstruction, lock in any zero entries permanently.
This lets SV decide additional entries to kill, beyond what magnitude chose.
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
SV_ZERO_EPS = 1e-8  # entries with |w|<eps after SV are considered killed
LOG_FILE = os.path.join(os.path.dirname(__file__), "iter_sv_decides_log.txt")


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


def init_masks(model):
    masks = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, (SplittableConv, SplittableLinear)):
            continue
        subs = get_subs(layer)
        masks[name] = [torch.ones_like(s.weight.data, dtype=torch.bool) for s in subs]
    return masks


def magnitude_prune_step(model, masks, frac):
    for name, layer in model.named_modules():
        if not isinstance(layer, (SplittableConv, SplittableLinear)):
            continue
        subs = get_subs(layer)
        if not subs:
            continue
        live_vals = []
        for i, sub in enumerate(subs):
            m = masks[name][i]
            live_vals.append(sub.weight.data.abs()[m])
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
                kill = m & (sub.weight.data.abs() <= threshold)
                m &= ~kill
                sub.weight.data *= m.float()


def sv_prune_step(model, masks):
    """SV reconstruct, then update mask: anything that became ~0 is locked in."""
    for name, layer in model.named_modules():
        if not isinstance(layer, (SplittableConv, SplittableLinear)):
            continue
        W = layer.get_matrix()
        splus = compute_splus(W)
        W_pruned = sv_prune_z_bulk(W, splus, SV_Z, SV_CUT)
        layer.set_params("layer1", torch.from_numpy(W_pruned).float(),
                         bias=None, change_bias=False)
        # Now update masks based on what SV produced
        subs = get_subs(layer)
        with torch.no_grad():
            for i, sub in enumerate(subs):
                # Live entries that became near-zero get killed
                still_live = sub.weight.data.abs() > SV_ZERO_EPS
                masks[name][i] &= still_live
                # Re-apply mask to enforce previously-killed entries stay zero
                sub.weight.data *= masks[name][i].float()


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
        f.write(f"Strategy C (SV decides) started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"SV params: z={SV_Z}, cutoff={SV_CUT}\n")
        f.write(f"Per-iter mag prune: {PRUNE_FRAC*100:.0f}%, iters: {N_ITERS}\n\n")
    log(f"Device: {DEVICE}")

    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)
    val_loader = get_val_dataset(preprocess=preprocess)

    replace_layers(model, ALPHA, BETA, GOF, depth=0)
    total_params = count_total_params(model)
    base_state = copy.deepcopy(model.cpu().state_dict())
    del model; gc.collect(); torch.cuda.empty_cache()

    log("\n=== Strategy C: 5% mag prune + SV decides extra kills ===")
    log(f"{'iter':>4}  {'top1':>7}  {'kept':>7}  {'time':>6}")
    m = build_model(base_state)
    masks = init_masks(m)
    for it in range(1, N_ITERS + 1):
        t0 = time.time()
        m.cpu()
        magnitude_prune_step(m, masks, PRUNE_FRAC)
        sv_prune_step(m, masks)  # also updates masks
        top1, pct = eval_model(m, val_loader, total_params)
        log(f"{it:4d}  {top1:6.2f}%  {pct:6.2f}%  {time.time()-t0:5.0f}s")

    log(f"\nFinished {time.strftime('%Y-%m-%d %H:%M:%S')}")

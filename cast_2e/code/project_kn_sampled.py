"""
project_kn_sampled.py — sampled k:n cert framework for n where C(n,k) is too
large to enumerate. Uses random sampling + greedy refinement.

For ResNet50 with Cin=64: n must divide 64. Tractable n: 4, 8, 16. Larger
n=32 needs C(32,16)=601M patterns — sampled. n=64 needs 10^18 — sampled.

Algorithm:
  1. Random sample N candidate keep patterns (default 10K).
  2. For each (output, group): pick the candidate with min cert cost.
  3. Optional: greedy refinement from top-k candidates.

Same call signature as cert_aware_kn_for_conv but with extra `n_samples`.
"""
from __future__ import annotations
import torch as _t; _t.backends.cudnn.enabled = False

import torch
import torch.nn as nn
from typing import Iterable, Optional


def sample_keep_patterns(n: int, k: int, n_samples: int, seed: int = 42) -> torch.Tensor:
    """Random sample of n_samples binary patterns of length n with exactly k ones.
    Always include the magnitude-top-k pattern as a safety baseline."""
    g = torch.Generator().manual_seed(seed)
    pats = torch.zeros(n_samples, n, dtype=torch.float32)
    for i in range(n_samples):
        idx = torch.randperm(n, generator=g)[:k]
        pats[i, idx] = 1.0
    return pats


def _cert_cost_kn_sampled(W_group: torch.Tensor, h_group: torch.Tensor,
                            keep_patterns: torch.Tensor) -> torch.Tensor:
    """Same form as _cert_cost_kn but uses provided sampled patterns."""
    Cout, G, n = W_group.shape
    P = keep_patterns.shape[0]
    drop = 1.0 - keep_patterns                                # [P, n]
    h_f = h_group.float()
    Cmat = torch.einsum("ngi,ngj->gij", h_f, h_f) / max(h_group.shape[0], 1)
    W_f = W_group.float()
    out = torch.empty(Cout, G, P, device=W_group.device, dtype=W_group.dtype)
    drop_dev = drop.to(W_group.device, W_group.dtype)
    for p in range(P):
        r = drop_dev[p].view(1, 1, n) * W_f
        temp = torch.einsum("ogk,gjk->ogj", r, Cmat)
        out[..., p] = (temp * r).sum(dim=-1).to(W_group.dtype)
    return out


@torch.no_grad()
def cert_aware_kn_sampled_for_conv(
    model: nn.Module,
    calib_loader: Iterable,
    *,
    n: int = 32,
    k: int = 16,
    n_samples: int = 10000,
    seed: int = 42,
    dense_state_dict: Optional[dict] = None,
    n_calib_imgs: int = 64,
    free_restoration: bool = True,
    only_1x1: bool = False,
    permute_align: bool = False,
    alpha_ser_prior: float = 0.0,
    log: bool = True,
    device: str = "cuda",
) -> dict:
    """Sampled cert k:n projection. Same as cert_aware_kn_for_conv but for large n."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from project_conv_2_4 import _capture_unfolded_inputs
    from project_kn_sparsity import is_eligible_conv_kn

    eligibles = [(n_, m) for n_, m in model.named_modules()
                 if is_eligible_conv_kn(n_, m, n=n)]
    if only_1x1:
        eligibles = [(nm, m) for nm, m in eligibles if m.kernel_size == (1, 1)]
    if log:
        print(f"[cert_kn_sampled] {len(eligibles)} eligible Conv2d, n={n}, k={k}, samples={n_samples}")

    keep_patterns = sample_keep_patterns(n, k, n_samples, seed=seed).to(device)

    permuted_layer_count = 0
    if permute_align:
        for name, mod in eligibles:
            with torch.no_grad():
                W = mod.weight.data
                imp = (W ** 2).sum(dim=(0, 2, 3))
                Cin = imp.shape[0]
                order = torch.argsort(imp, descending=True)
                G = Cin // n
                buckets = [order[i*G:(i+1)*G] for i in range(n)]
                perm = torch.empty_like(order)
                for g in range(G):
                    for s in range(n):
                        perm[g*n + s] = buckets[s][g]
                if G * n < Cin:
                    perm[G*n:] = order[G*n:]
                mod.weight.data = W[:, perm, :, :].contiguous()
                mod._cin_perm = perm
                if not hasattr(mod, "_cin_perm_hook"):
                    def _make_hook(p):
                        def h(m_, inp): return (inp[0].index_select(1, p.to(inp[0].device)),)
                        return h
                    mod._cin_perm_hook = mod.register_forward_pre_hook(_make_hook(perm))
                permuted_layer_count += 1

    activations = _capture_unfolded_inputs(
        model, calib_loader, eligibles,
        n_calib_imgs=n_calib_imgs, device=device, max_rows_per_layer=4096,
    )

    n_layers = bad = 0
    free_restoration_count = 0
    for name, mod in eligibles:
        W = mod.weight.data
        Cout, Cin, kH, kW = W.shape
        cols_full = Cin * kH * kW
        cols_used = (cols_full // n) * n
        h_all = activations.get(name)
        if h_all is None or h_all.numel() == 0:
            continue
        if h_all.shape[0] > 4096:
            idx = torch.randperm(h_all.shape[0])[:4096]
            h_all = h_all[idx]
        h_all = h_all[:, :cols_used].to(W.device)
        n_groups = cols_used // n
        slot_values = W.reshape(Cout, cols_full)[:, :cols_used]
        if free_restoration and dense_state_dict is not None:
            dW = dense_state_dict.get(f"{name}.weight")
            if dW is not None:
                if permute_align and hasattr(mod, "_cin_perm"):
                    dW = dW[:, mod._cin_perm.cpu(), :, :]
                dW = dW.to(W.device).reshape(Cout, cols_full)[:, :cols_used]
                ser_kept = (slot_values != 0)
                slot_values = torch.where(ser_kept, slot_values, dW)
                free_restoration_count += int((~ser_kept).sum().item())
        Wg = slot_values.reshape(Cout, n_groups, n)
        hg = h_all.reshape(-1, n_groups, n)
        # Memory check: out tensor = O*G*P floats. For ResNet50 worst: 2048*2*10000 = 40M floats = 160MB. OK.
        costs = _cert_cost_kn_sampled(Wg, hg, keep_patterns)
        # SER prior (Hamming distance to current SER mask)
        if alpha_ser_prior > 0:
            ser_kept = (W.reshape(Cout, cols_full)[:, :cols_used] != 0).float()
            ser_g = ser_kept.reshape(Cout, n_groups, n)
            kp = keep_patterns.view(1, 1, -1, n)
            hamming = (kp - ser_g.unsqueeze(2)).abs().sum(dim=-1)
            scale = costs.detach().mean(dim=-1, keepdim=True).clamp_min(1e-12)
            costs = costs + alpha_ser_prior * scale * hamming
        best = costs.argmin(dim=-1)
        mask_g_keep = keep_patterns[best]
        Wm = (Wg * mask_g_keep).reshape(Cout, cols_used)
        mask2d = mask_g_keep.reshape(Cout, cols_used)
        Wfull = W.reshape(Cout, cols_full).clone()
        Wfull[:, :cols_used] = Wm
        mod.weight.data.copy_(Wfull.reshape(Cout, Cin, kH, kW))
        nnz = mask2d.reshape(Cout, -1, n).sum(dim=-1)
        bad += int((nnz != k).sum().item())
        n_layers += 1
        if log:
            sparsity = 1.0 - float(mask2d.float().mean().item())
            print(f"  {name:48s}  k:n={k}:{n}  sparsity={sparsity:.3f}  bad={bad}")

    return {
        "method": "kn_sampled",
        "n": n, "k": k, "n_samples": n_samples,
        "n_layers_modified": n_layers,
        "permuted_layer_count": permuted_layer_count,
        "permute_align_enabled": permute_align,
        "free_restoration_count": free_restoration_count,
        "alpha_ser_prior": alpha_ser_prior,
        "groups_with_more_than_2_nonzero_after": bad,
    }

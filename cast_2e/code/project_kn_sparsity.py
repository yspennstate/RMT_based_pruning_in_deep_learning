"""
project_kn_sparsity.py — generalized k:n cert-aware sparsity for both Conv2d and
Linear layers. k:n means "keep exactly k of every n contiguous Cin entries."

Specializations of interest:
  k=2, n=4  — 2:4    (NVIDIA Ampere+ tensor cores, 50% sparse, 2× speedup)
  k=4, n=8  — 4:8    (Hopper+ tensor cores, 50% sparse, 2× speedup, more flexibility)
  k=1, n=4  — 1:4    (Hopper+ N:M, 75% sparse, 4× theoretical speedup)
  k=3, n=4  — 3:4    (25% sparse, very mild, useful as ablation)

The cert objective is the same elastic-net form as project_conv_2_4.py:
  cost(m) = E_x[ ‖((1-m)·W) · h(x)‖² ]   +  α_ser · Hamming(m, SER_mask)

with C(n,k) candidate keep patterns per row per quartile. The optimum is found
by brute-force enumeration (cheap; C(8,4)=70, C(4,2)=6, C(4,1)=4).

Reuses project_conv_2_4.py for:
  - compute_cin_permutation
  - _capture_unfolded_inputs
  - is_eligible_conv (with relaxed divisibility)
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _all_keep_patterns(n: int, k: int) -> torch.Tensor:
    """All C(n,k) binary patterns of length n with exactly k ones (= keep slots)."""
    pats = []
    for combo in combinations(range(n), k):
        p = [0] * n
        for i in combo:
            p[i] = 1
        pats.append(p)
    return torch.tensor(pats, dtype=torch.float32)


def _cert_cost_kn(W_group: torch.Tensor, h_group: torch.Tensor,
                   n: int, k: int) -> torch.Tensor:
    """Generalized cert cost for k:n sparsity.

    For each n-tuple, enumerate C(n,k) keep patterns and return the EXACT
    covariance-form certificate cost
        c_g(m) = E_x[ ‖((1-m)·W_g) · h_g(x)‖² ]
                = r^T · C_g · r           where r = (1-m) ⊙ W_g, C_g = E[h_g h_g^T]

    W_group: [Cout, G, n]
    h_group: [N,    G, n]
    Returns: [Cout, G, P]   where P = C(n,k)   (lower = better)
    """
    Cout, G, _ = W_group.shape
    keep = _all_keep_patterns(n, k).to(W_group.device, W_group.dtype)   # [P, n]
    drop = 1.0 - keep                                                    # [P, n]
    P = keep.shape[0]
    h_f = h_group.float()
    Cmat = torch.einsum("ngi,ngj->gij", h_f, h_f) / max(h_group.shape[0], 1)  # [G, n, n]
    W_f = W_group.float()
    out = torch.empty(Cout, G, P, device=W_group.device, dtype=W_group.dtype)
    for p in range(P):
        r = drop[p].view(1, 1, n) * W_f                                  # [O, G, n]
        temp = torch.einsum("ogk,gjk->ogj", r, Cmat)                     # [O, G, n]
        out[..., p] = (temp * r).sum(dim=-1).to(W_group.dtype)
    return out


def _hamming_to_ser(ser_kept: torch.Tensor, n: int, k: int,
                    keep_patterns: torch.Tensor) -> torch.Tensor:
    """Hamming distance from each candidate keep pattern to the SER source mask.
    ser_kept: [Cout, cols_used] bool/float — 1 where SER kept the weight
    keep_patterns: [P, n]
    Returns: [Cout, n_groups, P]
    """
    Cout, cols = ser_kept.shape
    n_groups = cols // n
    ser_g = ser_kept.reshape(Cout, n_groups, n).float()                  # [O, G, n]
    kp = keep_patterns.view(1, 1, -1, n)                                  # [1, 1, P, n]
    return (kp - ser_g.unsqueeze(2)).abs().sum(dim=-1)                   # [O, G, P]


def is_eligible_conv_kn(name: str, mod: nn.Module, n: int = 4) -> bool:
    """Eligible: Conv2d with Cin divisible by n. Skips downsample 1x1 = same as 2:4 case."""
    if not isinstance(mod, nn.Conv2d):
        return False
    out_c, in_c = mod.weight.shape[:2]
    if in_c % n != 0 or out_c < n:
        return False
    return True


def is_eligible_linear_kn(name: str, mod: nn.Module, n: int = 4) -> bool:
    if not isinstance(mod, nn.Linear):
        return False
    out_f, in_f = mod.weight.shape
    return in_f % n == 0 and out_f >= n


@torch.no_grad()
def cert_aware_kn_for_conv(
    model: nn.Module,
    calib_loader: Iterable,
    *,
    n: int = 4,
    k: int = 2,
    dense_state_dict: Optional[dict] = None,
    n_calib_imgs: int = 64,
    free_restoration: bool = True,
    only_1x1: bool = False,
    permute_align: bool = False,
    alpha_ser_prior: float = 0.0,
    log: bool = True,
    device: str = "cuda",
) -> dict:
    """Project every eligible Conv2d to k:n sparsity using the cert framework.

    Parameters
    ----------
    n, k : int
        k:n sparsity pattern. k=2,n=4 is standard 2:4; k=4,n=8 is 4:8 (Hopper).
    dense_state_dict : dict | None
        Dense pretrained weights (used for free_restoration). If None,
        free_restoration is disabled.
    permute_align : bool
        Apply Cin-permutation alignment ("flatten the layer", variant B).
    alpha_ser_prior : float
        Weight on the Hamming prior toward the SER (source) keep mask.
        SER mask is read from the CURRENT student weights (which we assume is
        the SER-pruned ckpt at this point in the pipeline).
    """
    # Reuse project_conv_2_4 helpers (only the activation capture; perm we do here).
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from project_conv_2_4 import _capture_unfolded_inputs

    def _kn_cin_permutation(imp: torch.Tensor, group_size: int) -> torch.LongTensor:
        """Importance-balanced n-tuple interleave: rank Cin by imp, partition into
        n_groups buckets ordered by importance, then interleave so every n-tuple
        has 1 channel from each bucket. Generalizes the 2:4 quartile-balance idea."""
        Cin = imp.shape[0]
        order = torch.argsort(imp, descending=True)
        G = Cin // group_size
        buckets = [order[i * G:(i + 1) * G] for i in range(group_size)]
        perm = torch.empty_like(order)
        for g in range(G):
            for s in range(group_size):
                perm[g * group_size + s] = buckets[s][g]
        # Append any leftover cols at the tail unchanged
        if G * group_size < Cin:
            perm[G * group_size:] = order[G * group_size:]
        return perm

    def _apply_cin_perm_to_conv(mod, perm):
        with torch.no_grad():
            mod.weight.data = mod.weight.data[:, perm, :, :].contiguous()

    def _make_perm_pre_hook(perm):
        def _hook(mod_, inp):
            x = inp[0]
            return (x.index_select(1, perm.to(x.device)),)
        return _hook

    eligibles = [(n_, m) for n_, m in model.named_modules()
                 if is_eligible_conv_kn(n_, m, n=n)]
    if only_1x1:
        eligibles = [(nm, m) for nm, m in eligibles
                     if m.kernel_size == (1, 1)]
    if log:
        print(f"[cert_kn] {len(eligibles)} eligible Conv2d layers (n={n}, k={k})")

    keep_patterns = _all_keep_patterns(n, k).to(device)                   # [P, n]
    P = keep_patterns.shape[0]

    # Step 0: compute and apply Cin permutations BEFORE capturing activations
    permuted_layer_count = 0
    if permute_align:
        for name, mod in eligibles:
            with torch.no_grad():
                # Importance signal: per-Cin-column ‖W[:, c, :, :]‖₂² over output channels
                W = mod.weight.data
                imp = (W ** 2).sum(dim=(0, 2, 3))                         # [Cin]
                perm = _kn_cin_permutation(imp, group_size=n)
                _apply_cin_perm_to_conv(mod, perm)
                mod._cin_perm = perm                                       # for state_dict round-trip
                if not hasattr(mod, "_cin_perm_hook_handle"):
                    h = mod.register_forward_pre_hook(_make_perm_pre_hook(perm))
                    mod._cin_perm_hook_handle = h
                permuted_layer_count += 1
        if log:
            print(f"[cert_kn] applied Cin perm to {permuted_layer_count} layers")

    # Step 1: capture activations from the (possibly permuted) model
    activations = _capture_unfolded_inputs(
        model, calib_loader, eligibles,
        n_calib_imgs=n_calib_imgs, device=device, max_rows_per_layer=4096,
    )

    # Step 2: project each layer
    n_layers_modified = 0
    free_restoration_count = 0
    bad_groups_total = 0
    layer_stats = []
    for name, mod in eligibles:
        W = mod.weight.data
        Cout, Cin, kH, kW = W.shape
        cols_full = Cin * kH * kW
        cols_used = (cols_full // n) * n                                  # round down to mult of n
        W2d = W.reshape(Cout, cols_full)
        Wleft = W2d[:, :cols_used]                                        # [Cout, cols_used]

        # SER mask = current student weight is non-zero
        ser_kept = (Wleft != 0).bool()                                    # [Cout, cols_used]

        # Slot values (where the kept slots will be sourced from)
        if free_restoration and dense_state_dict is not None:
            dense_W = dense_state_dict.get(f"{name}.weight")
            if dense_W is None:
                slot_values = Wleft.clone()
            else:
                # Apply the same Cin perm to dense_W
                if permute_align and hasattr(mod, "_cin_perm"):
                    perm = mod._cin_perm
                    dense_W = dense_W[:, perm.cpu(), :, :]
                dense_W = dense_W.to(W.device)
                Wdense2d = dense_W.reshape(Cout, cols_full)[:, :cols_used]
                slot_values = torch.where(ser_kept, Wleft, Wdense2d)
                free_restoration_count += int((~ser_kept).sum().item())
        else:
            slot_values = Wleft.clone()

        h_all = activations.get(name)
        if h_all is None or h_all.numel() == 0:
            # Fallback to magnitude on slot_values
            mag = slot_values.abs()
            n_groups_tmp = cols_used // n
            mag_g = mag.reshape(Cout, n_groups_tmp, n)
            topk = mag_g.topk(k, dim=-1).indices
            mask_flat = torch.zeros_like(mag_g)
            mask_flat.scatter_(-1, topk, 1.0)
            Wm = (mag_g.sign() * mag_g.abs() * mask_flat).reshape(Cout, cols_used)  # noop, signs preserved via slot_values
            # Use slot_values not mag for actual data; redo cleanly:
            Wg = slot_values.reshape(Cout, n_groups_tmp, n)
            Wm = (Wg * mask_flat).reshape(Cout, cols_used)
            mask2d = mask_flat.reshape(Cout, cols_used)
            mode_used = "magnitude_fallback"
        else:
            if h_all.shape[0] > 4096:
                idx = torch.randperm(h_all.shape[0])[:4096]
                h_all = h_all[idx]
            h_all = h_all[:, :cols_used].to(W.device)
            n_groups = cols_used // n
            Wg = slot_values.reshape(Cout, n_groups, n)
            hg = h_all.reshape(-1, n_groups, n)
            costs = _cert_cost_kn(Wg, hg, n, k)                           # [O, G, P]
            if alpha_ser_prior > 0:
                hamming = _hamming_to_ser(ser_kept, n, k, keep_patterns)  # [O, G, P]
                scale = costs.detach().mean(dim=-1, keepdim=True).clamp_min(1e-12)
                costs = costs + alpha_ser_prior * scale * hamming
            best = costs.argmin(dim=-1)                                   # [O, G]
            mask_g_keep = keep_patterns[best]                              # [O, G, n]
            Wm = (Wg * mask_g_keep).reshape(Cout, cols_used)
            mask2d = mask_g_keep.reshape(Cout, cols_used)
            mode_used = "cert_kn"

        # Write back
        W2d_new = W2d.clone()
        W2d_new[:, :cols_used] = Wm
        mod.weight.data.copy_(W2d_new.reshape(Cout, Cin, kH, kW))

        # Sanity: every group has exactly k nonzeros
        Mg = mask2d.reshape(Cout, -1, n)
        nnz_per_group = Mg.sum(dim=-1)
        bad = int((nnz_per_group != k).sum().item())
        bad_groups_total += bad

        n_layers_modified += 1
        if log:
            sparsity = 1.0 - float(mask2d.float().mean().item())
            print(f"  {name:48s}  k:n={k}:{n}  sparsity={sparsity:.3f}  "
                  f"mode={mode_used}  bad={bad}")
        layer_stats.append({
            "name": name, "n": n, "k": k,
            "Cout": Cout, "Cin": Cin, "cols_used": cols_used,
            "bad_groups": bad,
        })

    return {
        "n": n, "k": k,
        "n_layers_modified": n_layers_modified,
        "permuted_layer_count": permuted_layer_count,
        "permute_align_enabled": permute_align,
        "free_restoration_count": free_restoration_count,
        "alpha_ser_prior": alpha_ser_prior,
        "groups_with_more_than_2_nonzero_after": bad_groups_total,  # naming kept for compat
        "layers": layer_stats,
    }


@torch.no_grad()
def cert_aware_kn_for_linear(
    model: nn.Module,
    calib_loader: Iterable,
    *,
    n: int = 4,
    k: int = 2,
    dense_state_dict: Optional[dict] = None,
    n_calib_imgs: int = 64,
    free_restoration: bool = True,
    permute_align: bool = False,
    alpha_ser_prior: float = 0.0,
    log: bool = True,
    device: str = "cuda",
) -> dict:
    """Same as cert_aware_kn_for_conv but for nn.Linear (ViT-B etc.)."""
    eligibles = [(nm, m) for nm, m in model.named_modules()
                 if is_eligible_linear_kn(nm, m, n=n)]
    if log:
        print(f"[cert_kn-linear] {len(eligibles)} eligible Linear layers (n={n}, k={k})")

    keep_patterns = _all_keep_patterns(n, k).to(device)                   # [P, n]

    # Optional Cin permute (for Linear: in_features = Cin)
    permuted_layer_count = 0
    if permute_align:
        for name, mod in eligibles:
            W = mod.weight.data                                            # [Out, In]
            imp = (W ** 2).sum(dim=0)                                      # [In]
            # Quartile-balanced permutation: greedy interleave (same logic as conv)
            order = torch.argsort(imp, descending=True)
            G = imp.shape[0] // n
            buckets = [order[i*G:(i+1)*G] for i in range(n)]
            perm = torch.empty_like(order)
            for g in range(G):
                for s in range(n):
                    perm[g*n + s] = buckets[s][g]
            mod.weight.data = W[:, perm].contiguous()
            mod._cin_perm = perm
            if not hasattr(mod, "_cin_perm_hook_lin"):
                def _hook(p):
                    def h(mod_, inp):
                        x = inp[0]
                        return (x.index_select(-1, p.to(x.device)),)
                    return h
                handle = mod.register_forward_pre_hook(_hook(perm))
                mod._cin_perm_hook_lin = handle
            permuted_layer_count += 1
        if log:
            print(f"[cert_kn-linear] permuted {permuted_layer_count} Linear layers")

    # Capture activations
    captures: dict[str, list] = {nm: [] for nm, _ in eligibles}
    handles = []
    for nm, mod in eligibles:
        def _make(name):
            def hook(m, inp, out):
                x = inp[0].detach()
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                if x.shape[0] > 4096:
                    idx = torch.randperm(x.shape[0])[:4096]
                    x = x[idx]
                captures[name].append(x.cpu())
            return hook
        h = mod.register_forward_hook(_make(nm))
        handles.append(h)
    seen = 0
    model.eval()
    for batch in calib_loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=True)
        _ = model(x)
        seen += x.shape[0]
        if seen >= n_calib_imgs:
            break
    for h in handles:
        h.remove()

    # Project
    n_layers_modified = 0
    bad_groups_total = 0
    free_restoration_count = 0
    layer_stats = []
    for name, mod in eligibles:
        W = mod.weight.data                                                # [Out, In]
        Cout, Cin = W.shape
        cols_used = (Cin // n) * n
        Wleft = W[:, :cols_used]
        ser_kept = (Wleft != 0).bool()

        if free_restoration and dense_state_dict is not None:
            dense_W_full = dense_state_dict.get(f"{name}.weight")
            if dense_W_full is None:
                slot_values = Wleft.clone()
            else:
                if permute_align and hasattr(mod, "_cin_perm"):
                    dense_W_full = dense_W_full[:, mod._cin_perm.cpu()]
                dense_W_full = dense_W_full.to(W.device)
                slot_values = torch.where(ser_kept, Wleft, dense_W_full[:, :cols_used])
                free_restoration_count += int((~ser_kept).sum().item())
        else:
            slot_values = Wleft.clone()

        h_all = torch.cat(captures[name], dim=0) if captures[name] else None
        if h_all is None or h_all.numel() == 0:
            # Magnitude fallback
            mag = slot_values.abs()
            n_groups = cols_used // n
            mag_g = mag.reshape(Cout, n_groups, n)
            topk = mag_g.topk(k, dim=-1).indices
            mask_flat = torch.zeros_like(mag_g)
            mask_flat.scatter_(-1, topk, 1.0)
            Wm = (slot_values.reshape(Cout, n_groups, n) * mask_flat).reshape(Cout, cols_used)
            mask2d = mask_flat.reshape(Cout, cols_used)
        else:
            h_all = h_all[:, :cols_used].to(W.device)
            n_groups = cols_used // n
            Wg = slot_values.reshape(Cout, n_groups, n)
            hg = h_all.reshape(-1, n_groups, n)
            costs = _cert_cost_kn(Wg, hg, n, k)                            # [O, G, P]
            if alpha_ser_prior > 0:
                hamming = _hamming_to_ser(ser_kept, n, k, keep_patterns)
                scale = costs.detach().mean(dim=-1, keepdim=True).clamp_min(1e-12)
                costs = costs + alpha_ser_prior * scale * hamming
            best = costs.argmin(dim=-1)
            mask_g_keep = keep_patterns[best]
            Wm = (Wg * mask_g_keep).reshape(Cout, cols_used)
            mask2d = mask_g_keep.reshape(Cout, cols_used)

        W_new = W.clone()
        W_new[:, :cols_used] = Wm
        mod.weight.data.copy_(W_new)

        nnz_per_group = mask2d.reshape(Cout, -1, n).sum(dim=-1)
        bad = int((nnz_per_group != k).sum().item())
        bad_groups_total += bad
        n_layers_modified += 1

    return {
        "n": n, "k": k,
        "n_layers_modified": n_layers_modified,
        "permuted_layer_count": permuted_layer_count,
        "permute_align_enabled": permute_align,
        "free_restoration_count": free_restoration_count,
        "alpha_ser_prior": alpha_ser_prior,
        "groups_with_more_than_2_nonzero_after": bad_groups_total,
    }

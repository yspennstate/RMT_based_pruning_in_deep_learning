"""
project_conv_2_4.py — CAST-2E extension to nn.Conv2d layers.

Reuses the certificate-aware 2-of-4 selection from CAST-2E-for-Linear by treating
each conv weight as a 2-D matrix `[Cout, Cin*kH*kW]` (the im2col-equivalent shape).
The certificate cost `c_g(m) = ‖((1-m)*W_g) · h_g(x)‖_p` is computed with the
unfolded input tensor for h_g(x), exactly mirroring the Linear case.

Two projection variants are exposed:

  - magnitude_2_4_for_conv(model, ...): pick top-2 of every contiguous 4-tuple
    by absolute weight magnitude. No calibration data needed. This is the
    "free 2:4 restoration" variant when paired with a SER-pre-pruned ckpt — for
    each 4-tuple where the SER mask zeroed >2 weights, we restore back up to 2-of-4
    using the dense pretrained values.

  - cert_aware_2_4_for_conv(model, calib_loader, dense_state_dict, ...):
    for each 4-tuple, evaluate all 6 possible 2-of-4 patterns by the certificate
    cost and keep the argmin. Restoration uses dense weights when the SER mask
    left fewer than 2 NNZ in a tuple.

Both produce in-place changes to `model`. They return a stats dict suitable for
JSON audit output (matching the schema used by the Linear path's
`two_four_stats` block).

Skip rules:
  - Conv2d with `groups != 1` (depthwise / grouped — typically too few channels
    per group to support a 2-of-4 partition along the input axis).
  - Conv2d with `in_channels < 4` (the 3→64 stem).
  - Optionally skip 1×1 convs when their `in_channels < 8`.

This file is self-contained: no dependency on the v11_pod_debug Linear-path code.
The output schema mirrors `two_four_stats` so downstream (FT runner, eval,
postsweep) treats Conv layers the same as Linear layers.

Expected ResNet-50 coverage: approximately 52 eligible Conv2d layers (all 3×3
and 1×1 layers in the bottleneck blocks). The 7×7 stem is skipped because
Cin=3.

Acceleration note: PyTorch `torch.sparse.to_sparse_semi_structured` accelerates
`nn.Linear` weights only. For ResNet, this implementation reports FLOP reduction
and accuracy preservation; measured Conv2d wall-clock speedup requires a
separate inference-kernel path such as `nn.Unfold + nn.Linear + Fold`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Cin permutation alignment ("flatten-the-layer" — see PERMUTATION_ALIGN_DESIGN.md)
#
# The unstructured RMT prior produces a sparsity that does NOT align with the
# 2-of-4 hardware partition (which is an arbitrary contiguous chunk of input
# channels per row). When the projection has to keep exactly 2 of every 4
# contiguous Cin slots, an *interleaved* permutation of Cin can push the
# RMT-signal columns into different 4-tuples, so each tuple has at least one
# signal column for the 2:4 mask to keep instead of clustering all signal into
# the same tuple. This is the same idea as a cyclic block-row reordering for
# block-sparse SpMV, but applied to the 2:4 hardware partition.
#
# At inference, the permutation is absorbed via a forward pre-hook on the
# Conv2d (free at runtime, just an index_select). The permutation is stored
# as a buffer on the module so state_dict save/load round-trips correctly.
# ---------------------------------------------------------------------------


def compute_cin_permutation(
    weight: torch.Tensor,           # [Cout, Cin, kH, kW]
    h_unfolded: torch.Tensor,       # [N, Cin*kH*kW]; from _capture_unfolded_inputs
) -> torch.LongTensor:
    """Return a permutation `π` over Cin such that the 4-tuples of `W[:, π]`
    have interleaved importance: top-quartile importance lands in position 0
    of every 4-tuple, 2nd quartile in position 1, etc.

    Importance per Cin channel:
        I(c) = ‖W[:, c, :, :]‖_F^2  ·  E_x[‖h(x)[c]‖^2]
    averaged over kernel-spatial positions for kH*kW > 1 convs.

    Returns a LongTensor of shape [Cin]. For 1×1 convs the partition lands
    exactly on Cin boundaries, so this directly determines 4-tuple structure.
    For 3×3 convs the unfolded axis is Cin*9; the Cin-only permutation only
    affects which Cin's 9-slot block sits adjacent to which other (a smaller
    but still measurable effect on cross-boundary 4-tuples).
    """
    Cout, Cin, kH, kW = weight.shape
    if Cin < 4:
        return torch.arange(Cin, dtype=torch.long, device=weight.device)

    if h_unfolded.numel() > 0:
        h_per_col = h_unfolded.pow(2).mean(dim=0)            # [Cin*kH*kW]
        h_per_cin = h_per_col.reshape(Cin, kH * kW).mean(dim=1)  # [Cin]
    else:
        h_per_cin = torch.ones(Cin, device=weight.device)

    w_per_cin = weight.detach().pow(2).sum(dim=(0, 2, 3))    # [Cin]
    importance = h_per_cin.to(w_per_cin.device) * w_per_cin  # [Cin]

    # Sort Cin channels by importance descending
    sorted_idx = importance.detach().cpu().argsort(descending=True)  # [Cin]

    n_groups = Cin // 4
    perm = torch.empty(Cin, dtype=torch.long)
    # Interleave: position k of every 4-tuple gets the k-th quartile of importance.
    # → group g, position k = sorted_idx[k * n_groups + g]
    for k in range(4):
        for g in range(n_groups):
            perm[g * 4 + k] = sorted_idx[k * n_groups + g].item()
    # Append any tail columns (Cin % 4 != 0) in their natural order.
    tail = Cin - n_groups * 4
    if tail > 0:
        used = set(perm[: n_groups * 4].tolist())
        unused = [c for c in range(Cin) if c not in used]
        for i, c in enumerate(unused):
            perm[n_groups * 4 + i] = c
    return perm


def _make_permute_pre_hook(perm: torch.Tensor):
    """Return a forward pre-hook that index_selects the Cin axis of input."""
    def _hook(_module, inputs):
        x = inputs[0]
        p = _module._cin_perm.to(x.device)
        return (x.index_select(1, p),)
    return _hook


def apply_cin_permutation_to_conv(
    conv: nn.Conv2d,
    perm: torch.LongTensor,
) -> None:
    """In-place: permute `conv.weight` along the Cin axis by `perm`, register
    `perm` as a buffer on the module so it round-trips through state_dict, and
    install a forward pre-hook that permutes incoming activations by the same
    perm. After this, the conv operates in the *permuted* Cin order — every
    subsequent operation (cert-cost search, 2:4 projection, FT mask freeze)
    just sees a normal-looking conv with reordered Cin.
    """
    if perm.shape[0] != conv.in_channels:
        raise ValueError(
            f"perm has {perm.shape[0]} entries but conv has Cin={conv.in_channels}"
        )
    perm = perm.to(conv.weight.device, dtype=torch.long).contiguous()
    with torch.no_grad():
        conv.weight.data = conv.weight.data[:, perm, :, :].contiguous()
    conv.register_buffer("_cin_perm", perm)
    handle = conv.register_forward_pre_hook(_make_permute_pre_hook(perm))
    # store the handle in case the caller wants to remove it later
    conv._cin_perm_hook_handle = handle


def attach_permutations_from_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> int:
    """Round-trip helper: for every `*._cin_perm` key in `state_dict`, register
    the buffer on the matching Conv2d module AND install the forward pre-hook
    that permutes input. Call this BEFORE `model.load_state_dict(state_dict)`
    on a freshly-built model so the load_state_dict fills the buffers with the
    saved permutations.
    """
    count = 0
    for full_key in list(state_dict.keys()):
        if not full_key.endswith("._cin_perm"):
            continue
        mod_name = full_key[: -len("._cin_perm")]
        mod = model
        try:
            for part in mod_name.split("."):
                mod = getattr(mod, part)
        except AttributeError:
            continue
        if not isinstance(mod, nn.Conv2d):
            continue
        perm = state_dict[full_key].long()
        mod.register_buffer("_cin_perm", perm.clone())
        mod.register_forward_pre_hook(_make_permute_pre_hook(perm))
        count += 1
    return count


def permute_unfolded_h(
    h_unfolded: torch.Tensor,       # [N, Cin*kH*kW]
    perm: torch.LongTensor,         # [Cin]
    kH: int, kW: int,
) -> torch.Tensor:
    """Reorder the unfolded activation tensor to match the post-Cin-permutation
    weight layout. Same indexing scheme as torch.nn.functional.unfold:
        col_idx_orig = c * (kH*kW) + p
        col_idx_new  = c'* (kH*kW) + p     where c' = position s.t. perm[c']=c
    """
    Cin = perm.shape[0]
    L = kH * kW
    # For each new position c' (0..Cin-1) and kernel slot p (0..L-1), gather
    # from old position perm[c']*L + p.
    new_indices = (perm.unsqueeze(1) * L + torch.arange(L).unsqueeze(0)).flatten().to(h_unfolded.device)
    return h_unfolded.index_select(1, new_indices)


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

def is_eligible_conv(mod: nn.Module, *, min_in_channels: int = 4,
                     allow_grouped: bool = False,
                     only_1x1: bool = True,
                     require_in_div_4: bool = True) -> bool:
    """True iff `mod` is a Conv2d we can mask with 2-of-4 along the input axis.

    Default (only_1x1=True) restricts to 1×1 convs. This matches the theory
    cleanly: a 1×1 conv `[Cout, Cin, 1, 1]` acts as a per-location Linear map
    `y_{h,w} = W · x_{h,w}`, so the certificate-inspired CAST logic ports over
    without modification (same row-wise 2:4 grouping over input channels).

    For an appendix-style 3×3 extension, set only_1x1=False (uses the im2col
    reshape — the certificate cost is then computed on the unfolded input
    tensor, which is the per-spatial-location patch flattened into a single
    long vector).
    """
    if not isinstance(mod, nn.Conv2d):
        return False
    if mod.in_channels < min_in_channels:
        return False
    if not allow_grouped and mod.groups != 1:
        return False
    Cout, Cin, kH, kW = mod.weight.shape
    if only_1x1 and (kH, kW) != (1, 1):
        return False
    if require_in_div_4 and (Cin % 4 != 0):
        return False
    cols = Cin * kH * kW
    return cols >= 4


def list_eligible_convs(model: nn.Module, *, only_1x1: bool = True) -> list[tuple[str, nn.Conv2d]]:
    return [(n, m) for n, m in model.named_modules()
            if is_eligible_conv(m, only_1x1=only_1x1)]


# ---------------------------------------------------------------------------
# Mask helpers (used by the FT runner to freeze 2:4 entries during training)
# ---------------------------------------------------------------------------

def collect_nonzero_masks(model: nn.Module, *, include_linear: bool = True,
                           only_1x1: bool = True) -> dict[str, torch.Tensor]:
    """Snapshot every projected layer's nonzero pattern as a boolean mask.
    The FT loop must (a) zero `weight.grad` at masked positions and
    (b) re-apply the mask via in-place `weight.mul_(mask)` after each
    optimizer.step() to keep 2:4 legality across training."""
    masks: dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and is_eligible_conv(mod, only_1x1=only_1x1):
            masks[name] = mod.weight.data.detach().ne(0).to(mod.weight.device)
        elif include_linear and isinstance(mod, nn.Linear) and mod.weight.shape[0] == 1000:
            # Final classifier head, if it was projected
            masks[name] = mod.weight.data.detach().ne(0).to(mod.weight.device)
    return masks


def apply_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    """In-place: set masked-out weights back to zero. Call after optimizer.step()."""
    with torch.no_grad():
        for name, mod in model.named_modules():
            if name in masks:
                mod.weight.data.mul_(masks[name])


def freeze_grad_at_masked(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    """In-place: zero gradient at masked-out positions. Call AFTER backward(),
    BEFORE optimizer.step()."""
    for name, mod in model.named_modules():
        if name in masks and mod.weight.grad is not None:
            mod.weight.grad.mul_(masks[name])


def assert_2_4_legality(model: nn.Module, *, only_1x1: bool = True,
                         include_linear_head: bool = True) -> dict[str, dict]:
    """Verify every projected layer has exactly 2 NNZ in every 4-tuple.
    Raises AssertionError on violation. Returns per-layer summary on success."""
    out: dict[str, dict] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            if not is_eligible_conv(mod, only_1x1=only_1x1):
                continue
            W = mod.weight.detach()
            Cout, Cin, kH, kW = W.shape
            cols_full = Cin * kH * kW
        elif include_linear_head and isinstance(mod, nn.Linear) and mod.weight.shape[0] == 1000:
            W = mod.weight.detach()
            Cout, cols_full = W.shape
        else:
            continue
        cols_used = (cols_full // 4) * 4
        if cols_used == 0:
            continue
        W2 = W.reshape(Cout, cols_full)[:, :cols_used].reshape(Cout, cols_used // 4, 4)
        nnz_per_group = (W2 != 0).sum(dim=-1)
        bad = int((nnz_per_group != 2).sum().item())
        out[name] = {"groups": int(nnz_per_group.numel()), "bad_groups": bad}
        if bad != 0:
            raise AssertionError(
                f"2:4 legality broken at {name}: {bad}/{nnz_per_group.numel()} "
                f"groups have != 2 NNZ (e.g. one tuple has "
                f"{int(nnz_per_group.flatten()[0].item())} NNZ)"
            )
    return out


# ---------------------------------------------------------------------------
# Magnitude-based 2-of-4 (no calibration data needed)
# ---------------------------------------------------------------------------

def _apply_2_of_4_magnitude(W2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """W2d: [Cout, ncol] with ncol divisible by 4. Returns (W_masked, mask)."""
    Cout, ncol = W2d.shape
    n_groups = ncol // 4
    Wg = W2d.reshape(Cout, n_groups, 4)
    abs_g = Wg.abs()
    # top-2 indices per group
    top2_idx = abs_g.topk(2, dim=-1).indices                        # [Cout, n_groups, 2]
    mask = torch.zeros_like(Wg)
    mask.scatter_(-1, top2_idx, 1.0)
    Wm = (Wg * mask).reshape(Cout, ncol)
    return Wm, mask.reshape(Cout, ncol)


def magnitude_2_4_for_conv(
    model: nn.Module,
    dense_state_dict: dict[str, torch.Tensor] | None = None,
    *,
    free_restoration: bool = True,
    only_1x1: bool = True,
    log: bool = True,
) -> dict:
    """In-place: zero out 2-of-4 columns by magnitude in every eligible Conv2d.

    If `dense_state_dict` is provided AND `free_restoration=True`, the
    "free restoration" rule applies: any 4-tuple with fewer than 2 NNZ post-mask
    has the empty slot(s) refilled from the dense weight (preserves the 2-of-4
    pattern that the kernel will pay for anyway, so it's free FLOPs).

    Returns a stats dict matching the Linear-path `two_four_stats` schema:
      {
        "mode": "conv_magnitude_with_free_restoration",
        "layers": {layer_name: {...per-layer fields...}},
        "linear_params": ...,             # actually conv_params here
        "linear_nonzero_before": ...,
        "linear_nonzero_after": ...,
        ...
      }
    """
    layers_out: dict[str, dict] = {}
    total_params = 0
    total_nnz_before = 0
    total_nnz_after = 0
    total_groups = 0
    total_bad = 0
    restored = 0

    for name, mod in list_eligible_convs(model, only_1x1=only_1x1):
        W = mod.weight.data                                         # [Cout, Cin, kH, kW]
        Cout, Cin, kH, kW = W.shape
        cols_full = Cin * kH * kW
        cols_used = (cols_full // 4) * 4
        if cols_used == 0:
            continue
        W2d = W.reshape(Cout, cols_full).contiguous()
        nnz_before = int((W2d != 0).sum().item())

        # Project the divisible-by-4 prefix; leftover (cols_full - cols_used) tail
        # stays untouched. For all common ResNet shapes (Cin*kH*kW always divisible
        # by 4 when Cin>=4), there is no tail.
        Wleft = W2d[:, :cols_used].clone()
        Wm, mask2d = _apply_2_of_4_magnitude(Wleft)

        # Free restoration from dense
        if free_restoration and dense_state_dict is not None:
            dense_key = f"{name}.weight"
            if dense_key in dense_state_dict:
                Wdense = dense_state_dict[dense_key].reshape(Cout, cols_full)[:, :cols_used]
                # Find groups where post-mask NNZ < 2 (i.e. SER+magnitude killed too many)
                Wg_m = Wm.reshape(Cout, cols_used // 4, 4)
                mask_g = mask2d.reshape(Cout, cols_used // 4, 4)
                Wd_g = Wdense.reshape(Cout, cols_used // 4, 4)
                nnz_per_group = (Wg_m != 0).sum(dim=-1)            # [Cout, n_groups]
                short = nnz_per_group < 2
                if bool(short.any()):
                    # For each "short" group, restore the dense entry that minimizes
                    # |Wm - Wdense * (1-mask)|; greedily fill until we hit 2 NNZ.
                    # Simple proxy: restore the dense entry with the largest |Wd|
                    # NOT already kept, until pattern == 2-of-4.
                    abs_dense = Wd_g.abs() * (mask_g == 0).float()  # candidates
                    needed = (2 - nnz_per_group).clamp(min=0)       # how many to restore per group
                    # Restore up to 2 per group, sorted by candidate magnitude.
                    cand_sorted_idx = abs_dense.argsort(dim=-1, descending=True)
                    for k in range(2):                              # at most need 2 fills
                        # boolean: groups still short
                        still_short = needed > k
                        if not bool(still_short.any()):
                            break
                        slot = cand_sorted_idx.select(-1, k)        # [Cout, n_groups]
                        slot_unsq = slot.unsqueeze(-1)
                        # write the dense value into Wm at this slot, mark mask
                        new_vals = Wd_g.gather(-1, slot_unsq).squeeze(-1)
                        # only update where still_short
                        sel = still_short.unsqueeze(-1).expand(-1, -1, 4).clone()
                        sel.scatter_(-1, slot_unsq, still_short.unsqueeze(-1))
                        # write
                        Wg_m_flat = Wg_m.clone()
                        rows, cols, ks = torch.meshgrid(
                            torch.arange(Cout, device=Wm.device),
                            torch.arange(cols_used // 4, device=Wm.device),
                            torch.arange(1, device=Wm.device),
                            indexing="ij",
                        )
                        # Vectorized update with masks for speed.
                        update_mask = torch.zeros_like(Wg_m_flat, dtype=torch.bool)
                        update_mask.scatter_(-1, slot_unsq, still_short.unsqueeze(-1))
                        Wg_m_flat = torch.where(update_mask, Wd_g, Wg_m_flat)
                        Wg_m = Wg_m_flat
                        mask_g = (Wg_m != 0).float()
                    Wm = Wg_m.reshape(Cout, cols_used)
                    mask2d = mask_g.reshape(Cout, cols_used)
                    restored_layer = int((mask2d.sum() - mask2d.numel() / 2).item())
                    restored += max(0, restored_layer)

        W2d_new = W2d.clone()
        W2d_new[:, :cols_used] = Wm
        # Re-write back into the conv weight tensor
        mod.weight.data.copy_(W2d_new.reshape(Cout, Cin, kH, kW))

        # Per-group bad detection (groups with ≠ 2 NNZ after restoration; should be 0)
        Wg_final = mod.weight.data.reshape(Cout, cols_full)[:, :cols_used].reshape(Cout, cols_used // 4, 4)
        nnz_per_group_final = (Wg_final != 0).sum(dim=-1)
        bad_groups = int(((nnz_per_group_final != 2)).sum().item())

        nnz_after = int((mod.weight.data != 0).sum().item())
        params = W.numel()
        n_groups = (cols_used // 4) * Cout

        layers_out[name] = {
            "shape": list(W.shape),
            "in_channels": Cin,
            "kernel_size": [kH, kW],
            "params": params,
            "cols_used": cols_used,
            "cols_skipped_tail": cols_full - cols_used,
            "nonzero_before": nnz_before,
            "nonzero_after": nnz_after,
            "sparsity_before": 1 - nnz_before / params,
            "sparsity_after": 1 - nnz_after / params,
            "groups": n_groups,
            "bad_groups_after": bad_groups,
        }
        total_params += params
        total_nnz_before += nnz_before
        total_nnz_after += nnz_after
        total_groups += n_groups
        total_bad += bad_groups

        if log:
            print(f"  conv2_4 {name:>40s}  "
                  f"shape={tuple(W.shape)}  "
                  f"sparsity {layers_out[name]['sparsity_before']:.2f}->{layers_out[name]['sparsity_after']:.2f}  "
                  f"bad_groups={bad_groups}")

    stats = {
        "mode": "conv_magnitude_with_free_restoration" if free_restoration else "conv_magnitude_only",
        "include_head": False,
        "layers": layers_out,
        "linear_params": total_params,           # sum across Conv2d layers
        "linear_nonzero_before": total_nnz_before,
        "linear_nonzero_after": total_nnz_after,
        "groups": total_groups,
        "groups_with_more_than_2_nonzero_after": total_bad,
        "linear_sparsity_before": (1 - total_nnz_before / max(total_params, 1)),
        "linear_sparsity_after": (1 - total_nnz_after / max(total_params, 1)),
        "n_layers_modified": len(layers_out),
        "free_restoration_count": restored,
    }
    return stats


# ---------------------------------------------------------------------------
# Certificate-aware 2-of-4 (uses calibration data)
# ---------------------------------------------------------------------------

def _capture_unfolded_inputs(
    model: nn.Module,
    calib_loader: Iterable,
    layers: list[tuple[str, nn.Conv2d]],
    *,
    n_calib_imgs: int = 256,
    device: str = "cuda",
    max_rows_per_layer: int = 4096,
) -> dict[str, torch.Tensor]:
    """Forward `n_calib_imgs` total through the model, collecting the unfolded input
    tensor for each named Conv2d in `layers`. Returns name→tensor of shape
    [N_rows ≤ max_rows_per_layer, Cin*kH*kW].

    Memory-safe: capture is downsampled INSIDE the hook (per-layer cap of
    `max_rows_per_layer`) so we never accumulate the full unfolded tensor in
    CPU RAM. With `--include-3x3-convs` the unfolded tensor for early layers
    is huge (e.g. 1×1 with B=64, H=W=56 produces 64*3136 = 200K rows per
    forward pass — for 256 calib images that's 800K rows × 64 floats * 4 B =
    ≈200 MB *per layer*; aggregated over 50+ layers this can blow up to 10s
    of GB without per-layer capping).
    """
    captures: dict[str, list[torch.Tensor]] = {n: [] for n, _ in layers}
    handles = []

    def make_hook(name: str, conv: nn.Conv2d):
        def _hook(_mod, inputs, _outputs):
            already = sum(t.shape[0] for t in captures[name])
            remaining = max_rows_per_layer - already
            if remaining <= 0:
                return
            x = inputs[0].detach()
            uf = F.unfold(
                x,
                kernel_size=conv.kernel_size,
                dilation=conv.dilation,
                padding=conv.padding,
                stride=conv.stride,
            )                                                         # [B, Cin*kH*kW, L]
            # Reshape to [B*L, Cin*kH*kW]
            uf = uf.permute(0, 2, 1).contiguous().view(-1, uf.shape[1])
            # Downsample INSIDE the hook so we never carry a giant tensor.
            if uf.shape[0] > remaining:
                idx = torch.randperm(uf.shape[0], device=uf.device)[:remaining]
                uf = uf.index_select(0, idx)
            captures[name].append(uf.to("cpu"))
        return _hook

    for n, c in layers:
        handles.append(c.register_forward_hook(make_hook(n, c)))

    try:
        model.eval()
        seen = 0
        with torch.no_grad():
            for batch in calib_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device)
                model(x)
                seen += x.shape[0]
                if seen >= n_calib_imgs:
                    break
    finally:
        for h in handles:
            h.remove()

    out: dict[str, torch.Tensor] = {}
    for n, parts in captures.items():
        out[n] = torch.cat(parts, dim=0) if parts else torch.zeros(0)
    return out


def _cert_cost_2_of_4(W_group: torch.Tensor, h_group: torch.Tensor) -> torch.Tensor:
    """For each 4-tuple, enumerate the 6 possible 2-of-4 masks and return the
    EXACT covariance-form certificate cost
        c_g(m) = E_x[ ‖((1-m)·W_g) · h_g(x)‖² ]
                = r^T · C_g · r            where r = (1-m) ⊙ W_g
                                          and C_g = E_x[h_g h_g^T]  (4×4 per group)

    This includes the cross-terms 2·E[W_i·h_i · W_j·h_j] that the per-slot
    diagonal approximation drops — important when h channels are correlated
    (e.g. spatially adjacent unfolded patches).

    Memory: O(G·16 + O·G·6) instead of O(N·O·G·4), which avoids OOM on large
    bottleneck layers.

    W_group: [Cout, G, 4]
    h_group: [N,    G, 4]
    Returns: [Cout, G, 6] costs (lower = better).
    """
    Cout, G, _ = W_group.shape
    # Per-group 4x4 covariance averaged over N calibration samples
    # einsum: for each group g, sum over n of h[n,g,i] * h[n,g,j] / N
    h_f = h_group.float()
    C = torch.einsum("ngi,ngj->gij", h_f, h_f) / max(h_group.shape[0], 1)   # [G, 4, 4]
    # 6 binary "keep" patterns; "drop" = 1 - keep
    keep_patterns = torch.tensor(
        [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1],
         [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]],
        dtype=W_group.dtype, device=W_group.device,
    )                                                                       # [6, 4]
    drop = 1.0 - keep_patterns                                              # [6, 4]
    W_f = W_group.float()
    out = torch.empty(Cout, G, 6, device=W_group.device, dtype=W_group.dtype)
    # For each pattern p: r = drop[p] * W -> [O, G, 4]; cost = r·C·r
    for p in range(6):
        r = drop[p].view(1, 1, 4) * W_f                                     # [O, G, 4]
        # temp[o,g,j] = sum_k r[o,g,k] * C[g,j,k]
        temp = torch.einsum("ogk,gjk->ogj", r, C)                            # [O, G, 4]
        out[..., p] = (temp * r).sum(dim=-1).to(W_group.dtype)
    return out


def _capture_kd_grad_W(
    model: nn.Module,
    teacher: nn.Module,
    calib_loader: Iterable,
    layers: list[tuple[str, nn.Conv2d]],
    *,
    n_calib_imgs: int = 64,
    distill_temp: float = 2.0,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """One forward+backward pass on calibration set with KD loss; returns
    `grad_W` per eligible Conv2d layer (Fisher-saliency input for the
    alpha-KD term in the cert cost).

    KD loss: T^2 * KL(softmax(s/T), softmax(t/T)).  Backward populates
    `mod.weight.grad` for every trainable conv. We snapshot it, then zero out.
    """
    model.zero_grad(set_to_none=True)
    # Make every projected layer's weight require grad just for this pass.
    saved_req = {}
    for name, mod in layers:
        saved_req[name] = mod.weight.requires_grad
        mod.weight.requires_grad_(True)

    teacher.eval()
    model.eval()  # forward through student in eval mode is fine for KD-loss
    seen = 0
    n_backward = 0
    with torch.enable_grad():
        for batch in calib_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            with torch.no_grad():
                t_logits = teacher(x).detach()
            s_logits = model(x)
            T = float(distill_temp)
            kd = F.kl_div(
                F.log_softmax(s_logits / T, dim=1),
                F.softmax(t_logits / T, dim=1),
                reduction="batchmean",
            ) * (T * T)
            kd.backward()
            seen += x.shape[0]
            n_backward += 1
            if seen >= n_calib_imgs:
                break

    grad_W: dict[str, torch.Tensor] = {}
    for name, mod in layers:
        if mod.weight.grad is not None:
            # `reduction="batchmean"` normalizes each batch, but gradients from
            # multiple calibration batches accumulate. Average them so alpha_kd
            # has the same scale for n_calib=64 and n_calib=256 sweeps.
            grad_W[name] = (mod.weight.grad.detach() / max(n_backward, 1)).clone()
        mod.weight.requires_grad_(saved_req[name])

    model.zero_grad(set_to_none=True)
    return grad_W


def _cert_cost_2_of_4_linf(
    W_group: torch.Tensor,           # [Cout, G, 4]
    h_group: torch.Tensor,           # [N, G, 4]
) -> torch.Tensor:
    """Section-5 ℓ_∞ form of the cert cost:
        c_g(m) = max_n |((1-m) ⊙ W_g) · h_g(n)|^2          (per output row)

    This implements the `B_T(R) = (2/|T|) Σ_s L_s ‖R ψ_1(s)‖_∞` quantity from
    Section 5. The covariance-form `_cert_cost_2_of_4` remains the default for
    the main CAST projection; this function is used when literal ℓ_∞ rescoring
    is requested.

    Returns [Cout, G, 6] (lower = better).
    """
    Cout, G, _ = W_group.shape
    N = h_group.shape[0]
    keep_patterns = torch.tensor(
        [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1],
         [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]],
        dtype=W_group.dtype, device=W_group.device,
    )
    drop = 1.0 - keep_patterns                                              # [6, 4]
    W_f = W_group.float()
    h_f = h_group.float()
    out = torch.empty(Cout, G, 6, device=W_group.device, dtype=W_group.dtype)
    for p in range(6):
        r = drop[p].view(1, 1, 4) * W_f                                     # [O, G, 4]
        # response[n, o, g] = sum_i r[o,g,i] * h[n,g,i]
        resp = torch.einsum("ogi,ngi->nog", r, h_f)                          # [N, O, G]
        # ℓ_∞ over calibration samples n; squared so it composes with ℓ² scale
        max_resp = resp.abs().max(dim=0).values                              # [O, G]
        out[..., p] = (max_resp ** 2).to(W_group.dtype)
    return out


def _ser_hamming_prior(
    ser_kept_group: torch.Tensor,    # [Cout, G, 4] bool
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """**Section-5 weighted-elastic-net prior**: Hamming distance per (o, g)
    between the SER source mask and each candidate 2-of-4 keep pattern.

    Lower = more aligned with the SER source. Adding this with a positive
    coefficient α to the cert cost biases the argmin toward the source mask,
    matching Section 5's weighted elastic-net form: higher penalty for
    SER-removable entries and lower penalty for SER-protected entries.

    Returns [Cout, G, 6] (lower = better).
    """
    keep_patterns = torch.tensor(
        [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1],
         [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]],
        dtype=torch.float32, device=device,
    )                                                                       # [6, 4]
    sk = ser_kept_group.float().to(device).unsqueeze(2)                     # [O, G, 1, 4]
    kp = keep_patterns.view(1, 1, 6, 4)                                     # [1, 1, 6, 4]
    hamming = (sk - kp).abs().sum(dim=-1)                                   # [O, G, 6]
    return hamming


def _fisher_2_of_4(
    W_group: torch.Tensor,           # [Cout, G, 4]
    g_W_group: torch.Tensor,         # [Cout, G, 4]
) -> torch.Tensor:
    """Fisher / SquareGradient saliency aggregated over the 6 candidate
    2-of-4 keep patterns: for each pattern, sum the saliency `(g·W)^2` of
    the slots that pattern would DROP. Lower = better.
    Returns [Cout, G, 6].
    """
    keep_patterns = torch.tensor(
        [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1],
         [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]],
        dtype=W_group.dtype, device=W_group.device,
    )
    drop_patterns = 1.0 - keep_patterns                                       # [6, 4]
    saliency = (g_W_group.float() * W_group.float()).pow(2)                  # [O, G, 4]
    out = torch.einsum("ogc,pc->ogp", saliency, drop_patterns)               # [O, G, 6]
    return out.to(W_group.dtype)


def cert_aware_2_4_for_conv(
    model: nn.Module,
    calib_loader: Iterable,
    dense_state_dict: dict[str, torch.Tensor] | None = None,
    *,
    n_calib_imgs: int = 256,
    device: str = "cuda",
    free_restoration: bool = True,
    only_1x1: bool = True,
    permute_align: bool = False,    # see PERMUTATION_ALIGN_DESIGN.md
    alpha_ser_prior: float = 0.0,   # RMT/SER Hamming prior toward source mask
    ser_prior_layer_scale: bool = True,
    alpha_kd: float = 0.0,          # KD Fisher term added to the cert cost
    teacher_for_kd: "nn.Module | None" = None,
    distill_temp: float = 2.0,
    cost_form: str = "l2",          # "l2" (default, covariance) or "linf" (Section-5 literal)
    log: bool = True,
) -> dict:
    """In-place: cert-aware 2-of-4 projection on every eligible Conv2d.

    For each 4-tuple, evaluate all 6 possible 2-of-4 masks by the EXACT
    covariance-form certificate cost
        c_g(m) = E_x[ ‖((1-m)·W̃_g) · h_g(x)‖² ]
    where W̃_g = where(SER kept it, W_sparse, W_dense) — i.e. free restoration
    is part of the OPTIMIZATION OBJECTIVE, not a post-hoc fill. The kept slots
    in the chosen pattern get the W_dense value if SER had zeroed them and
    the W_sparse value otherwise.

    Memory-safe (covariance form, ~16·G floats per layer instead of N·O·G·4).

    Returns the same stats schema as `magnitude_2_4_for_conv`.
    """
    layers = list_eligible_convs(model, only_1x1=only_1x1)
    if log:
        print(f"  cert_aware_2_4_for_conv: capturing activations on "
              f"{n_calib_imgs} calibration images for {len(layers)} conv layers "
              f"(only_1x1={only_1x1})")
    captures = _capture_unfolded_inputs(
        model, calib_loader, layers, n_calib_imgs=n_calib_imgs, device=device,
    )

    # Optional alpha-KD Fisher saliency capture (one fwd+bwd on calib).
    grad_W_per_layer: dict[str, torch.Tensor] = {}
    if alpha_kd > 0 and teacher_for_kd is not None:
        if log:
            print(f"  alpha-KD term: capturing grad_W via KD fwd+bwd on calib "
                  f"(alpha_kd={alpha_kd}, T={distill_temp})")
        grad_W_per_layer = _capture_kd_grad_W(
            model, teacher_for_kd, calib_loader, layers,
            n_calib_imgs=n_calib_imgs, distill_temp=distill_temp, device=device,
        )

    keep_patterns = torch.tensor(
        [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1],
         [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]],
        dtype=torch.float32,
    )                                                                       # [6, 4]

    layers_out: dict[str, dict] = {}
    total_params = total_before = total_after = total_groups = total_bad = 0
    permuted_layer_count = 0
    if alpha_ser_prior > 0 and ser_prior_layer_scale:
        densities = []
        for _, m in layers:
            if m.weight.numel() > 0:
                densities.append(float(m.weight.detach().ne(0).float().mean().item()))
        global_ser_density = sum(densities) / max(len(densities), 1)
    else:
        global_ser_density = 1.0

    for name, mod in layers:
        W = mod.weight.data
        Cout, Cin, kH, kW = W.shape
        cols_full = Cin * kH * kW
        cols_used = (cols_full // 4) * 4
        if cols_used == 0:
            continue
        nnz_before = int((W != 0).sum().item())   # snapshot BEFORE we overwrite

        h_all_raw = captures.get(name, torch.zeros(0))

        # ---------- Optional Cin permutation alignment ----------
        # Compute permutation from importance, apply to conv (in-place + hook),
        # AND apply the equivalent reorder to captured activations so the cert
        # cost search runs in the post-permutation basis.
        layer_permuted = False
        if permute_align and Cin >= 4 and h_all_raw.numel() > 0:
            perm = compute_cin_permutation(W, h_all_raw)
            apply_cin_permutation_to_conv(mod, perm)
            # Re-fetch the now-permuted weight + reorder captured activations
            W = mod.weight.data
            h_all_raw = permute_unfolded_h(h_all_raw, perm, kH, kW)
            layer_permuted = True
            permuted_layer_count += 1

        W2d = W.reshape(Cout, cols_full).contiguous()

        # Build slot_values: SER-kept entries take W_sparse, SER-zeroed entries
        # take W_dense (the candidate "free restoration" values). The pattern
        # search then picks the 2-of-4 pattern that minimizes cert cost when
        # those slot_values are the ones used.
        Wleft_sparse = W2d[:, :cols_used]
        ser_kept = (Wleft_sparse != 0)
        if free_restoration and dense_state_dict is not None and f"{name}.weight" in dense_state_dict:
            Wdense_full = dense_state_dict[f"{name}.weight"]
            # If we permuted Cin, the dense weight must be permuted to match before
            # extracting slot candidates.
            if layer_permuted:
                Wdense_full = Wdense_full[:, perm.cpu(), :, :].to(W.device)
            else:
                Wdense_full = Wdense_full.to(W.device)
            Wdense2d = Wdense_full.reshape(Cout, cols_full)[:, :cols_used]
            slot_values = torch.where(ser_kept, Wleft_sparse, Wdense2d)
        else:
            slot_values = Wleft_sparse.clone()

        h_all = h_all_raw
        if h_all.numel() == 0:
            print(f"  WARN no activations captured for {name}; falling back to magnitude on slot_values")
            Wm, mask2d = _apply_2_of_4_magnitude(slot_values)
        else:
            # Downsample if huge, then move to device
            if h_all.shape[0] > 4096:
                idx = torch.randperm(h_all.shape[0])[:4096]
                h_all = h_all[idx]
            h_all = h_all[:, :cols_used].to(W.device)
            n_groups = cols_used // 4
            Wg = slot_values.reshape(Cout, n_groups, 4)
            hg = h_all.reshape(-1, n_groups, 4)
            if cost_form == "linf":
                costs = _cert_cost_2_of_4_linf(Wg, hg)                       # Section-5 literal
            else:
                costs = _cert_cost_2_of_4(Wg, hg)                            # [O, G, 6]
            # Optional RMT/SER prior. The source SER mask was produced by the
            # sigma_+-guided Hybrid Magnitude--SER pipeline, so a Hamming prior
            # toward that mask is the local 2:4 analogue of SER restoration.
            if alpha_ser_prior > 0:
                ser_g = ser_kept.reshape(Cout, n_groups, 4).float()
                kp = keep_patterns.to(W.device).view(1, 1, 6, 4)
                hamming = (kp - ser_g.unsqueeze(2)).abs().sum(dim=-1)          # [O, G, 6]
                scale = costs.detach().mean(dim=-1, keepdim=True).clamp_min(1e-12)
                if ser_prior_layer_scale:
                    layer_density = float(ser_kept.float().mean().item())
                    density_scale = layer_density / max(global_ser_density, 1e-12)
                    density_scale = max(0.25, min(4.0, density_scale))
                else:
                    density_scale = 1.0
                costs = costs + (alpha_ser_prior * density_scale) * scale * hamming
            # Optional alpha-KD Fisher term added to cert cost.
            # The Fisher saliency `(g·W)^2` is computed in the ORIGINAL
            # (un-permuted) Cin order, so we must apply the same Cin
            # permutation to grad_W as we did to W.
            if alpha_kd > 0 and name in grad_W_per_layer:
                gW = grad_W_per_layer[name]
                if layer_permuted:
                    gW = gW[:, perm.cpu(), :, :].to(W.device)
                gW2d = gW.reshape(Cout, cols_full)[:, :cols_used]
                gWg = gW2d.reshape(Cout, n_groups, 4)
                fisher_cost = _fisher_2_of_4(Wg, gWg)                         # [O, G, 6]
                costs = costs + alpha_kd * fisher_cost.to(costs.dtype)
            best = costs.argmin(dim=-1)                                       # [O, G]
            mask_g_keep = keep_patterns.to(W.device)[best]                    # [O, G, 4]
            Wm = (Wg * mask_g_keep).reshape(Cout, cols_used)
            mask2d = mask_g_keep.reshape(Cout, cols_used)

        # Write back into the conv weight tensor
        W2d_new = W2d.clone()
        W2d_new[:, :cols_used] = Wm
        mod.weight.data.copy_(W2d_new.reshape(Cout, Cin, kH, kW))

        Wg_final = mod.weight.data.reshape(Cout, cols_full)[:, :cols_used].reshape(Cout, cols_used // 4, 4)
        bad_groups = int((((Wg_final != 0).sum(dim=-1)) != 2).sum().item())

        # nnz_before was snapshot BEFORE the weight was overwritten (above).
        nnz_after = int((mod.weight.data != 0).sum().item())
        params = W.numel()
        n_groups = (cols_used // 4) * Cout
        layers_out[name] = {
            "shape": list(W.shape),
            "in_channels": Cin,
            "kernel_size": [kH, kW],
            "params": params,
            "cols_used": cols_used,
            "cols_skipped_tail": cols_full - cols_used,
            "nonzero_before": nnz_before,
            "nonzero_after": nnz_after,
            "sparsity_before": 1 - nnz_before / params,
            "sparsity_after": 1 - nnz_after / params,
            "groups": n_groups,
            "bad_groups_after": bad_groups,
            "cin_permuted": layer_permuted,
        }
        total_params += params
        total_before += nnz_before
        total_after += nnz_after
        total_groups += n_groups
        total_bad += bad_groups
        if log:
            tag = "+perm" if layer_permuted else ""
            print(f"  conv2_4_cert{tag:5s} {name:>36s}  shape={tuple(W.shape)}  "
                  f"sparsity {layers_out[name]['sparsity_before']:.2f}->{layers_out[name]['sparsity_after']:.2f}  "
                  f"bad={bad_groups}")

    return {
        "mode": "conv_cert_aware_with_free_restoration" + ("_permuted" if permute_align else "") if free_restoration else "conv_cert_aware_only",
        "include_head": False,
        "layers": layers_out,
        "linear_params": total_params,
        "linear_nonzero_before": total_before,
        "linear_nonzero_after": total_after,
        "groups": total_groups,
        "groups_with_more_than_2_nonzero_after": total_bad,
        "linear_sparsity_before": (1 - total_before / max(total_params, 1)),
        "linear_sparsity_after": (1 - total_after / max(total_params, 1)),
        "n_layers_modified": len(layers_out),
        "permuted_layer_count": permuted_layer_count,
        "permute_align_enabled": bool(permute_align),
        "alpha_ser_prior": float(alpha_ser_prior),
        "ser_prior_layer_scale": bool(ser_prior_layer_scale),
        "alpha_kd": float(alpha_kd),
        "alpha_kd_layers_with_grad": len(grad_W_per_layer),
    }


# ---------------------------------------------------------------------------
# Linear-side cert-aware 2-of-4 — same logic as cert_aware_2_4_for_conv but for
# nn.Linear weights. ViT/DeiT/MLP architectures use Linear layers, so we need
# this to apply the 5-method knobs (perm, alpha_kd, cost_form, alpha_ser_prior)
# to ViT-B/16 etc. for the pre-FT ablation.
# ---------------------------------------------------------------------------


def is_eligible_linear(name: str, mod: nn.Module, *, min_in_features: int = 4,
                       require_in_div_4: bool = True, skip_classifier: bool = True) -> bool:
    if not isinstance(mod, nn.Linear):
        return False
    if mod.in_features < min_in_features:
        return False
    if require_in_div_4 and mod.in_features % 4 != 0:
        return False
    if skip_classifier and mod.weight.shape[0] == 1000:
        # Skip the classifier head (head MACs are ~0.1% of eligible MACs).
        return False
    return True


def list_eligible_linears(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [(n, m) for n, m in model.named_modules() if is_eligible_linear(n, m)]


def _capture_linear_inputs(
    model: nn.Module,
    calib_loader: Iterable,
    layers: list[tuple[str, nn.Linear]],
    *,
    n_calib_imgs: int = 256,
    device: str = "cuda",
    max_rows_per_layer: int = 4096,
) -> dict[str, torch.Tensor]:
    """Capture pre-Linear input activations for each named Linear in `layers`.
    Per-layer downsample inside the hook (memory-safe). For ViT, the Linear
    sees a flattened [B*tokens, in_features] view; this function uses that view
    directly.
    """
    captures: dict[str, list[torch.Tensor]] = {n: [] for n, _ in layers}
    handles = []

    def make_hook(name: str):
        def _hook(_mod, inputs, _outputs):
            already = sum(t.shape[0] for t in captures[name])
            remaining = max_rows_per_layer - already
            if remaining <= 0:
                return
            x = inputs[0].detach()
            # Flatten any leading dims, keep last (in_features)
            x = x.reshape(-1, x.shape[-1])
            if x.shape[0] > remaining:
                idx = torch.randperm(x.shape[0], device=x.device)[:remaining]
                x = x.index_select(0, idx)
            captures[name].append(x.to("cpu"))
        return _hook

    for n, m in layers:
        handles.append(m.register_forward_hook(make_hook(n)))

    try:
        model.eval()
        seen = 0
        with torch.no_grad():
            for batch in calib_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device)
                model(x)
                seen += x.shape[0]
                if seen >= n_calib_imgs:
                    break
    finally:
        for h in handles:
            h.remove()

    out: dict[str, torch.Tensor] = {}
    for n, parts in captures.items():
        out[n] = torch.cat(parts, dim=0) if parts else torch.zeros(0)
    return out


def cert_aware_2_4_for_linear(
    model: nn.Module,
    calib_loader: Iterable,
    dense_state_dict: dict[str, torch.Tensor] | None = None,
    *,
    n_calib_imgs: int = 256,
    device: str = "cuda",
    free_restoration: bool = True,
    permute_align: bool = False,
    alpha_kd: float = 0.0,
    teacher_for_kd: "nn.Module | None" = None,
    distill_temp: float = 2.0,
    cost_form: str = "l2",
    alpha_ser_prior: float = 0.0,
    ser_prior_layer_scale: bool = True,
    log: bool = True,
) -> dict:
    """Same algorithm as cert_aware_2_4_for_conv but for nn.Linear modules.
    Treats each Linear weight `[out, in]` as the unfolded equivalent of a
    1×1 conv with `[out, in, 1, 1]`, so 2:4 partitions over `in` directly.

    Supported knobs (all 5 method axes from the experiment):
      - permute_align: importance-balanced Cin permutation alignment
      - alpha_kd: Fisher KD saliency term
      - cost_form: "l2" (covariance) or "linf" (Section-5 literal)
      - alpha_ser_prior: Hamming prior toward SER mask (Section-5 weighted-elastic-net)
      - free_restoration: dense fill-in when SER kept <2 NNZ in a 4-tuple
    """
    layers = list_eligible_linears(model)
    if log:
        print(f"  cert_aware_2_4_for_linear: capturing inputs on {n_calib_imgs} "
              f"images for {len(layers)} Linear layers (cost={cost_form}, "
              f"perm={permute_align}, α_kd={alpha_kd}, α_ser={alpha_ser_prior})")

    captures = _capture_linear_inputs(
        model, calib_loader, layers, n_calib_imgs=n_calib_imgs, device=device,
    )

    grad_W_per_layer: dict[str, torch.Tensor] = {}
    if alpha_kd > 0 and teacher_for_kd is not None:
        if log:
            print(f"  alpha-KD: capturing grad_W via KD fwd+bwd")
        # _capture_kd_grad_W only uses .weight.requires_grad and .weight.grad,
        # so the same helper works for Linear layers.
        grad_W_per_layer = _capture_kd_grad_W(
            model, teacher_for_kd, calib_loader, layers,
            n_calib_imgs=n_calib_imgs, distill_temp=distill_temp, device=device,
        )

    keep_patterns_t = torch.tensor(
        [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1],
         [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]],
        dtype=torch.float32,
    )

    layers_out: dict[str, dict] = {}
    total_params = total_before = total_after = total_groups = total_bad = 0
    permuted_layer_count = 0

    if alpha_ser_prior > 0 and ser_prior_layer_scale:
        densities = []
        for _, m in layers:
            if m.weight.numel() > 0:
                densities.append(float(m.weight.detach().ne(0).float().mean().item()))
        global_ser_density = sum(densities) / max(len(densities), 1)
    else:
        global_ser_density = 1.0

    for name, mod in layers:
        W = mod.weight.data                                                 # [out, in]
        out_f, in_f = W.shape
        cols_used = (in_f // 4) * 4
        if cols_used == 0:
            continue
        nnz_before = int((W != 0).sum().item())
        h_all_raw = captures.get(name, torch.zeros(0))

        # Permutation alignment (Linear analog: just permute in_f)
        layer_permuted = False
        if permute_align and in_f >= 4 and h_all_raw.numel() > 0:
            # Build importance: ‖W[:, c]‖^2 · E[h_c^2]
            h_per_col = h_all_raw.pow(2).mean(dim=0)                        # [in_f]
            w_per_col = W.detach().pow(2).sum(dim=0)                         # [in_f]
            importance = (h_per_col.to(w_per_col.device) * w_per_col).cpu()
            sorted_idx = importance.argsort(descending=True)
            n_groups_perm = in_f // 4
            perm = torch.empty(in_f, dtype=torch.long)
            for k in range(4):
                for g in range(n_groups_perm):
                    perm[g * 4 + k] = sorted_idx[k * n_groups_perm + g].item()
            tail = in_f - n_groups_perm * 4
            if tail > 0:
                used = set(perm[: n_groups_perm * 4].tolist())
                unused = [c for c in range(in_f) if c not in used]
                for i, c in enumerate(unused):
                    perm[n_groups_perm * 4 + i] = c
            perm = perm.to(device)
            with torch.no_grad():
                mod.weight.data = mod.weight.data[:, perm].contiguous()
            mod.register_buffer("_cin_perm", perm)
            # forward pre-hook to permute input at runtime
            def _hook(_module, inputs):
                x = inputs[0]
                p = _module._cin_perm.to(x.device)
                return (x.index_select(-1, p),)
            mod.register_forward_pre_hook(_hook)
            # reorder captured inputs to match
            h_all_raw = h_all_raw.index_select(1, perm.cpu())
            W = mod.weight.data
            layer_permuted = True
            permuted_layer_count += 1

        Wleft_sparse = W[:, :cols_used]
        ser_kept = (Wleft_sparse != 0)
        if free_restoration and dense_state_dict is not None and f"{name}.weight" in dense_state_dict:
            Wdense_full = dense_state_dict[f"{name}.weight"]
            if layer_permuted:
                Wdense_full = Wdense_full[:, perm.cpu()].to(W.device)
            else:
                Wdense_full = Wdense_full.to(W.device)
            Wdense2d = Wdense_full[:, :cols_used]
            slot_values = torch.where(ser_kept, Wleft_sparse, Wdense2d)
        else:
            slot_values = Wleft_sparse.clone()

        h_all = h_all_raw
        if h_all.numel() == 0:
            print(f"  WARN no captures for {name}; falling back to magnitude")
            Wm, mask2d = _apply_2_of_4_magnitude(slot_values)
        else:
            if h_all.shape[0] > 4096:
                idx = torch.randperm(h_all.shape[0])[:4096]
                h_all = h_all[idx]
            h_all = h_all[:, :cols_used].to(W.device)
            n_groups = cols_used // 4
            Wg = slot_values.reshape(out_f, n_groups, 4)
            hg = h_all.reshape(-1, n_groups, 4)
            if cost_form == "linf":
                costs = _cert_cost_2_of_4_linf(Wg, hg)
            else:
                costs = _cert_cost_2_of_4(Wg, hg)
            if alpha_ser_prior > 0:
                ser_g = ser_kept.reshape(out_f, n_groups, 4).float()
                kp = keep_patterns_t.to(W.device).view(1, 1, 6, 4)
                hamming = (kp - ser_g.unsqueeze(2)).abs().sum(dim=-1)
                scale = costs.detach().mean(dim=-1, keepdim=True).clamp_min(1e-12)
                if ser_prior_layer_scale:
                    layer_density = float(ser_kept.float().mean().item())
                    density_scale = layer_density / max(global_ser_density, 1e-12)
                    density_scale = max(0.25, min(4.0, density_scale))
                else:
                    density_scale = 1.0
                costs = costs + (alpha_ser_prior * density_scale) * scale * hamming
            if alpha_kd > 0 and name in grad_W_per_layer:
                gW = grad_W_per_layer[name]
                if layer_permuted:
                    gW = gW[:, perm.cpu()].to(W.device)
                gW2d = gW[:, :cols_used]
                gWg = gW2d.reshape(out_f, n_groups, 4)
                fisher_cost = _fisher_2_of_4(Wg, gWg)
                costs = costs + alpha_kd * fisher_cost.to(costs.dtype)
            best = costs.argmin(dim=-1)
            mask_g_keep = keep_patterns_t.to(W.device)[best]
            Wm = (Wg * mask_g_keep).reshape(out_f, cols_used)
            mask2d = mask_g_keep.reshape(out_f, cols_used)

        W_new = W.clone()
        W_new[:, :cols_used] = Wm
        mod.weight.data.copy_(W_new)

        Wg_final = mod.weight.data[:, :cols_used].reshape(out_f, cols_used // 4, 4)
        bad_groups = int((((Wg_final != 0).sum(dim=-1)) != 2).sum().item())
        nnz_after = int((mod.weight.data != 0).sum().item())
        params = W.numel()
        n_groups_total = (cols_used // 4) * out_f
        layers_out[name] = {
            "shape": list(W.shape), "in_features": in_f,
            "params": params,
            "cols_used": cols_used, "cols_skipped_tail": in_f - cols_used,
            "nonzero_before": nnz_before, "nonzero_after": nnz_after,
            "sparsity_before": 1 - nnz_before / params,
            "sparsity_after": 1 - nnz_after / params,
            "groups": n_groups_total, "bad_groups_after": bad_groups,
            "cin_permuted": layer_permuted,
        }
        total_params += params
        total_before += nnz_before
        total_after += nnz_after
        total_groups += n_groups_total
        total_bad += bad_groups
        if log:
            tag = "+perm" if layer_permuted else ""
            print(f"  lin2_4_cert{tag:5s} {name:>40s}  shape={tuple(W.shape)}  "
                  f"sp {layers_out[name]['sparsity_before']:.2f}->{layers_out[name]['sparsity_after']:.2f}  "
                  f"bad={bad_groups}")

    return {
        "mode": f"linear_cert_aware{'+perm' if permute_align else ''}{'+linf' if cost_form=='linf' else ''}{'+ser' if alpha_ser_prior>0 else ''}{'+kd' if alpha_kd>0 else ''}",
        "include_head": False,
        "layers": layers_out,
        "linear_params": total_params,
        "linear_nonzero_before": total_before,
        "linear_nonzero_after": total_after,
        "groups": total_groups,
        "groups_with_more_than_2_nonzero_after": total_bad,
        "linear_sparsity_before": (1 - total_before / max(total_params, 1)),
        "linear_sparsity_after": (1 - total_after / max(total_params, 1)),
        "n_layers_modified": len(layers_out),
        "permuted_layer_count": permuted_layer_count,
        "permute_align_enabled": bool(permute_align),
        "alpha_kd": float(alpha_kd),
        "alpha_kd_layers_with_grad": len(grad_W_per_layer),
        "alpha_ser_prior": float(alpha_ser_prior),
        "cost_form": cost_form,
    }


# ---------------------------------------------------------------------------
# Convenience: extend the existing Linear stats with conv stats
# ---------------------------------------------------------------------------

def merge_linear_and_conv_stats(linear_stats: dict, conv_stats: dict) -> dict:
    """Combine Linear-path and Conv-path two_four_stats into one report."""
    merged_layers = {}
    for src in (linear_stats.get("layers", {}), conv_stats.get("layers", {})):
        merged_layers.update(src)
    p = (linear_stats.get("linear_params", 0) + conv_stats.get("linear_params", 0))
    b = (linear_stats.get("linear_nonzero_before", 0) + conv_stats.get("linear_nonzero_before", 0))
    a = (linear_stats.get("linear_nonzero_after", 0) + conv_stats.get("linear_nonzero_after", 0))
    g = (linear_stats.get("groups", 0) + conv_stats.get("groups", 0))
    bad = (linear_stats.get("groups_with_more_than_2_nonzero_after", 0)
           + conv_stats.get("groups_with_more_than_2_nonzero_after", 0))
    return {
        "mode": f"merged({linear_stats.get('mode','?')},{conv_stats.get('mode','?')})",
        "include_head": True,
        "layers": merged_layers,
        "linear_params": p,
        "linear_nonzero_before": b,
        "linear_nonzero_after": a,
        "groups": g,
        "groups_with_more_than_2_nonzero_after": bad,
        "linear_sparsity_before": (1 - b / max(p, 1)),
        "linear_sparsity_after": (1 - a / max(p, 1)),
        # Preserve per-side metadata that downstream manifest writers rely on.
        "n_layers_modified": (
            linear_stats.get("n_layers_modified", 0)
            + conv_stats.get("n_layers_modified", 0)
        ),
        "permuted_layer_count": conv_stats.get("permuted_layer_count", 0),
        "permute_align_enabled": conv_stats.get("permute_align_enabled", False),
        "free_restoration_count": (
            linear_stats.get("free_restoration_count", 0)
            + conv_stats.get("free_restoration_count", 0)
        ),
    }


if __name__ == "__main__":
    # Smoke test on a fresh resnet50 — no calibration data needed for magnitude path
    import argparse
    import sys
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet50.tv_in1k", help="timm model name")
    p.add_argument("--out", default="conv_2_4_stats.json")
    p.add_argument("--method", choices=["magnitude", "cert_aware"], default="magnitude")
    args = p.parse_args()

    try:
        import timm
    except ImportError:
        print("pip install timm  (required)")
        sys.exit(1)

    model = timm.create_model(args.model, pretrained=True)
    print(f"Loaded {args.model}: "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"Eligible Conv2d layers: {len(list_eligible_convs(model))}")

    if args.method == "magnitude":
        stats = magnitude_2_4_for_conv(
            model, dense_state_dict=dict(model.named_parameters()),
            free_restoration=True, log=True,
        )
    else:
        # Cert-aware needs a calibration loader. For smoke test, fake it with random.
        import torch.utils.data
        rand_imgs = torch.randn(32, 3, 224, 224)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(rand_imgs), batch_size=8,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        stats = cert_aware_2_4_for_conv(
            model, loader, dict(model.named_parameters()),
            n_calib_imgs=32, device=device, log=True,
        )

    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2, default=float)
    print(f"\nSaved {args.out}")
    print(f"Net sparsity (eligible Conv2d only): "
          f"{stats['linear_sparsity_before']:.3f} -> {stats['linear_sparsity_after']:.3f}")
    print(f"Layers modified: {stats['n_layers_modified']}")
    print(f"Bad groups (should be 0): {stats['groups_with_more_than_2_nonzero_after']}")

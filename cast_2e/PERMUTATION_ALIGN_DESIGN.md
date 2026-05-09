# Permutation-Alignment for 2:4 Conv Pruning ("Flatten the Layer")

> **Status:** implemented in `project_conv_2_4.py` (functions `compute_cin_permutation`, `apply_cin_permutation_to_conv`, `permute_unfolded_h`, `attach_permutations_from_state_dict`) and exposed via the `--permute-align` flag in `run_resnet_cast_aws.py`. Round-trip tested (bitwise-identical via state_dict). Started **2026-05-08** as a response to the ResNet50d CAST result.

## What problem this solves

The ResNet CAST result on `resnet50d.ra2_in1k` showed a curious pattern:

| Stage | top-1 |
|---|---:|
| Dense teacher | 80.48% |
| SER source (s=0.35 RMT prior) | 80.04% |
| **Post-projection (pre-FT)** | **5.92%** |
| Post-FT (3 ep distill) | 78.08% |

Post-projection top-1 is severely degraded (5.92% on 1000-class ImageNet, where literal random would be ≈0.1% — so this is a near-collapse, not literal random, but functionally indistinguishable from a broken model). The 3 epochs of distillation FT recover the model to 78.08%, but essentially *all* of the headline result comes from the FT phase. The projection step itself produces a model that does not work.

The same pattern is *much milder* on ViTs:

| | Dense | SER source | Post-projection | Post-FT |
|---|---:|---:|---:|---:|
| ViT-B (CAST canonical) | 85.11 | 84.55 | **66.24** | 83.41 |
| ResNet50d (CAST-conv) | 80.48 | 80.04 | **5.92** | 78.08 |

ViT-B keeps a *recognizable model* after the 2:4 projection (66% top-1 — degraded but not random). ResNet50d collapses entirely.

### Why this happens

The 2:4 hardware constraint is a **fixed-stride contiguous partition** of the input axis: every 4-tuple of consecutive Cin slots must end up with exactly 2 nonzero entries. For convs, this partition is over `Cin · kH · kW` (the unfolded axis), but for the 1×1 convs that dominate the bottleneck, it's over `Cin` directly.

The unstructured RMT prior at s=0.35 leaves a sparse mask whose *signal* (RMT-edge-aligned) columns are clustered: the top-K most informative input channels for a layer can easily land in the same 4-tuple by accident. When 2:4 forces "keep exactly 2 of 4 in this group, drop the other 2," all four columns in that group might be RMT-signal — and we have to drop two of them. Meanwhile, an adjacent 4-tuple may be all RMT-noise and the 2 we keep there contribute almost nothing.

ViTs are more robust to this because (i) their Linear weights have flatter weight distributions per-row, so the cost of dropping a "signal" column is closer to that of dropping a "noise" column, and (ii) attention provides multi-head redundancy that buffers single-channel information loss.

ResNets do not have either property: bottleneck convs concentrate signal heavily in a few `Cin` channels, and there is no head-level redundancy.

## The fix: importance-balanced permutation

For each eligible Conv2d we compute an *importance score* per Cin channel:

```
I(c) = ‖W[:, c, :, :]‖_F² · E_x[‖h(x)[c]‖²]
```

(weight-energy times activation-energy — the contribution of channel `c` to the layer output, integrated over the calibration set). We then build a permutation `π` over `Cin` that places:

- the top-quartile-importance channels at position 0 of every 4-tuple,
- the 2nd-quartile at position 1,
- the 3rd-quartile at position 2,
- the bottom-quartile at position 3.

This guarantees that **every 4-tuple has at least one signal column the 2:4 mask can keep**, while still dropping the lowest-importance two of every four.

The permutation is applied in-place to the conv weight (`W[:, π, :, :]`) and a forward pre-hook is registered on the conv that does an `index_select` on the input's Cin axis at runtime. The permutation buffer (`_cin_perm`) is part of `state_dict` so it round-trips through save/load; a helper `attach_permutations_from_state_dict` registers the buffer + hook on a freshly-loaded model before `load_state_dict`.

After the permutation, we run the same cert-aware 2-of-4 search (`cert_cost = r^T · C · r`, where `r` is the dropped columns of the row and `C` is the activation covariance) on the *permuted* weight and *permuted* activations. The free-restoration step is unchanged.

## Why this preserves 50% MAC reduction

The permutation is a *zero-FLOP* operation:

1. The conv weight is permuted *once at projection time*, then frozen for FT.
2. At inference, the input is permuted via an `index_select` on Cin (memory bandwidth, not compute) before the conv. This is a memory reorder, not a multiply.
3. Optionally, the permutation can be *absorbed* into the previous layer's BatchNorm channel statistics + that layer's Cout axis, eliminating the runtime `index_select` entirely. (Not yet implemented; minor optimization.)

The 2-of-4 hardware partition still keeps exactly 2 of every 4 input channels, so the sparse-execution MAC count is identical to the original CAST. The only thing that changes is *which* 2 of the 4 the cert search picks, which is the same 50% sparsity but with better-aligned content.

## Architectural details

### What gets permuted

For 1×1 convs: the Cin axis directly aligns with the 2-of-4 partition. The permutation has its full effect.

For 3×3 convs: the unfolded axis is `Cin · kH · kW = Cin · 9`. With the standard `torch.nn.functional.unfold` layout, slot `c · 9 + p` is `(channel c, kernel position p)`. Most 4-tuples lie *within* one channel's 9-position block and the permutation only affects which channel sits next to which (the cross-boundary 4-tuples). The 1×1 effect is therefore much larger than the 3×3 effect; we apply the same Cin permutation logic to both for uniformity but expect the gain to land on the 1×1 convs.

For Linear layers (the FC head, when not skipped): same logic, treating `Cin = in_features`.

### What does NOT get permuted

- The 7×7 stem conv (Cin=3): not eligible.
- Depthwise / grouped convs (`groups > 1`): not eligible (each group has too few channels for 4-tuples).
- The skip-connection itself: the residual `x + F(x)` requires F's output to be in the same channel order as `x`. We only ever permute *Cin* of an internal conv; the output channel order of that conv is unchanged, and the residual remains correct.

### Cost vs. gain

| Item | Cost |
|---|---|
| Compute permutation (once per layer at projection) | O(Cin · log Cin) per layer; ≈ 1 ms |
| Apply permutation to weight (once) | O(Cout · Cin · kH · kW); ≈ 1 ms |
| Forward pre-hook at inference (every batch) | One `index_select` on `[B, Cin, H, W]`; ≈ 50–200 µs on T4 / A100; **dwarfed by the conv compute** |
| Storage overhead | One `LongTensor[Cin]` per layer ≈ 256 bytes |

Expected gain (hypothesis):

- ResNet50d post-projection top-1 climbs from 5.92% → 30–60% (we expect signal-rich 4-tuples to keep at least one signal column instead of clustering them).
- ResNet50d post-FT top-1 climbs from 78.08% → ~79–80% (the +1–2pp gap that the 2.40pp dense penalty is leaving on the table).

These are predictions, not measurements; the rerun on `--permute-align` produces the actual numbers.

### Why this is *not* a re-discovery of channel pruning

Channel pruning prunes *whole* channels (Cout-wise structured sparsity). The permutation here changes nothing about which channels exist — every channel still has 50% of its weights kept under 2:4. We only change *which 50%* by reordering the partition boundary. The model produces a tensor of the same shape; only the row-content is different.

## Affected vs. unaffected runs

| Model | Code path | Affected by --permute-align? |
|---|---|---|
| ResNet50/50d/101d | `run_resnet_cast_aws.py` → `cert_aware_2_4_for_conv` | **YES** (rerun with new flag) |
| ConvNeXt-Base / V2-Base | `run_cast_2e.py` → `cast_2e_pre_ft.py` (Linear pipeline, separate code) | **NO** (would need a separate Linear-side variant; deferred) |
| ViT-Large CAST | `run_cast_2e.py` (Linear pipeline) | **NO** (separate code path; existing run continues) |
| DeiT-T/S/B | (already done on GCP, results in cast_canonical_local) | Not rerun — use existing CAST results |

## Run plan (this branch)

After implementing + smoke-testing locally:

1. Sync new `project_conv_2_4.py` + `run_resnet_cast_aws.py` to S3.
2. **Pod 1 (RunPod A100):** kill current M3 (running with old method), restart with `--permute-align`. Chain M2-redo → M3-redo → M1-redo (resnet50.tv_in1k) → ConvNeXt-Base.
3. **AWS M1:** kill current AWS run, restart with `--permute-align` (redundant insurance).
4. **Pod 2 (ViT-L):** unaffected, keeps running canonical CAST.

After all reruns finish, append a **`+permute`** column to Table 5 of the paper showing the gain.

## Future extensions (deferred)

- **Absorb permutation into upstream BN + conv Cout**: zero-runtime-cost variant (free at inference), at the cost of more careful graph rewriting.
- **Linear-side permutation**: same idea applied to ViT/Swin/ConvNeXt 1×1 Linear layers. Would need to be plumbed into `cast_2e_pre_ft.py`, not just `cert_aware_2_4_for_conv`.
- **Joint optimization with the 2:4 cost**: the current implementation greedily picks π then runs the cert search. A small joint search (alternating the two) is a paper-grade follow-up.
- **Spatial-axis permutation for 3×3 convs**: permute the `Cin · 9` unfolded axis (not just Cin), eliminating the 1×1-only gain. Requires a `Unfold + Linear + Fold` rewrite at inference time.

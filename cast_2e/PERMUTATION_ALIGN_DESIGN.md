# Permutation alignment for 2:4 convolution pruning

> **Status:** implemented in `project_conv_2_4.py` through
> `compute_cin_permutation`, `apply_cin_permutation_to_conv`,
> `permute_unfolded_h`, and `attach_permutations_from_state_dict`; exposed by
> the `--permute-align` flag in `run_resnet_cast_aws.py`. State-dict
> round-trip tests are bitwise identical.

## Motivation

The ResNet CAST run on `resnet50d.ra2_in1k` produced the following result:

| Stage | top-1 |
|---|---:|
| Dense teacher | 80.48% |
| SER source (s=0.35 RMT prior) | 80.04% |
| **Post-projection (pre-FT)** | **5.92%** |
| Post-FT (3 ep distill) | 78.08% |

The 2:4 projection caused a large pre-fine-tuning accuracy drop. Fine-tuning
recovered most of the model accuracy, but the projection itself left a weak
starting point for recovery.

The drop was much smaller in the ViT CAST run:

| Model | Dense | SER source | Post-projection | Post-FT |
|---|---:|---:|---:|---:|
| ViT-B (CAST canonical) | 85.11 | 84.55 | **66.24** | 83.41 |
| ResNet50d (CAST-conv) | 80.48 | 80.04 | **5.92** | 78.08 |

This difference motivates an alignment step for convolutional input channels
before enforcing the fixed 2:4 partition.

### Mechanism

The 2:4 hardware constraint uses a fixed-stride contiguous partition of the
input axis: every 4-tuple of consecutive `Cin` slots must contain exactly two
nonzero entries. For convolutions, this partition is over `Cin * kH * kW`
after unfolding; for the 1x1 bottleneck convolutions, it is directly over
`Cin`.

The unstructured RMT prior at `s = 0.35` can cluster high-importance input
channels in the same 4-tuple. If more than two high-importance channels land in
one group, the 2:4 constraint forces the mask search to drop at least one of
them. A neighboring group can contain lower-importance channels and still keep
two entries. This mismatch can damage the projected model before fine-tuning.

ResNet bottleneck convolutions can concentrate signal in relatively few input
channels. ViT linear layers tend to have flatter per-row weight distributions,
and attention blocks provide additional redundancy. These architectural
differences explain why the same projection is less stable for the ResNet run.

## Importance-balanced permutation

For each eligible `Conv2d`, the implementation computes a channel-importance
score:

```text
I(c) = ||W[:, c, :, :]||_F^2 * E_x[||h(x)[c]||^2]
```

This is weight energy multiplied by calibration-set activation energy for input
channel `c`. The implementation then builds a permutation `pi` over `Cin` that
places one quartile of channels into each position of the 4-tuple:

- top-quartile channels at position 0,
- second-quartile channels at position 1,
- third-quartile channels at position 2,
- bottom-quartile channels at position 3.

The resulting partition spreads channel importance across 4-tuples before the
2:4 mask search. It keeps the same sparsity target but changes which channels
share a 4-tuple.

The permutation is applied to the convolution weight as `W[:, pi, :, :]`. A
forward pre-hook applies `index_select` to the input `Cin` axis at runtime. The
permutation buffer `_cin_perm` is part of the `state_dict`; on reload,
`attach_permutations_from_state_dict` registers the buffer and hook before
`load_state_dict`.

After permutation, the same certificate-aware 2-of-4 search is run on the
permuted weight and permuted activations:

```text
cert_cost = r^T C r
```

Here `r` contains the dropped columns of the row and `C` is the activation
covariance. The free-restoration step is unchanged.

## MAC accounting

The permutation does not change the 2:4 MAC count:

1. The convolution weight is permuted once at projection time and then frozen
   for fine-tuning.
2. At inference, the input permutation is an `index_select` on `Cin`. This is a
   memory reorder, not a multiply-add operation.
3. The permutation can later be absorbed into the upstream BatchNorm statistics
   and previous convolution output axis, eliminating the runtime `index_select`
   in a graph-rewrite implementation.

The 2-of-4 hardware partition still keeps exactly two entries per 4-tuple, so
the sparse-execution MAC count matches the original CAST projection.

## Architectural scope

### Permuted layers

For 1x1 convolutions, the `Cin` axis directly aligns with the 2-of-4 partition,
so the permutation applies to the full grouping used by the mask search.

For 3x3 convolutions, the unfolded axis is `Cin * kH * kW = Cin * 9`. With the
standard `torch.nn.functional.unfold` layout, slot `c * 9 + p` is channel `c`
at kernel position `p`. Most 4-tuples lie within a single channel's 9-position
block, so a `Cin` permutation mainly changes cross-boundary groups. The largest
effect is therefore expected on 1x1 bottleneck convolutions.

For linear layers, the same idea can be applied with `Cin = in_features`, but
the current implementation is the ResNet convolution path.

### Excluded layers

- The 7x7 stem convolution has `Cin = 3` and is not eligible.
- Depthwise and grouped convolutions with `groups > 1` are not eligible.
- Residual skip connections are not permuted. The method changes only the input
  channel order of internal convolutions; each convolution's output channel
  order is unchanged, so `x + F(x)` remains aligned.

## Cost accounting

| Item | Cost |
|---|---|
| Compute permutation once per layer | `O(Cin log Cin)` per layer; about 1 ms |
| Apply permutation to weight once | `O(Cout * Cin * kH * kW)`; about 1 ms |
| Forward pre-hook at inference | One `index_select` on `[B, Cin, H, W]`; about 50-200 us on T4 / A100 |
| Storage overhead | One `LongTensor[Cin]` per layer, about 256 bytes |

Measured accuracy and runtime effects should be taken from the run reports and
benchmark artifacts, not from projected values in this design note.

## Relation to channel pruning

Channel pruning removes entire channels. Permutation alignment does not remove
channels; every channel remains present, and 2:4 pruning keeps 50% of weights
within each hardware group. The method changes which weights share a 4-tuple
before the certificate-aware projection.

## Affected code paths

| Model | Code path | Covered by `--permute-align` |
|---|---|---|
| ResNet50 / ResNet50d / ResNet101d | `run_resnet_cast_aws.py` -> `cert_aware_2_4_for_conv` | Yes |
| ConvNeXt-Base / ConvNeXt-V2-Base | `run_cast_2e.py` -> `cast_2e_pre_ft.py` | No; separate linear pipeline |
| ViT-Large CAST | `run_cast_2e.py` | No; separate linear pipeline |
| DeiT-T / DeiT-S / DeiT-B | Existing GCP CAST results | No rerun planned in this design |

## Open implementation items

- Absorb the permutation into upstream BatchNorm and the previous convolution's
  output axis to remove the runtime `index_select`.
- Add a linear-layer variant for ViT, Swin, and ConvNeXt paths in
  `cast_2e_pre_ft.py`.
- Evaluate alternating optimization of the permutation and 2:4 certificate
  cost.
- Evaluate unfolded-axis permutation for 3x3 convolutions through an
  `Unfold + Linear + Fold` inference rewrite.

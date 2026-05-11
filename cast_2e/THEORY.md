# Theory grounding for ResNet CAST-2E

This document records which parts of the ResNet CAST-2E experiments follow
from the paper's certificate, which parts are convolutional extensions, and
which parts are empirical protocols or implementation controls. It is separate
from `SIMULATION_PLAN.md` because it defines claim boundaries rather than
experiment logistics.

## Claim boundary

The manuscript can state the following:

> CAST-2E extends to convolutional bottleneck architectures by treating each
> 1x1 convolution as a per-spatial-location linear map and each 3x3 convolution
> through its im2col representation. The certificate cost from Section 5
> (Lemma 5.4 / Corollary 5.5) applies through the per-layer covariance `C_g`
> of the input vector used by that representation. Reported ResNet results give
> analytical 2:4 MAC reductions and ImageNet top-1 accuracy. They do not report
> measured sparse-kernel speedup for Conv2d.

All other items in this file are implementation controls, ablation definitions,
or limitations on the interpretation of the experiments.

## Claim categories

### Direct consequences of the paper

| Component | Grounding |
|---|---|
| **SER s=0.35 source checkpoint** | The checkpoint is produced by the paper's RMT-based sparsification gate and is used as the input to CAST-2E. |
| **1x1 conv as a linear map** | For each spatial position, `y_{h,w} = W x_{h,w}` with `W in R^{Cout x Cin}`. The certificate cost is the Section 5 linear-layer cost with `h_g(x) = x_{h,w}`. |
| **Corollary 5.5 covariance form** | The per-group cost is `c_g(m) = r^T(m) C_g r(m)`, where `r(m) = (1 - m) o W_g` and `C_g = E_x[h_g h_g^T]`. The current code uses this covariance form rather than the earlier diagonal approximation. |
| **Hard 2:4 enumeration** | The optimization enumerates the `C(4, 2) = 6` masks with exactly two nonzero entries in each 4-tuple. The selected mask is optimal for the stated per-layer certificate objective under the calibration distribution. |

### Convolutional extensions and caveats

| Component | Scope |
|---|---|
| **3x3 conv through im2col** | The receptive field at one spatial position is a length `Cin * kH * kW = 9 * Cin` vector. The certificate cost on this unfolded form is `C_g = E_x[h_unfold h_unfold^T]`. The 4-tuple partition runs along the unfolded axis, so adjacent slots may represent different spatial and channel positions. |
| **Free 2:4 restoration through `slot_values`** | When SER zeroes more than two of four slots, the dense pretrained weight supplies the candidate restored value. The certificate cost is computed on `slot_values = where(SER kept, W_sparse, W_dense)`, so restoration is part of the mask search rather than a post-processing fill. This treats SER as a prior instead of a hard support constraint. |

### Implementation controls

| Control | Reason |
|---|---|
| **Mask freezing during fine-tuning** | `freeze_grad_at_masked` and `apply_masks` after `optimizer.step()` prevent gradient, momentum, and weight decay from regrowing zeroed entries. `assert_2_4_legality()` checks the structure after every epoch. |
| **SER load coverage check** | Prefix mismatches under `strict=False` can silently load few or no checkpoint weights. The run aborts unless load coverage is at least 95%. |
| **Hook-based MAC counting** | The hook-based counter evaluates ResNet layers at native resolution and avoids the earlier generic non-ViT fallback that reported no sparse-execution reduction. |
| **Deterministic calibration** | The calibration loader avoids random crops so that the certificate-aware mask search is reproducible. |

### Empirical protocols and ablations

| Item | Purpose |
|---|---|
| **3-epoch distillation fine-tuning** | Matches the prior ViT CAST runs for direct row-level comparability. The paper does not make a theory claim about this schedule. |
| **Distillation loss** | Uses `alpha * KL(T) + (1 - alpha) * CE` with `alpha = 0.5`, `T = 2`, and label smoothing 0.1. |
| **AdamW with cosine schedule** | Recovery optimizer and schedule for the empirical runs. |
| **Magnitude-only 2:4 ablation** | Replaces the certificate objective with `argmax_2(|W|)` within each 4-tuple to measure the value of the certificate. |
| **No-free-restore ablation** | Disables dense-weight restoration through `slot_values` to isolate the contribution of that mechanism. |
| **1x1-only ablation** | Restricts the convolutional extension to the clean 1x1 case and compares it with the 1x1+3x3 setting. |

## Non-claims

- The ResNet experiments do not claim Conv2d sparse-kernel wall-clock speedup.
  PyTorch `to_sparse_semi_structured` is Linear-only; the reported ResNet
  reduction is analytical MAC reduction.
- The selected 2:4 mask is optimal only for the stated per-layer certificate
  objective. It is not claimed to be globally optimal for all loss-aware or
  cross-layer objectives.
- The ResNet and ViT rows do not establish a theorem transferring one
  architecture's proof to the other. They use the same per-position linear or
  im2col primitive and are reported as separate empirical cases.
- Three epochs of fine-tuning are not directly comparable with structured
  pruning methods that use much longer schedules, such as NViT or HRank-style
  channel pruning. Comparisons should state the fine-tuning budget.

## Limitations and scope boundaries

- The 4-tuple partition for unfolded convolutional inputs is hardware-aligned
  and fixed. The implementation does not optimize the partition itself.
- The mask search is layerwise. It does not jointly optimize correlations
  between a classifier head and feature-extracting convolutional layers; the
  ResNet runs skip the head because it contributes a negligible share of MACs.
- Batch normalization is not sparsified. The 2:4 pattern is applied to
  convolution weights, while BN parameters remain dense or are folded according
  to the inference configuration.
- Depthwise and grouped convolutions are excluded by
  `is_eligible_conv(allow_grouped=False)`. This exclusion is irrelevant for
  ResNet but affects architectures with extensive grouped or depthwise layers.

These limitations constrain how the ResNet experiments should be interpreted;
they do not expand or weaken the claim boundary stated above.

## Run manifest requirements

- [ ] `git_commit` SHA in every `manifest.yaml`
- [ ] `engine_version` in every manifest
- [ ] `data_inputs.checksum_sha256` for SER ckpt + ImageNet val + train
- [ ] `n_calib_imgs = 256`, `seed = 42` with deterministic calibration
- [ ] `parameters.{method, include_3x3_convs, free_restoration, epochs, lr, ...}`
- [ ] `two_four_legality_check_after_each_epoch == PASS` for every epoch
- [ ] `bad_groups_after == 0` post-projection and post-FT
- [ ] Per-layer sparsity table preserved in `two_four_stats.json`
- [ ] Pre-FT and post-FT top-1 both reported

A run that fails any item in the checklist is not counted in Table 5.

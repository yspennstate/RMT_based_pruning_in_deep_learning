# Theory grounding — what's paper-aligned and what isn't

This document is the explicit "what is theoretical, what is empirical" map for
the ResNet CAST-2E experiments. It's separate from `SIMULATION_PLAN.md` so
reviewers can audit the theory claims without wading through experiment logistics.

## The headline claim we are willing to defend in the paper

> "We extend CAST-2E to convolutional bottleneck architectures by treating each
>  1×1 convolution as a per-spatial-location linear map and each 3×3 convolution
>  as its im2col equivalent. The certificate cost from §5 (Lemma 5.4 / Cor 5.5)
>  ports to both via the per-layer covariance C_g of the relevant input vector.
>  We report exact analytical 2:4 MAC reductions and ImageNet top-1 accuracy.
>  We do not claim measured sparse-kernel speedup for Conv2d."

That's it. Everything else is engineering.

## The full theory grounding ladder

### Tier 1 — Direct paper consequence (no new derivation needed)

| Component | Why it's tier-1 |
|---|---|
| **SER s=0.35 source checkpoint** | This IS the paper's RMT-based sparsification gate. It's the input, not something we re-justify. Paper §3 / §5. |
| **1×1 conv → Linear identity** | `y_{h,w} = W · x_{h,w}` with `W ∈ R^{Cout × Cin}`. The certificate cost is identical to the Linear case in §5 with `h_g(x) = x_{h,w}` for the chosen spatial position `(h,w)`. No new theorem. |
| **Cor 5.5 covariance form** | `c_g(m) = r^T(m) · C_g · r(m)` where `r(m) = (1−m) ⊙ W_g` and `C_g = E_x[h_g h_g^T]` is the per-group 4×4 input covariance. This is what the paper proves controls the per-layer reconstruction error. v3 code implements exactly this (was diagonal approximation in v1/v2). |
| **Hard 2:4 enumeration (6 patterns)** | The paper's optimization is over 4-tuple masks with `|m|_0 = 2`. There are exactly `C(4,2) = 6` such masks. We enumerate. No relaxation, no Gumbel, no STE. The result is the certificate-optimal mask under the calibration distribution. |

### Tier 2 — Clean extension with one explicit caveat

| Component | Caveat |
|---|---|
| **3×3 conv via im2col** | The receptive field at one spatial position is a length `Cin·kH·kW = 9·Cin` vector. The certificate cost on this unfolded form is well-defined: `C_g = E_x[h_unfold · h_unfold^T]`. **Caveat:** the 4-tuple partition runs along the unfolded axis, so adjacent slots may correspond to different `(spatial, channel)` positions. The covariance correctly captures the resulting correlation, but the partition no longer has the clean "4 input channels at one position" interpretation that the 1×1 case has. We say this in the footnote. |
| **Free 2:4 restoration via slot_values** | When SER zeroed >2 of 4 slots, the dense pretrained weight is the natural fill (it's what was there before SER pruned). The certificate cost is computed on `slot_values = where(SER kept, W_sparse, W_dense)`, so restoration is part of the optimization, not a heuristic post-fill. **The math is fine.** The caveat is only that the dense weight is "out of distribution" for the SER-pruned model, but since SER is a *prior* (not a hard constraint), this is consistent with the paper's framing. |

### Tier 3 — Engineering correctness (must be right; not theory)

| Component | What could break if wrong |
|---|---|
| **Mask freezing during FT** | Without `freeze_grad_at_masked` + `apply_masks` after `optimizer.step()`, gradient/momentum/weight-decay regrows zeroed entries within ~10 steps. The 2:4 structure dies, all reported reductions become wrong. v3 has this. Tested by `assert_2_4_legality()` after every epoch. |
| **SER load coverage check** | `strict=False` with `module.` prefix mismatch silently loads ~0% of weights. Without coverage assertion, you train an un-pre-trained model thinking it's SER-pruned. v3 enforces ≥95% coverage or the run aborts. |
| **Exact hook-based MAC counting** | The prior "generic non-ViT fallback" reported `tome_dense = sparse_exec` (no reduction at all) for ResNet50 in the original GCP runs. v3's hook-based counter at native resolution gives the real numbers. |
| **Deterministic calibration** | `RandomResizedCrop` on the calibration loader makes the cert-aware mask search non-reproducible. v3 uses a separate deterministic loader for calibration only. |

### Tier 4 — Empirical recovery (no theory claim, just standard practice)

| Component | Justification |
|---|---|
| **3-epoch distill FT** | Mirrors the prior ViT GCP runs for direct table-row comparability. The paper makes no claim about FT schedules. (Earlier doc drafts said "2-epoch"; canonical is 3.) |
| **Distill loss `α·KL(T) + (1−α)·CE`** | Standard knowledge distillation. `α=0.5`, `T=2`, label smoothing 0.1. Off-the-shelf. |
| **AdamW + cosine schedule** | Default for image classification recovery. Not theory. |

### Tier 5 — Ablation baselines (NOT theory, used only to quantify what theory buys)

| Component | What ablation is for |
|---|---|
| **Magnitude-only 2:4 (run A2)** | "What if we don't use the certificate at all and just pick `argmax_2(|W|)` per 4-tuple?" Quantifies the certificate's empirical value. |
| **No-free-restore (run A3)** | "What does slot_values restoration buy?" Quantifies that mechanism's contribution. |
| **1×1-only (run A1)** | "What's the gap between the cleanest theory case (1×1, tier-1) and the fuller 1×1+3×3 case (tier-2)?" |

## Things we will NOT claim in the paper

- ❌ "Sparse-kernel wall-clock speedup on ResNet". PyTorch `to_sparse_semi_structured`
  is Linear-only. We get FLOP reduction analytically; throughput is Conv2d-dense.
- ❌ "Optimal 2:4 mask under any objective". The certificate cost is one objective
  (per-layer reconstruction error). It's not the *only* one (e.g. cross-layer
  loss-aware optimization is provably tighter — but expensive). We're explicit
  that we use the per-layer cert cost.
- ❌ "ResNet ⇒ ViT generalization". Our framing is "the same primitive applies
  to both via per-position Linear / im2col equivalence", NOT "the proof for
  one transfers to the other". Each gets its own table row.
- ❌ "Beats published structured-pruning specialists at long schedules". 3 epochs
  of FT is not directly comparable to NViT (~50 epochs) or HRank-style channel
  pruning (~80 epochs). We say "at 3-epoch budget" everywhere.

## What a reviewer should pin us on

If a reviewer wanted to find a weakness in our theory grounding for ResNet,
the legitimate angle is:

1. **"The 4-tuple partition along the unfolded axis is arbitrary"** — yes,
   it is. We'd defend this by noting that any partition would give *some*
   certificate; we use the row-major partition for hardware compatibility.
   A future improvement is partition optimization (which is the v3 future
   work in `MaskLLM`-style learned masks), but we explicitly defer.

2. **"Why not joint Linear+Conv cert-aware optimization?"** — currently we
   run them independently. A joint formulation tracks correlations between
   the FC head's column-wise certificate and a feature-extracting conv's
   row-wise certificate. We'd argue: the head is 0.1% of MACs and we skip
   it for ResNet anyway, so the joint problem reduces to the conv-only
   problem we solve.

3. **"How are batch normalizations handled?"** — BN folds into adjacent
   convs (or stays separate, depending on inference mode). Our 2:4 pattern
   is on the conv weight; BN is unaffected. We don't sparsify BN.

4. **"What about depthwise / grouped convs?"** — explicitly skipped (v3
   `is_eligible_conv(allow_grouped=False)`). For ResNet there are none.
   For ConvNeXt there are many, which is one reason we defer ConvNeXt.

These are honest answers. None of them invalidate the headline claim.

## Reproducibility checklist (for the paper supplement)

- [ ] `git_commit` SHA in every `manifest.yaml`
- [ ] `engine_version` in every manifest
- [ ] `data_inputs.checksum_sha256` for SER ckpt + ImageNet val + train
- [ ] `n_calib_imgs = 256`, `seed = 42` (deterministic calibration)
- [ ] `parameters.{method, include_3x3_convs, free_restoration, epochs, lr, ...}`
- [ ] `two_four_legality_check_after_each_epoch == PASS` for every epoch
- [ ] `bad_groups_after == 0` post-projection AND post-FT
- [ ] Per-layer sparsity table preserved (in `two_four_stats.json`)
- [ ] Pre-FT and Post-FT top-1 both reported

A run that fails any item in the checklist is *not* counted in Table 5.

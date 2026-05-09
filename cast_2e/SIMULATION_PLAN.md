# CAST-2E ResNet — simulation plan (AWS-ready)

This document defines exactly what experiments will run on AWS, why each one
is needed, what theory backs it, and what we expect to report in the paper's
Table 5 (ResNet rows + appendix).

Generated 2026-05-07 after AWS approved G/VT Spot quota (4 vCPUs, us-east-1).

## TL;DR

- **6 runs** total: 3 ResNets × 2 main modes + 3 ablations on resnet50
- **Target headline**: ≥50% FLOP reduction with maintained ImageNet top-1
- **Cost**: ~$5–10 of the $160 AWS credit (g6.xlarge spot, ~$0.30/hr × 2.5–5h per run)
- **Runtime**: ~16–24 GPU-hours total, sequential (4-vCPU quota = one g6.xlarge at a time)
- **Output per run**: post-FT top-1, dense GMACs, eligible GMACs, sparse-exec GMACs, MAC reduction %, per-layer sparsity table, S3 checkpoints + manifest

## What we run

### Main paper rows (3 runs, one per model)

Each model: **cert-aware 2:4 on 1×1 + 3×3 convs** + **distill-FT 3 epochs** against the dense teacher.

(Prior CAST-2E ViT runs on GCP used 3 epochs; we keep 3 here for direct table-row comparability. The paper's "2-epoch budget" framing in early drafts referred to a tighter ablation; the canonical Table 5 row uses 3.)

| Run | Model | Mode | Expected FLOP reduction | Wall clock | Cost |
|---|---|---|---:|---:|---:|
| M1 | resnet50.tv_in1k | cert_aware, 1×1+3×3, free-restore | **48.6%** | ~2.5 h | ~$0.75 |
| M2 | resnet50d.ra2_in1k | cert_aware, 1×1+3×3, free-restore | **49.9%** | ~3 h | ~$0.90 |
| M3 | resnet101d.ra2_in1k | cert_aware, 1×1+3×3, free-restore | **49.9%** | ~5 h | ~$1.50 |

These three are the deliverable. They go in the paper's Table 5 ResNet rows.

### Diagnostic ablations on resnet50 (3 runs, paper appendix)

These pin down which ingredient is responsible for which gain. Same SER source,
same FT schedule, same teacher — only the projection method varies.

| Run | Model | Mode | Purpose |
|---|---|---|---|
| A1 | resnet50 | cert_aware, **1×1-only**, free-restore | Theory-purest baseline (≈26% reduction). Shows the gap between the cleanest theory case and the full 1×1+3×3. |
| A2 | resnet50 | **magnitude**, 1×1+3×3, free-restore | "What if we drop the certificate and just use \|W\|?" Quantifies cert-aware's value. |
| A3 | resnet50 | cert_aware, 1×1+3×3, **no free-restore** | "What does free restoration buy?" Quantifies the slot_values mechanism's contribution. |

Total: 6 runs. Sequential because spot quota = 4 vCPUs (one g6.xlarge at a time).

## Theory grounding — strongest to weakest

The paper's claim is "RMT-based, certificate-driven sparsification". Different
parts of the pipeline have different theory bonds. Honest reporting requires
distinguishing these.

### 🟢 Strong theory grounding (paper-aligned)

1. **SER s=0.35 source checkpoint.** Paper §3 / §5 — Sparsity-Equivalent Replacement is the RMT-based magnitude prior derived from the bulk eigenvalue distribution. We start every CAST-2E run from a checkpoint that already passed this gate. **Use as-is, no changes.**

2. **1×1 conv ≡ per-spatial-location Linear map.** Mathematical identity:
   ```
   y_{h,w} = W · x_{h,w}    where W ∈ R^{Cout × Cin}, x_{h,w} ∈ R^{Cin}
   ```
   The certificate cost from §5 (Lemma 5.4 / Cor 5.5) ports directly with no
   re-derivation — `c_g(m) = r^T · C_g · r` where `C_g = E[h h^T]` is the
   4×4 covariance over input channels at the chosen spatial position.

3. **Covariance-form certificate cost (`r^T C_g r`).** Includes the cross-terms
   `2·E[W_i h_i · W_j h_j]` that the per-slot diagonal approximation drops. For
   correlated input channels (which IS the case in deep ResNet stages), the
   cross-terms are not negligible and must be in the cost function. v3 implements
   this, replacing the v1/v2 diagonal approximation. Fixed per Codex review.

4. **Free 2:4 restoration via slot_values.** When SER zeroed >2 slots in a
   4-tuple, we restore from the dense pretrained weight. The key theoretical
   point: restoration is part of the certificate **search**, not a post-hoc fill.
   `slot_values = where(SER kept, W_sparse, W_dense)` is fed into the
   pattern-cost evaluation. Each 2-of-4 mask is scored as if those values were
   the actual weights. The restored value lives in the kept slot only if its
   pattern wins. This is faithful to the certificate's objective.

### 🟡 Reasonable but caveat-required (paper-aligned with a footnote)

5. **3×3 conv via im2col reshape.** Reshape `[Cout, Cin, kH, kW] → [Cout, Cin·kH·kW]`,
   apply 2-of-4 along the input axis. The certificate math still holds: at each
   spatial position the receptive field unfolds to a length-`Cin·kH·kW` vector,
   and `C_g = E[h_unfolded h_unfolded^T]` is well-defined. **The caveat:** the
   "input channels" of the unfolded form mix kernel-position with input-channel,
   so 4-tuples partition this combined axis. Adjacent slots in a 4-tuple may
   correspond to different `(kernel_position, channel)` pairs — that's a more
   heterogeneous covariance structure than the 1×1 case. The covariance form
   handles it correctly, but the interpretation is less clean than 1×1.

   **Footnote we'll write:** *"For 3×3 convolutions we apply CAST via the im2col
   equivalence between conv and per-position linear maps. The 4-tuple partition
   along the unfolded input axis (length Cin·kH·kW) mixes spatial neighbors and
   input channels; the certificate covariance C_g captures the resulting
   correlation structure but the partition's geometric interpretation is less
   direct than the pure 1×1 case."*

### 🟠 Empirically motivated, not derived from §5 (paper appendix only)

6. **3-epoch distill fine-tune** with `α·KL(T) + (1−α)·CE` loss, dense teacher,
   `α=0.5`, `T=2`, label smoothing 0.1. This is the recovery step. The paper's
   theory says nothing about FT schedules; this choice mirrors the original ViT
   experiments for consistency, not for theoretical reasons.

7. **Mask freezing during FT** — `freeze_grad_at_masked` + `apply_masks` after
   `optimizer.step()`. Engineering necessity (without it, the projection
   collapses on the first gradient step). Not theory, just correctness.

### 🔴 NOT theory (used only as ablation baselines)

8. **Magnitude-only 2:4** (run A2). No certificate, just `argmax_2(|W|)` per
   4-tuple. Reports as the ablation that quantifies how much the certificate
   buys you.

## What we measure per run

The runner (`run_resnet_cast_aws.py`) saves a `manifest.yaml` per run with:

```yaml
run_id: <model>_<mode>_<timestamp>
git_commit: <sha>
engine: tradeswarm.cast_2e_resnet  # or current path
data_inputs:
  imagenet_train: {path, checksum_sha256, last_validated}
  imagenet_val:   {path, checksum_sha256, last_validated}
  ser_checkpoint: {path, checksum_sha256, ser_load_coverage_fraction}
parameters:
  method: cert_aware|magnitude
  include_3x3_convs: true|false
  free_restoration: true|false
  epochs: 3
  batch_size: 64
  lr: 1e-5
  distill_alpha: 0.5
  distill_temp: 2.0
  label_smoothing: 0.1
  n_calib_imgs: 256
two_four_legality_check_after_each_epoch: PASS|FAIL
mac_report:
  dense_gmacs: float
  eligible_layer_count: int
  eligible_gmacs: float
  eligible_fraction: float
  sparse_exec_gmacs: float
  mac_reduction_fraction: float
sparsity_after:
  conv_layers_modified: int
  conv_nnz_before: int
  conv_nnz_after: int
  conv_sparsity_after: 0.500
  bad_groups_after: 0  # MUST be 0
eval:
  pre_ft_top1: float
  post_ft_top1: float
  delta_vs_baseline: float
ts_started: ISO8601
ts_completed: ISO8601
duration_seconds: float
```

Plus the per-epoch checkpoints (S3-mirrored every 10 minutes).

## Order of operations on AWS

```
0. Verify quota   (already confirmed: G/VT Spot = 4)
1. Upload local code + SER ckpts + ImageNet val to S3
   (train data: download in-instance from HF Hub, ~150 GB,
    one-time per region — see aws_setup.sh)
2. Launch g6.xlarge spot in us-east-1 with DLAMI Ubuntu
3. SSH in via SSM, run aws_setup.sh
4. Run M1 (resnet50, 1×1+3×3, cert_aware): tee log to S3, ~2.5 h
5. Re-evaluate before each next run: spot still alive? S3 backup current?
6. Run M2 (resnet50d): ~3 h
7. Run M3 (resnet101d): ~5 h
8. Run A1, A2, A3 (resnet50 ablations): ~7-8 h combined
9. Final eval table built from S3 manifest.yaml files
10. Terminate the spot instance
```

After step 9 we have:
- 6 manifest.yaml files in S3 → paste-into-paper Table 5 ResNet rows + appendix
- 6 final post_ft_*.pt checkpoints in S3 (each ~100 MB)
- Per-epoch checkpoints (small, kept for resume)

## Acceptance criteria (the runs we count as "succeeded")

A run is reported in the paper IF:
1. `bad_groups_after == 0` for every layer (2:4 legality preserved)
2. `ser_load_coverage_fraction ≥ 0.95` (SER ckpt actually loaded)
3. `two_four_legality_check_after_each_epoch == PASS`
4. **`post_ft_top1` is within ±0.5 pp of `dense_teacher_top1`** — the meaningful
   "maintained ImageNet top-1" comparison is against the **DENSE** baseline
   (what the original timm-published model scores), NOT against the post-projection
   pre-FT student (which has already lost some accuracy). The runner records
   `dense_teacher_top1`, `ser_source_top1`, `pre_ft_top1`, and `post_ft_top1`
   as four separate fields in `final_eval.json` and `manifest.yaml`.
5. The manifest.yaml is complete and signed (git_commit present)

Failures get logged but don't go in the paper. Specifically: if a run hits a
spot interruption that loses MORE than the last epoch boundary, we restart
from the latest S3 checkpoint, NOT from scratch — but we record this in the
manifest's `interruptions:` list for transparency.

## What's NOT in this run (deliberate)

- ConvNeXt — not BasicBlock or bottleneck; was on the to-do list for the prior
  GCP attempt, FAILED (Linear-only pipeline was incompatible). With v3 code
  (1×1 + 3×3 conv support) it would technically run, but ConvNeXt's depthwise
  + pointwise structure has different MAC distribution and merits its own
  paper analysis. **Defer.**
- Swin — windowed-shifted attention; same situation, defer.
- Hardware throughput speedup — PyTorch `to_sparse_semi_structured` is Linear-
  only. ResNet 1×1+3×3 sparsity gives FLOP reduction but no measured wall-clock
  speedup unless we wrap each Conv2d in `Unfold + Linear + Fold`. **Footnote
  in paper, defer to future work.**
- BasicBlock ResNets (resnet18, resnet34) — 3×3 dominates more, but the paper
  story is bottleneck-focused. **Defer.**

## Cost summary (from the $160 AWS credit)

| Bucket | Cost |
|---|---:|
| 6 g6.xlarge spot runs (~16-24 GPU-h × $0.30/h) | $5–10 |
| S3 storage (~5 GB checkpoints × $0.023/GB-mo, prorated) | <$1 |
| S3 transfer (training data already pulled to instance disk; outbound minimal) | <$1 |
| Cost Explorer / CloudWatch | $0 |
| **Total** | **$6–12** |

Leaves $148+ headroom for retries, hyperparameter exploration, and
unexpected items.

## Risks + mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Spot interruption mid-run | Medium (us-east-1 g6 spot interrupts ~5%/h) | per-epoch checkpoint to S3 every 10 min; runner auto-resumes from latest |
| ImageNet train data 150 GB download fails | Low | retry script, partial-download tolerant |
| HuggingFace gated model auth | Low | the relevant ResNet weights are public via timm, not gated |
| Numerical issues in covariance form (rank-deficient C_g) | Low | `_cert_cost_2_of_4` adds 1e-6·I if det==0; fallback to magnitude on failure |
| 50% reduction missed (e.g. only 47%) | Low | catalog includes per-layer eligible_fraction; we know exactly why |
| Top-1 drops >0.5pp on cert_aware | Medium | run M2 with higher LR or 3 epochs as fallback; reported in appendix |

## Next step

Execute via the master script `run_all_resnets_aws.sh` in this same folder.
That script handles: instance launch, setup.sh upload, S3 sync, the 6 runs
in sequence, and instance termination.

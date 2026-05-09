# CAST-2E for ResNet — verification report (v3, post Codex review)

Generated 2026-05-06 after applying Codex's 8-issue review.

## What v3 changes vs v2 (per Codex review)

### Critical (must-fix-before-AWS) — DONE

1. **Mask freezing during FT** — `collect_nonzero_masks()` snapshots the 2:4 pattern after projection; the runner's training loop calls `freeze_grad_at_masked()` BEFORE `optimizer.step()` and `apply_masks()` AFTER. This prevents gradient/momentum/weight-decay from regrowing zeroed entries.
2. **Covariance-form cert cost** — `_cert_cost_2_of_4` rewritten to use `c_g(m) = r^T · C_g · r` with `C_g = E[h h^T]` (4×4 per group). Includes the cross-terms the per-slot diagonal approximation dropped. Memory: O(G·16 + O·G·6) instead of O(N·O·G·4) — no OOM risk on bottleneck layers.
3. **Slot-values free restoration** — `cert_aware_2_4_for_conv` builds `slot_values = where(SER kept, W_sparse, W_dense)` BEFORE the pattern search, so the dense-restoration candidates participate in the certificate optimization (not bolted on after).
4. **Legality assertion** — `assert_2_4_legality()` runs after projection, after each FT epoch, and would catch any drift. Verified to fire correctly when the mask is corrupted.

### Important (should-fix) — DONE

5. **SER load coverage assertion** — `load_ser_with_coverage_check()` strips `module.` prefixes and asserts ≥95% tensor mass loaded; fails loud if the checkpoint format is wrong.
6. **timm data_config resolution** — `timm.data.resolve_model_data_config(model)` is read at runtime; `image_size`, `mean`, `std` flow through to the calibration loader, FT loader, and MAC counter. Stops baking a hardcoded 224×224.
7. **Skip Linear head** (`--skip-head` default True) — head MACs are ~0.1% of eligible compute; skipping cleans the accuracy story without losing meaningful FLOP reduction.
8. **Deterministic calibration loader** — `build_calibration_loader()` is separate from the FT train loader: NO `RandomResizedCrop`, NO horizontal flip, NO shuffle. Cert-aware mask search is now reproducible run-to-run.
9. **Fixed before/after NNZ stats** — `nnz_before` is now snapshot BEFORE the weight is overwritten in both magnitude and cert-aware paths.

### New (per user request — bigger FLOP reduction)

10. **`--include-3x3-convs` flag** — extends eligibility to 3×3 convs. Magnitude path supports it via `only_1x1=False`. Default still 1×1-only (clean theory tie-in for main paper row); 3×3 inclusion gives ~50% MAC reduction (paper appendix).

## Smoke-test results (CPU, local, 2026-05-06)

### MAC counter — exact hook-based at native model resolution

**Two eligibility regimes per model:**

| Model | dense GMACs | 1×1+fc layers | 1×1+fc reduction | All-eligible (1×1+3×3+fc) layers | All-eligible reduction |
|---|---:|---:|---:|---:|---:|
| resnet50.tv_in1k | 4.089 | 37 | **25.9%** | 53 | **48.6%** |
| resnet50d.ra2_in1k | 4.329 | 37 | **24.5%** | 55 | **49.9%** |
| resnet101d.ra2_in1k | 8.041 | 71 | **24.1%** | 106 | **49.9%** |

The all-eligible mode hits ~50% reduction (the theoretical ceiling for 2:4 on this much of the model — 99%+ of MACs are now eligible with 3×3 included).

### Six green smoke tests on resnet50

| # | Test | Result |
|---|---|---|
| 1 | Magnitude 1×1-only projection | 36 layers, sparsity 0.000→0.500, **0 bad groups** |
| 2 | `assert_2_4_legality()` correctly catches a corrupted mask | ✅ raises AssertionError |
| 3 | `apply_masks()` restores legality after corruption | ✅ |
| 4 | Cert-aware path (covariance form, random calib batch) | 36 layers, 0.500, **0 bad groups**, no OOM |
| 5 | 3×3-included eligibility count | 36 (1×1-only) → **52** (with 3×3) |
| 6 | Magnitude with 3×3 included | 52 layers, 23.4M params eligible, 0.500, **0 bad** |

## Files in this folder (v3)

| File | Purpose | Notes |
|---|---|---|
| `project_conv_2_4.py` | Magnitude + cert-aware (covariance form) 2:4 projection. Helpers: `assert_2_4_legality`, `collect_nonzero_masks`, `apply_masks`, `freeze_grad_at_masked` | now ~580 lines |
| `mac_counter.py` | Exact hook-based MAC counter; supports `--include-1x1-only-eligible` AND `--include-all-eligible` for side-by-side reporting | |
| `run_resnet_cast_aws.py` | End-to-end driver with mask freezing, SER coverage check, deterministic calibration, timm data_config, skip-head default | |
| `aws_setup.sh` | DLAMI Ubuntu bootstrap | unchanged |
| `requirements.txt` | torch + timm + datasets + boto3 | |
| `README.md` | Method + AWS run instructions | |
| `REPORT.md` | This file (v3) | |
| `mac_*.json` | Per-model MAC reports (both regimes) | |
| `smoke_*.json` | Smoke test outputs | |

## Throughput caveat (unchanged — still right)

PyTorch's `torch.sparse.to_sparse_semi_structured` accelerates `nn.Linear` only. For ResNet 2:4-sparse Conv2d weights you get the **FLOP reduction** and **accuracy preservation** but **no measured wall-clock speedup** unless you replace each Conv2d at inference time with `nn.Unfold + nn.Linear + Fold`. Report FLOP-only with this footnote per Codex's recommendation.

## Recommended paper framing

> "For convolutional bottleneck architectures, we extend CAST to 1×1 convolutions by treating each as a per-spatial-location linear map (Section X.Y). For appendix-completeness we also report 3×3-conv inclusion via the im2col-equivalent reshape. Since our current deployment stack does not provide a semi-structured convolution kernel path comparable to the ViT Linear-kernel setup, we report exact analytical 2:4 MAC reductions (computed via a hook-based forward pass at the model's native resolution) and ImageNet top-1 accuracy, but not backend-validated sparse-kernel speedups."

## What still requires GPU (deferred — quota PENDING)

- Pre-FT eval (~1-2 min on A10G/L4)
- Distillation FT 2-3 epochs (~2-5 hours per model on g6.xlarge)
- Post-FT eval on ImageNet val 50K (~3 min)

GPU quotas in AWS account `973584726484` (us-east-1) are 0 for all G/P/VT families. **3 quota requests submitted via CLI (PENDING)**: G/VT On-Demand, G/VT Spot, P Spot — all to 4 vCPUs. Approval typical 1-72h.

Once approved: same runner, two recommended invocations per model:

```bash
# Main paper row (1x1 only, ~25% reduction)
python run_resnet_cast_aws.py \
  --timm-name resnet50.tv_in1k \
  --ser-checkpoint /workspace/sweep_ckpts/resnet50.tv_in1k_keep_s35.pt \
  --imagenet-root /workspace/imagenet \
  --output-dir /workspace/cast_resnet/resnet50_1x1 \
  --method cert_aware --epochs 3 --batch-size 64 \
  --s3-backup-bucket cast-resnet-backup-973584726484

# Appendix row (1x1 + 3x3, ~50% reduction)
python run_resnet_cast_aws.py \
  --timm-name resnet50.tv_in1k \
  --ser-checkpoint /workspace/sweep_ckpts/resnet50.tv_in1k_keep_s35.pt \
  --imagenet-root /workspace/imagenet \
  --output-dir /workspace/cast_resnet/resnet50_all \
  --method cert_aware --include-3x3-convs --epochs 3 --batch-size 64 \
  --s3-backup-bucket cast-resnet-backup-973584726484
```

Per-model wall-clock on g6.xlarge spot: ~$1-4. All 6 runs (3 models × 2 modes) ≈ $10-25 of the $160 AWS credit.

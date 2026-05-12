# Deployability Audit

This audit preserves checkpoint weights. It loads the paper checkpoints, checks whether their
zero pattern matches a real sparse backend, and joins the result with existing
throughput logs. It does not re-project, fine-tune, or rewrite any checkpoint.

## Files

- `cast_2e/code/audit_checkpoint_deployability.py`: tensor-pattern audit.
- `cast_2e/code/benchmark_deployable_backends.py`: runtime audit of
  exact checkpoint endpoints on A40/A100.
- `cast_2e/code/build_deployable_speedup_ledger.py`: joins audit output with
  paper-row metadata and throughput logs.
- `data/paper_checkpoint_deployable_speedup_ledger_20260512.csv`: table-ready
  ledger.
- `data/deployable_backend_comparison_20260512.csv`: A40/A100/H100 backend
  comparison and batch sweep summary.
- `data/deployable_backend_best_observed_20260512.csv`: best observed A40
  sparse-backend ratios from the batch sweep, plus ResNet 1x1-hybrid controls.
- `data/runpod_a40_deploy_audit_20260512/resnet_1x1_a40_b128.csv`: A40
  control that lowers only exact ResNet 1x1 Conv2d layers to Linear and applies
  the native 2:4 Linear backend.
- `data/conv_tensorrt_deployability_audit_20260512.csv`: Conv2d TensorRT-style
  2:4 audit for ResNet/ConvNeXt checkpoints.

## Gate

Use a measured deployable speedup only when both conditions hold:

1. The checkpoint passes the backend-specific tensor audit.
2. The benchmark has non-null sparse-backend throughput for the unchanged
   checkpoint.

For PyTorch/NVIDIA semi-structured sparsity this means exact 2:4 on convertible
`nn.Linear` weights and successful `torch.sparse.to_sparse_semi_structured`
conversion. Wider `6:12`, `8:16`, and `12:16` checkpoints remain MAC-accounting
rows unless a matching native runtime is supplied and measured.

For Conv2d, flattened `2:4` MAC legality is not enough for a TensorRT sparse
Conv2d claim. TensorRT-style Conv2d 2:4 requires every group of four input
channels at each kernel pixel to contain at most two nonzeros. The current
audit therefore reports ResNet `CAST-conv+perm` as an exact im2col+2:4
sparse-GEMM deployment path, not as a faster native Conv2d/TensorRT path.

## Current Paper Rows

| Row | Top-1 | Theoretical MAC speedup | Measured deployable speedup | Audit result |
|---|---:|---:|---:|---|
| ViT-B/16 CAST 2:4 + ToMe | 83.41 | 2.49x | 1.36x | A40 native Linear 2:4, batch 128 |
| ViT-L/16 CAST 2:4 + ToMe | 84.37 | 2.50x | 1.37x | A40 native Linear 2:4, batch 128 |
| DeiT-B CAST 2:4 + ToMe | 80.48 | 2.49x | 1.36x | A40 native Linear 2:4, batch 128 |
| DeiT-S CAST 2:4 + ToMe | 76.96 | 2.49x | 1.37x | A40 native Linear 2:4, batch 128 |
| DeiT-T CAST 2:4 + ToMe | 65.93 | 2.49x | 1.33x | A40 native Linear 2:4, batch 128 |
| ConvNeXtV2-B CAST 2:4 | 85.47 | 2.00x | 1.26x | A40 native Linear 2:4, batch 128 |
| ResNet50 CAST-conv+perm | 75.67 | 1.94x | 1.70x | A40 exact im2col+2:4 sparse-GEMM vs dense im2col |
| ResNet50d CAST-conv+perm | 78.00 | 1.99x | 1.50x | A40 exact im2col+2:4 sparse-GEMM vs dense im2col |
| ResNet101d CAST-conv+perm | 80.59 | 2.00x | 1.57x | A40 exact im2col+2:4 sparse-GEMM vs dense im2col |
| ResNet152d CAST-conv+perm | 81.33 | 2.00x | 1.62x | A40 exact im2col+2:4 sparse-GEMM vs dense im2col |
| ViT-B 6:12, ViT-L 8:16, ConvNeXtV2 8:16/12:16 | see paper | 1.33-2.00x | none | non-native wider patterns |

The ResNet CAST-conv+perm checkpoints pass the flattened Conv2d audit but fail
the TensorRT-style input-channel audit:

| Row | TensorRT bad 2:4 groups |
|---|---:|
| ResNet50 CAST-conv+perm | 913,673 |
| ResNet50d CAST-conv+perm | 904,895 |
| ResNet101d CAST-conv+perm | 1,700,005 |
| ResNet152d CAST-conv+perm | 2,356,907 |

At batch 128 on A40, the sparse-im2col endpoints are still slower than cuDNN
Conv2d end-to-end: 0.437x, 0.372x, 0.381x, and 0.386x of native Conv2d
throughput for ResNet50/50d/101d/152d respectively. This is why the paper
labels the ResNet numbers as exact sparse-GEMM deployability audits rather than
native Conv2d speedups.

Additional controls:

- A40 batch sweep over 64/128/256 found stronger sparse-over-autocast ratios:
  1.388x ViT-B, 1.394x ViT-L, 1.384x DeiT-B, 1.376x DeiT-S, 1.330x DeiT-T,
  and 1.295x ConvNeXtV2. The fixed batch-128 audit remains archived for
  row-to-row comparison.
- ResNet 1x1-only hybrid control: converting only exact 1x1 Conv2d layers to
  native 2:4 Linear gives 1.087x, 1.088x, 1.081x, and 1.082x over dense
  1x1-linear hybrids for ResNet50/50d/101d/152d, but only 0.813x, 0.802x,
  0.788x, and 0.782x of native cuDNN Conv2d throughput. These are controls, not
  positive end-to-end ResNet speedup claims.
- Dense-BF16 controls are archived. Against an already-BF16 dense endpoint, the
  native sparse Linear advantage is small or absent for several rows, so the
  table should be read as a backend switch against the dense-tensor autocast
  endpoint used by the audit, not as a universal optimized-dense speedup.
- H100 with the tested PyTorch 2.4.1 build is not a valid endpoint: the
  semi-structured matmul operator errors with "Supported only on GPUs with
  compute capability 8.x".

## Reproduce

```powershell
python cast_2e/code/audit_checkpoint_deployability.py `
  --paths-file data/main_structured_checkpoint_paths_20260512.json `
  --output data/main_structured_deployability_audit_20260512.json `
  --csv-output data/main_structured_deployability_audit_20260512.csv `
  --keep-going

python cast_2e/code/build_deployable_speedup_ledger.py
```

Remote runtime audit example:

```bash
DEPLOY_AUDIT_CKPT_DIR=/workspace/deploy_audit/checkpoints \
python cast_2e/code/benchmark_deployable_backends.py \
  --rows linear --batch 128 --warmup 20 --iters 100 \
  --include-dense-bf16 \
  --output results/linear_full_a40_b128_with_dense_bf16.json
```

Do not use the exploratory `6:12 -> 2:4` projection files as deployable evidence
for the fixed paper checkpoints. That projection changes weights and can change
validation accuracy; the paper table uses the checkpoint-preserving audit above.

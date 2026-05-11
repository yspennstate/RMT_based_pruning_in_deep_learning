# Result result artifact manifest

Each row of the four result tables in the main paper is backed by the
JSON evidence listed below and, for post-FT rows, by the corresponding
`.pt` checkpoint.

## Main paper Table 3 (`tab:param_to_flop_followup`) — FLOP / speedup

| Row | Top-1 | Evidence JSON (in repo OR Downloads/cast_2e_resnet_review/) | Checkpoint | Hardware |
|---|---|---|---|---|
| ViT-B/16 CAST 2:4+ToMe | 83.41% | `cast_2e/sweep_results_initial/vitl_canonical/ft_phase/results.json` (`accuracy_top1.tome_post_ft = 83.414`); also `Downloads/.../vit_base_canonical_cast_20260504T170942Z/ft_phase/results.json` | (Pod 2) | A100/L4 |
| ViT-B/16 CAST 6:12 SER+α=0.5 | 83.74% | `Downloads/cast_2e_resnet_review/github_archive/pod1_complete/vitb_ft_inline/D612_ser_a05/final_eval.json` (`post_ft_top1 = 0.83738`) | `student_final.pt` (Pod 1) | A100 |
| ViT-L/16 CAST 2:4+ToMe | 84.37% | `Downloads/cast_2e_resnet_review/github_archive/vitl_canonical/ft_phase/results.json` (`accuracy_top1.tome_post_ft = 84.368`) | `vit_large_patch16_224.augreg_in21k_ft_in1k_cast2e_2to4_tome_r8_post_ft.pt` | A100 (Pod 2) |
| ResNet50.tv CAST-conv+perm | 75.67% | `Downloads/cast_2e_resnet_review/github_archive/pod1_complete/4resnet_ft_20260508_023715/1_resnet50.tv_in1k/final_eval.json` (`post_ft_top1 = 0.75672`) | epoch3.pt | (Pod 1) |
| ResNet50d CAST-conv+perm | 78.00% | `.../2_resnet50d.ra2_in1k/final_eval.json` (`post_ft_top1 = 0.77998`) | epoch3.pt | (Pod 1) |
| ResNet101d CAST-conv+perm | 80.59% | `.../3_resnet101d.ra2_in1k/final_eval.json` (`post_ft_top1 = 0.80588`) | epoch3.pt | (Pod 1) |
| ResNet152d CAST-conv+perm | 81.33% | `cast_2e/post_ft_eval/pod1_4resnet/final_eval.json` (`post_ft_top1 = 0.8133`); `Downloads/.../4_resnet152d.ra2_in1k/final_eval.json` | epoch3.pt | (Pod 1) |
| ResNet50.tv CAST 8:16 | 75.87% | `cast_2e/post_ft_eval/4resnet_8_16/resnet50.tv_in1k_8_16_final_eval.json` (`post_ft_top1 = 0.7587`) | `student_final.pt` (Pod 3a) | A100 |
| ConvNeXtV2-B CAST 2:4 | 85.47% | `Downloads/cast_2e_resnet_review/github_archive/convnextv2_canonical/ft_phase/results.json` (`accuracy_top1.tome_post_ft = 85.466`) | `convnextv2_base...cast2e_2to4_tome_r0_post_ft.pt` | (Pod 2) |
| ConvNeXtV2-B CAST 12:16 | 86.35% | `Downloads/cast_2e_resnet_review/github_archive/convnextv2_d1216/ft_phase/results.json` (`accuracy_top1.tome_post_ft = 86.354`) | `..._d1216_..._post_ft.pt` | (Pod 2) |
| ConvNeXtV2-B CAST 8:16 | 85.85% | `Downloads/cast_2e_resnet_review/github_archive/convnextv2_d816/ft_phase/results.json` (`accuracy_top1.tome_post_ft = 85.846`) | `..._d816_..._post_ft.pt` | (Pod 2) |

## Main paper Table 3 — speedup numbers

| Endpoint | Speedup | Evidence JSON |
|---|---|---|
| ViT-B/16 dense → 2:4 alone, A100, batch 128 | 2.705× | `cast_2e/benchmarks/vitb_bench_with_24.json` (`kernel_speedup_2_4_vs_dense = 2.705253...`, dense 463.585 ips, sparse 1254.116 ips) |
| ViT-B/16 ToMe-only → 2:4+ToMe, L4, batch 128 | 1.41× incremental | `cast_2e/sweep_results_initial/vitl_canonical/ft_phase/results.json` (dense 589.52 ips, sparse 836.26 ips, ratio 1.419) |
| ViT-L/16 ToMe-only → 2:4+ToMe, A100, batch 128 | 1.193× incremental | `Downloads/cast_2e_resnet_review/github_archive/vitl_canonical/ft_phase/results.json` (dense 1248.66, sparse 1489.86) |

## Main paper Table 2 (`tab:multi_arch_hybrid`) — Hybrid Mag–SER multi-arch

Each row's full sparsity-vs-top-1 trajectory comes from a per-architecture
results.json under `optuna_run/randomness_audit_results_*/full_run_*_hybrid_mag20_v8/`.
The methodology paper Table M.E (lines ~1273-1332 of `cast_2e/methodology.tex`)
contains the consolidated ledger. Spot-check sources:

- ViT-B/16: `optuna_run/randomness_audit_results_v11_hybrid/full_run_2026_04_26_exact_magnitude_to20_then_v8/results.json`
- ResNet50d: `optuna_run/randomness_audit_results_model_queue/queue_a/full_run_2026_04_26_resnet50d_hybrid_mag20_v8/results.json`
- ResNet101d: `optuna_run/randomness_audit_results_model_queue/queue_v11_resnet/full_run_2026_04_26_resnet101d.ra2_in1k_hybrid_mag20_v8/results.json`
- Hiera-Base+: `optuna_run/randomness_audit_results_model_queue/queue_h/full_run_2026_04_27_hiera_base_plus_224.mae_in1k_ft_in1k_hybrid_mag20_v8/results.json`
- ConvNeXtV2-B: `optuna_run/randomness_audit_results_model_queue/queue_f/full_run_2026_04_26_convnextv2_base.fcmae_ft_in22k_in1k_hybrid_mag20_v8/results.json`

## Main paper Table 4 (`tab:result_comparison`) — comparison vs prior work

Cited rows are from each cited paper's primary table:

| Row | Paper | Page/Table |
|---|---|---|
| NViT 83.29% | NViT (CVPR 2023) | Table 1 |
| SViTE-Base 50% 81.51% | SViTE (arXiv 2106.04533) | Table 5 |
| Spartan 90% 81.18% | Spartan (NeurIPS 2022) | Table 3 |
| GPUSQ-ViT INT8 82.90% | GPUSQ-ViT (CVPR 2023, arXiv 2305.10727) | Table 1 |
| Beyond 2:4 64:2:8 81.08% | "Beyond 2:4" (arXiv 2410.16135) | Table 5 |
| ELSA / LPViT / SERo | best 50%-class DeiT-Base row in each | various |

The "SparseGPT-style", "Wanda-style", and "AlphaPruning-style" rows are
project reimplementations; the cited papers are LLM-only and do not report
ViT-B/16 ImageNet numbers.

## Reproducibility

For a row, main paper Table 5 (`tab:reproduce_recipe`) and the
corresponding script in `cast_2e/code/` provide the reproduction recipe.
The full 30 GB of post-FT checkpoints is slated for HuggingFace Hub after
finalization; the current copies are on the three RunPod A100 pods and in
the local archive at
`Downloads/cast_2e_resnet_review/checkpoints_for_drive/`.

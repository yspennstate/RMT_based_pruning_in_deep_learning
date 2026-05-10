# Full data audit — CAST-2E experimental sweep

Compiled 2026-05-09. Lists every checkpoint, JSON, mask snapshot, and log
that has been produced or saved across the 3 active RunPod A100 pods plus
the local archive.

## Top-level summary by location

| Location | Total disk | What's there |
|---|---|---|
| **Pod 1** (port 12896) | ~13 GB run_outputs + 3 GB sweep_ckpts | 4-ResNet FT chain (post-FT 75-81%) + best-method sweep + ViT-B inline FT in flight + ViT-L SER ckpt staged |
| **Pod 2** (port 12632) | ~24 GB cast_runs + ~6 GB run_outputs + 3 GB sweep_ckpts | **ViT-L canonical (16 GB)** + ConvNeXtV2 canonical (in flight) + 18 advanced_v2 cell ckpts + 5 speedup benchmarks |
| **Pod 3a** (port 12059) | ~46 GB (mostly transferring train) | 12 sweep result JSONs + 2 mask dirs + ImageNet train arriving (32%) |
| **Local archive** | 333 MB / 124 files | All code, JSONs, mask stats, paper TeX/PDF |

---

## Paper-headline checkpoints (must preserve)

| Result | Where | Status |
|---|---|---|
| **ViT-L 84.37% post-FT** (canonical 2:4 + ToMe) | Pod 2 `/cast_runs/vit_large_canonical_20260507T204113Z/ft_phase/checkpoints/*post_ft.pt` (1.5 GB) | ✓ on Pod 2 disk; results.json + run.log pulled to local |
| **ResNet50.tv 75.67% post-FT** (CAST-conv+perm) | Pod 1 `/run_outputs/4resnet_ft_*/1_resnet50.tv_in1k/checkpoints/epoch3.pt` | ✓ on Pod 1 disk; final_eval.json pulled to local |
| **ResNet50d 78.00% post-FT** | Pod 1 `/run_outputs/4resnet_ft_*/2_resnet50d.ra2_in1k/checkpoints/epoch3.pt` | ✓ on Pod 1 disk; JSON pulled |
| **ResNet101d 80.59% post-FT** | Pod 1 `/run_outputs/4resnet_ft_*/3_resnet101d.ra2_in1k/checkpoints/epoch3.pt` | ✓ on Pod 1 disk; JSON pulled |
| **ResNet152d 81.33% post-FT** | Pod 1 `/run_outputs/4resnet_ft_*/4_resnet152d.ra2_in1k/checkpoints/epoch3.pt` | ✓ on Pod 1 disk; JSON pulled |
| **ViT-B 6:12 SER+α=0.5** post-FT (in flight) | Pod 1 `/run_outputs/vitb_ft_inline/D612_ser_a05/student_final.pt` | ⏳ ep2 in progress (~5 hr remaining) |
| **ConvNeXtV2-Base canonical 2:4** (in flight) | Pod 2 `/cast_runs/convnextv2_base_canonical_*/ft_phase/checkpoints/*post_ft.pt` | ⏳ ep3 in progress (~1 hr remaining) |
| **ViT-L 8:16 SER+α=0.5** (queued, not started) | Pod 1, will be `/run_outputs/vitl_ft_inline/D816_ser_a05_best/student_final.pt` | ⏳ waiting for ViT-B FT to finish |
| **ResNet50 8:16 dense+perm** (queued, not started) | Pod 3a, will be `/run_outputs/resnet_ft/D816_dense_perm/student_final.pt` | ⏳ waiting for train transfer (~13 hr) |

## 2:4 hardware speedup measurements (paper Table 5)

| Endpoint | Throughput | Source |
|---|---|---|
| Dense ViT-B/16 (A100, batch 128) | 463.6 im/s | benchmarks/vitb_bench.json |
| 2:4 ViT-B/16 sparse kernel | **1254.1 im/s (2.705×)** | benchmarks/vitb_bench_with_24.json |
| Dense ViT-L/16 + ToMe-r=8 (A100) | 1248.7 im/s | vitl_canonical/ft_phase/results.json |
| 2:4 ViT-L/16 + ToMe-r=8 sparse kernel | **1489.9 im/s (1.193× over ToMe-only, ~1.55× total over dense)** | vitl_canonical/ft_phase/results.json |
| L4 ViT-B 2:4 + ToMe (paper Table 5) | 593 → 837 im/s (1.41× over ToMe-only, ~1.84× total) | paper Table 5 |

## Local archive layout (`Downloads/cast_2e_resnet_review/`)

```
cast_2e_resnet_review/
├── README.md                                   # repo overview
├── PERMUTATION_ALIGN_DESIGN.md
├── REPORT.md, THEORY.md, SIMULATION_PLAN.md
├── *.py (17 scripts)                           # full pipeline code
├── *.sh (10+ launch/queue scripts)
├── github_archive/
│   ├── AUDIT.md                                # this file
│   ├── README.md                               # GitHub release README
│   ├── code/                                   # mirror of *.py
│   ├── results/                                # 18 sweep result JSONs
│   ├── benchmarks/                             # 6 throughput JSONs
│   ├── benchmarks_speedup/                     # 5 6:12→2:4 projection JSONs
│   ├── masks/{resnet50,vitb,convnext}/         # 17 mask_stats JSONs
│   ├── pod3a_sweep_results/                    # 12 Pod 3a sweep JSONs
│   ├── pod1_4resnet/                           # 4-ResNet final_eval JSONs
│   ├── ft_results/                             # perm meta JSON
│   └── vitl_canonical/                         # cast_2e_stats + ft_phase/results + run.log + recipe.sh
└── pod_results/
    ├── 4resnet_ft/                             # 4 final_eval JSONs (named)
    └── best_pod1/benchmarks/                   # benchmark JSONs
```

Additional archived paper build: `Desktop/RMT_pruning_VITs_Final/main.{tex,pdf}`
(90 pages, 2.78 MB, built at 00:01 UTC on 2026-05-09 with the latest available results at that time).

## Sweep results saved (all on Pod 3a + local)

12 JSON files documenting:
- 6 sparsity patterns (1:4, 2:4, 3:4, 4:8, 12:16) × dense/SER × α-sweep × 5 seeds
- 3 architectures (ResNet50, ViT-B, ConvNeXtV2)
- Calibration batch size sweep (c=64/256/512)
- Permutation on/off ablation

Each JSON contains per-cell: pre_ft_top1, layers_modified, projection_time, eval_time, sparsity, calib batch size, seed, α_ser value, and source identification.

## Code in repo

17 Python scripts implementing the full CAST-2E pipeline:
- `project_kn_sparsity.py` — cert-aware k:n projection (Conv + Linear)
- `project_kn_sampled.py` — sampled k:n for large n (16:32 etc)
- `project_cert_advanced.py` — mixed-sparsity, iterative, robust ℓ∞
- `project_conv_2_4.py` — original 2:4 cert framework (predecessor)
- `cert_opt_eval_best.py` — winner-cell sweep
- `cert_opt_eval_kn_advanced_v2.py` — extended sweep
- `cert_opt_eval_kn_extended.py` — Pod 3a 5-seed audit driver
- `cert_opt_eval_vitb_kn_extended.py` — ViT-B specific
- `benchmark_all_ckpts.py` — directory throughput benchmark
- `benchmark_6_12_to_2_4_projection.py` — speedup measurement via 2:4 projection
- `benchmark_sparse_throughput.py` — single-ckpt benchmark
- `run_vitb_ft_inline.py` — ViT-B inline projection + FT (avoids perm-hook bug)
- `run_resnet_ft_inline.py` — ResNet inline projection + FT
- `mac_counter.py`, `parquet_to_imagefolder.py`, helper modules

10+ shell scripts for launching/queueing on each pod.

## Risks (data preservation)

Pod-local state is not durable. Critical preservation items:
1. **Pod 2 ViT-L canonical (16 GB)** — paper-headline result. JSONs are local; the 1.5 GB post-FT checkpoint remains only on Pod 2. Action: copy to S3 or local storage.
2. **Pod 2 ConvNeXtV2 canonical** (in flight) — paper-headline result. Action: pull post-FT checkpoint and results.json after ep3 finishes.
3. **Pod 1 4-ResNet FT chain ckpts** (~3 GB epoch3 ckpts) — paper-headline. JSONs are local; checkpoints remain only on Pod 1 disk.
4. **Pod 3a sweep mask snapshots** — action: pull after the sweep completes.

The local archive contains all small artifacts (JSONs, mask stats, code, and paper sources/builds), but not the large checkpoints (about 30 GB across the 3 pods). It supports manuscript rebuilds and result inspection from saved summaries. The post-FT checkpoints should be published separately, for example on HuggingFace Hub, for full external reproduction.

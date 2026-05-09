# `cast_2e/` — Methodology and extended-numerics companion

This subdirectory contains the **companion methodology paper**, all the new
CAST-2E code, the result JSONs, and the speedup benchmarks that support the
new headline FLOP table in the main manuscript.

## Quick links

| Artefact | Path |
|---|---|
| **Methodology paper PDF** | [`methodology.pdf`](methodology.pdf) |
| Methodology paper TeX source | [`methodology.tex`](methodology.tex) |
| Full data audit | [`AUDIT.md`](AUDIT.md) |
| Theory↔code map | [`THEORY.md`](THEORY.md) |
| Permutation alignment design doc | [`PERMUTATION_ALIGN_DESIGN.md`](PERMUTATION_ALIGN_DESIGN.md) |
| Cell-sweep simulation plan | [`SIMULATION_PLAN.md`](SIMULATION_PLAN.md) |
| Run report | [`REPORT.md`](REPORT.md) |

## Code (`cast_2e/code/`, 25 files)

### Pre-fine-tuning projection — generalised cert-aware $k{:}n$

| File | Role | Methodology paper § |
|---|---|---|
| [`project_kn_sparsity.py`](code/project_kn_sparsity.py) | Cert-aware $k{:}n$ projection on Linear and Conv2d layers (Conv unfolds to (Cout, Cin·kh·kw)). Supports permute_align and α_ser. **The core projection routine.** | §3 |
| [`project_kn_sampled.py`](code/project_kn_sampled.py) | Sampled $k{:}n$ projection variant for large $n$ (e.g. 16:32) where enumerating all $\binom{n}{k}$ patterns is prohibitive. Draws ≤1024 candidates uniformly without replacement. | §3 |
| [`project_cert_advanced.py`](code/project_cert_advanced.py) | Mixed-sparsity, iterative refinement, and robust-ℓ∞ variants of the cert framework. | §3 |
| [`project_conv_2_4.py`](code/project_conv_2_4.py) | Original 2:4-only Conv2d cert framework (predecessor to `project_kn_sparsity.py`). | §9 |
| [`project_convnextv2_d1216.py`](code/project_convnextv2_d1216.py) | Standalone wrapper for ConvNeXtV2 12:16 dense-source projection (the 86.35% headline). Linear-only pipeline (depthwise convs excluded). | §3 |
| [`quick_prune_vitb224.py`](code/quick_prune_vitb224.py) | Quick magnitude-based ViT-B/16 pruning sanity-check tool. | §3 |

### Fine-tuning runners — inline projection + 3-epoch distillation

| File | Role | Methodology paper § |
|---|---|---|
| [`run_vitb_ft_inline.py`](code/run_vitb_ft_inline.py) | ViT inline FT runner (used for the **83.74%** ViT-B headline + ViT-L 8:16). AdamW, cosine LR with warmup, label smoothing, distillation alpha=0.5 T=2.0, mask-freeze re-zero every step. | §4 |
| [`run_resnet_ft_inline.py`](code/run_resnet_ft_inline.py) | ResNet inline FT runner (used for the 4-ResNet 8:16 chain producing **75.67% / 78.00% / 80.59% / 81.33%**). SGD+momentum, otherwise same recipe as ViT. | §4 |
| [`run_vitb_ft_from_ckpt.py`](code/run_vitb_ft_from_ckpt.py) | Older save-load FT runner — kept for reference; **deprecated** because of the perm-hook serialisation bug (top-1 drops to 0.0006 on reload). Do not use with `permute_align=True`. | §3.5 |
| [`run_resnet_cast_aws.py`](code/run_resnet_cast_aws.py) | AWS-side ResNet CAST runner (used during the AWS spot-instance pilot before we standardised on RunPod A100). | scripts |

### Speedup benchmarks

| File | Role | Methodology paper § |
|---|---|---|
| [`benchmark_sparse_throughput.py`](code/benchmark_sparse_throughput.py) | Single-checkpoint A100/L4 throughput measurement. Converts each Linear with a 2:4 mask into `torch.sparse.SparseSemiStructuredTensor` and reports median throughput over 100 iterations after 30 warm-up. | §5 |
| [`benchmark_6_12_to_2_4_projection.py`](code/benchmark_6_12_to_2_4_projection.py) | 6:12 → 2:4 deployable-speedup projection benchmark. Maps each 4-sub-block of a 12-tuple to its nearest 2:4 pattern. | §5 |
| [`benchmark_all_ckpts.py`](code/benchmark_all_ckpts.py) | Directory-scan throughput benchmark — runs `benchmark_sparse_throughput.py` over every `*.pt` in a directory. | §5 |
| [`mac_counter.py`](code/mac_counter.py) | Theoretical MAC counter for sparse Linear/Conv layers; used to compute the headline FLOP-reduction percentages in the new FLOP table. | §5 |

### Cell-sweep evaluators

These drive the dense-vs-SER × $k{:}n$ × α_ser × seed sweep on the pods. Each
emits a JSON of the schema in methodology paper §11.2. Twelve total
(corresponding to distinct sweep generations and architectures, retained as
separate files for reproducibility):

| File | Role |
|---|---|
| [`cert_opt_eval.py`](code/cert_opt_eval.py) | Round-1 evaluator; ResNet50 and ViT-B 2:4-only baselines. |
| [`cert_opt_eval_8_16.py`](code/cert_opt_eval_8_16.py) | 8:16 specific evaluator (the largest deployable-pattern sweep). |
| [`cert_opt_eval_advanced.py`](code/cert_opt_eval_advanced.py) | Mixed-sparsity + iterative variants. |
| [`cert_opt_eval_best.py`](code/cert_opt_eval_best.py) | Winner-cell sweep — given the best cell of an architecture, replicate with 5 seeds. |
| [`cert_opt_eval_kn.py`](code/cert_opt_eval_kn.py) | Generalised $k{:}n$ evaluator (round-2). |
| [`cert_opt_eval_kn_advanced_v2.py`](code/cert_opt_eval_kn_advanced_v2.py) | Extended sweep — Pod 2 round-3 driver with α_ser sweep. |
| [`cert_opt_eval_kn_extended.py`](code/cert_opt_eval_kn_extended.py) | Pod 3a 5-seed audit driver. |
| [`cert_opt_eval_vitb.py`](code/cert_opt_eval_vitb.py) | ViT-B specific 2:4 evaluator. |
| [`cert_opt_eval_vitb_kn.py`](code/cert_opt_eval_vitb_kn.py) | ViT-B specific $k{:}n$ evaluator. |
| [`cert_opt_eval_vitb_kn_extended.py`](code/cert_opt_eval_vitb_kn_extended.py) | ViT-B specific extended sweep. |

### Helpers

| File | Role |
|---|---|
| [`parquet_to_imagefolder.py`](code/parquet_to_imagefolder.py) | Convert HuggingFace parquet ImageNet-1k shards to torchvision ImageFolder layout. Used once per pod when bootstrapping. |

## Scripts (`cast_2e/scripts/`, 16 launch files)

| File | Role |
|---|---|
| `pod1_best_2hr_chain.sh` | Pod 1 best-method 2-hour chain (the 4 paper-headline runs). |
| `pod1_2_4_speedup_supplement.sh` | Pod 1 2:4 throughput supplement. |
| `pod3_full_chain.sh`, `pod3_setup.sh` | Pod 3 (deprecated by Pod 3a). |
| `pod3a_comprehensive.sh` | Pod 3a comprehensive sweep (all 3 archs × 6 patterns × dense/SER × α_ser × seed). |
| `queue_pod3a_4resnet_8_16_chain.sh` | Pod 3a 4-ResNet 8:16 cert+perm 3-ep FT chain. |
| `queue_pod1_vitl_after_vitb.sh` | Sequential ViT-B → ViT-L runner on Pod 1. |
| `queue_pod2_post_vitl.sh`, `queue_pod2_post_vitl_v2.sh`, `queue_pod2_vitl_8_16.sh` | Pod 2 ViT-L queueing variants. |
| `launch_pod2_kn_sweep.sh`, `launch_pod2_resnet_rerun.sh`, `launch_pod2_round2_watcher.sh`, `launch_pod2_vitb_priority.sh` | Pod 2 launchers. |
| `aws_setup.sh`, `run_all_resnets_aws.sh` | AWS pilot launchers. |

## Data

| Subdir | What's there |
|---|---|
| [`benchmarks/`](benchmarks/) | Dense vs 2:4 throughput JSONs (A100, L4) — 6 files |
| [`benchmarks_speedup/`](benchmarks_speedup/) | 6:12 → 2:4 projection-speedup measurements — 5 files |
| [`sweep_results/`](sweep_results/) | k:n cell-sweep result JSONs (Pod 3a) — 12+ files |
| [`sweep_results_initial/`](sweep_results_initial/) | First-round sweep JSONs (Pod 1, Pod 2) — 18 files |
| [`post_ft_eval/`](post_ft_eval/) | Post-FT final-eval JSONs for the 4 paper-headline ResNet runs |
| [`masks/`](masks/), [`masks_archive/`](masks_archive/) | Per-layer mask statistics from the cell sweeps |

## Pointers from the main paper

The main paper's appendices B, E, F, G have been migrated here verbatim to
keep the main manuscript focused on theory and headline results:

| Main-paper appendix | Methodology-paper section |
|---|---|
| App B (Other Numerical Results) | §6 (`sec:fc_num_meth`) |
| App E (BEMA / RMT diagnostics) | §7 (`sec:bema`) |
| App F (Spectral Edge Budgeting) | §8 (`sec:seb-protocols`) |
| App G (CAST 2:4 + ToMe) | §9 (`sec:cast-old-app`) |

The methodology paper additionally introduces:

- §3 the generalised CAST-2E `k:n` projection with permutation alignment and `α_ser` prior
- §4 the FT recipe (distillation, mask-freeze, optimiser config)
- §5 speedup measurement (A100/L4, native-2:4, 6:12→2:4 projection)
- §10 the certification audit (three bridges, 282 cells, 18 architectures)
- §11 detailed numerics for §3 of the main paper
- §12 a quick-reference index from main-paper topics to methodology-paper sections

## Reproducing a paper-headline number end-to-end

```bash
# ViT-B 6:12 SER+α=0.5 (the 83.74% headline)
python cast_2e/code/run_vitb_ft_inline.py \
    --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --ser-checkpoint /path/to/vit_base_patch16_224_keep_s35.pt \
    --imagenet-train /path/to/imagenet/train \
    --imagenet-val   /path/to/imagenet/val \
    --output-dir     ./vitb_ft_out \
    --k 6 --n 12 --source ser --alpha-ser 0.5 \
    --epochs 3 --batch 32 --lr 1e-5 --weight-decay 0.01 \
    --label-smoothing 0.1 --distill-temp 2.0 --distill-alpha 0.5 \
    --warmup-steps 1000

# ResNet50.tv 8:16 dense+perm (the 75.67% headline)
python cast_2e/code/run_resnet_ft_inline.py \
    --timm-name resnet50.tv_in1k \
    --imagenet-train /path/to/imagenet/train \
    --imagenet-val   /path/to/imagenet/val \
    --output-dir     ./resnet50_ft_out \
    --k 8 --n 16 --source dense --alpha-ser 0.0 \
    --epochs 3 --batch 256 --lr 1e-3 --weight-decay 1e-4 --momentum 0.9
```

The runner emits a `final_eval.json` with the post-FT top-1, the per-epoch
`student_ep{1,2,3}.pt`, and a `student_pre_ft.pt` containing the projected
state-dict + cert metadata. See [`AUDIT.md`](AUDIT.md) for the complete map
from numerical claim to code path to data file.

## Big checkpoints

The 30 GB of post-fine-tuning checkpoints (`student_ep1.pt`, …, `student_final.pt`
for each of the 4 ResNet runs, ViT-L, ViT-B, ConvNeXtV2 D816/D1216, etc.)
are not stored in this Git repository — they are referenced in
[`AUDIT.md`](AUDIT.md) and will be uploaded to a HuggingFace Hub release when
this paper is finalised. Inference and reproducibility verification can use
the projection script + a fresh 3-epoch FT to reproduce each post-FT number
end-to-end.

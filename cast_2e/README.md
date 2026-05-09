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

## Code

| Subdir | What's there |
|---|---|
| [`code/`](code/) | All CAST-2E Python: cert-aware k:n projection (`project_kn_sparsity.py`), inline FT runners (`run_vitb_ft_inline.py`, `run_resnet_ft_inline.py`), throughput benchmarks (`benchmark_*.py`), MAC counter, sweep evaluators (`cert_opt_eval_*.py`) |
| [`scripts/`](scripts/) | Pod-side launch scripts (`pod1_*.sh`, `pod3a_comprehensive.sh`, `queue_*.sh`) |

## Data

| Subdir | What's there |
|---|---|
| [`benchmarks/`](benchmarks/) | Dense vs 2:4 throughput JSONs (A100, L4) |
| [`benchmarks_speedup/`](benchmarks_speedup/) | 6:12 → 2:4 projection-speedup measurements |
| [`sweep_results/`](sweep_results/) | k:n cell-sweep result JSONs (Pod 3a) |
| [`sweep_results_initial/`](sweep_results_initial/) | First-round sweep JSONs (Pod 1, Pod 2) |
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

## Reproducibility

Every paper-headline number can be reproduced by running the corresponding
script in `code/`. See `AUDIT.md` for the complete map from numerical claim
to code path to data file.

## Big checkpoints

The 30 GB of post-fine-tuning checkpoints (`student_ep1.pt`, `student_ep2.pt`,
`student_ep3.pt`, `student_final.pt` for each of the 4 ResNet runs, ViT-L,
ViT-B, ConvNeXtV2 D816/D1216, etc.) are not stored in this Git repository —
they are referenced in `AUDIT.md` and will be uploaded to a HuggingFace Hub
release when this paper is finalised. Inference and reproducibility
verification can use the projection script + a fresh 3-epoch FT to reproduce
each post-FT number end-to-end.

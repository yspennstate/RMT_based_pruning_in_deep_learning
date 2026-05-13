# Pruning Deep Neural Networks via Random Matrix Theory

Reproduction code, **paper PDFs**, and the **methodology companion paper** for
the manuscript: *"Pruning Deep Neural Networks via the Marchenko–Pastur
Distribution"*.

## Papers

| Paper | PDF | TeX source |
|---|---|---|
| **Main manuscript** (NPL letter-style paper, 24 pages including references) | [`paper/main.pdf`](paper/main.pdf) | [`paper/main.tex`](paper/main.tex) |
| **Online Resource 1** (proofs and mathematical supplement, 32 pages) | [`paper/ESM_1.pdf`](paper/ESM_1.pdf) | [`paper/ESM_1.tex`](paper/ESM_1.tex) |
| **Online Resource 2** (methodology, protocols, and numerical supplement, 72 pages) | [`cast_2e/methodology.pdf`](cast_2e/methodology.pdf) | [`cast_2e/methodology.tex`](cast_2e/methodology.tex) |

The main manuscript is the concise Neural Processing Letters submission. It keeps the main corollaries, the main ImageNet tables, the compact closest-work comparison table, Springer-style declarations, and explicit citations to Online Resource 1 and Online Resource 2. The complete proof stack is in Online Resource 1. Online Resource 2 contains the full RMT protocol stack (BEMA, SEB, SER, Hybrid Magnitude-SER), the CAST-2E pipeline (cert-aware $k{:}n$ projection, permutation alignment, free restoration, frozen-mask distillation), timing-audit details, checkpoint/protocol ledgers, migrated numerical appendix material, and the complete literature-context table.

The Overleaf-ready same-folder package is [`final_submission/NPL_Overleaf_20260513_ready.zip`](final_submission/NPL_Overleaf_20260513_ready.zip). It contains `main.tex`, `ESM_1.tex`, `ESM_2.tex`, the merged bibliography, required figures, cross-resource aux files, and the three compiled PDFs.

## Methods covered

The repository contains every method evaluated in either paper:

- **Classical magnitude pruning** — global unstructured baseline.
- **Classical RMT pruning** — Marchenko–Pastur (MP) edge fitting + sub-edge bulk denoising (negative control).
- **Spectral Edge Budgeting (SEB / S+)** — no-fine-tuning, layerwise magnitude budget allocated by the fitted MP edge \(\sigma_+(\ell)\) and Hill exponent \(\alpha_\ell\).
- **Spectral Edge Reallocation (SER)** — main RMT prune-restore method: prune beyond the target, form a reservoir, reinsert an RMT-ranked budget, run a short frozen-mask fine-tuning phase.
- **Hybrid Magnitude–SER** — exact global magnitude pruning through \(s = 0.20\), then SER for \(s \ge 0.25\). This is the protocol of Tables 1 and 2 in the paper.
- **Drop-threshold variant** — stage-1 magnitude continues until post–FT top-1 drops by more than 0.7 percentage points, then stage-2 (SER) begins. Two flavors:
  - **From-scratch**: stage-1 starts from the dense model.
  - **Seeded**: stage-1 starts from the canonical \(s=0.20\) checkpoint and continues magnitude past it.
- **CAST** — Certificate-Aware Sparse-Token conversion: cert-aware 2:4 mask + ToMe + 3-epoch frozen-mask distillation.
- **CAST-2E** (NEW; under [`cast_2e/`](cast_2e/)) — generalised cert-aware $k{:}n$ projection (k:n in {2:4, 4:8, 6:12, 8:16, 12:16}), Cin permutation alignment ("flatten the layer"), $\alpha_{\mathrm{ser}}$ Hamming-distance prior to the SER mask, and inline 3-epoch distillation FT. Underlies the new FLOP table in the main paper (ViT-B 83.74%, ViT-L 84.37%, ResNet50 75.67%, ResNet50d 78.00%, ResNet101d 80.59%, ResNet152d 81.33%, ConvNeXtV2 86.35% / 85.85%).

Code currently used to produce the validated checkpoints in the paper is preserved as-is — file names retain their internal ("v8", "until_drop", etc.) lineage, but the table below maps every file to the paper-facing method name.

## Relationship to the prior repository

A predecessor of this code is the public repo **[yspennstate/RMT_pruning_ViT](https://github.com/yspennstate/RMT_pruning_ViT)**, which implements the Marchenko–Pastur–based ViT-Base pruning algorithm of Berlyand, Bourdais, Owhadi & Shmalo (2025) (Pruning Deep Neural Networks via a Combination of the Marchenko–Pastur Distribution and Regularization; ResearchGate publication 389484743). The current repository is a strict superset of that code:

1. **Inherited modules.** `src/RMT.py`, `src/SplittableLayers.py`, `src/training.py`, `src/utils.py` are byte-identical with the prior repo. `src/pruning.py` and `src/validation.py` are extended with reservoir / reinsert primitives and a multi-resolution validation path (no behavioural change to the original primitives). The two top-level entry scripts `prune.py` and `fine_tune.py` are also identical with the prior repo, so the ViT-Base figure of Berlyand et al. reproduces exactly.
2. **What is new.** The protocols studied in the current paper — SER, Hybrid Magnitude–SER, drop-threshold variant (from-scratch and seeded), Spectral Edge Budgeting / S+ with a Haar bulk model, the multi-architecture queue runners, and the layer-aware adaptive RMT controller in `adaptive_rmt/` — are all new. They sit on top of the inherited library; they do not replace any of it.
3. **What changed in the inherited code.** Hardcoded paths and the prior repo’s baked-in HuggingFace token have been removed and replaced by environment variables (`HF_TOKEN`, `RMT_OPTUNA_RUN`, `RMT_CACHE`). No algorithmic changes.

For reproducing only the original ViT-Base figure, the prior repo is smaller and self-contained. For the full multi-architecture protocol comparison in Tables 1 and 2 of the new paper, use this repository.

---

## File map (paper method → code)

### Pipeline entry points

| Paper method | File | Notes |
|---|---|---|
| **Hybrid Magnitude–SER**, single architecture | `hybrid_mag20_then_v8.py` | Run for one architecture, full s=0.05–0.70 schedule. |
| **Hybrid Magnitude–SER**, single model with `--model` arg | `hybrid_mag20_then_v8_model.py` | Same protocol, parameterized by timm checkpoint name. |
| **Hybrid Magnitude–SER**, multi-arch queue runner | `hybrid_mag20_then_v8_model_queue.py` | Used to drive the multi-architecture rows of Table 2 (one process per pod queue). |
| **Drop-threshold variant**, single model | `hybrid_mag_until_drop_then_v8_model.py` | Stops stage-1 when post-FT top-1 drops > 0.7 pp. |
| **Drop-threshold variant**, queue runner | `hybrid_mag_until_drop_then_v8_model_queue.py` | Multi-arch driver for the variant. |
| **Multi-arch orchestrator** (calls the queue runners) | `model_queue_runner.py` | Top-level loop that selects the next model from the per-queue config. |
| **Multi-arch launcher** (queue control entry point) | `start_model_queue.py` | Spawns or resumes a queue. |

### Classical magnitude pruning (Appendix “S+ method”)

| Paper method | File |
|---|---|
| Classical magnitude pruning sweep | `magnitude_rmt_sweep.py` |
| Classical magnitude + RMT comparison sweep | `rmt_magnitude_sweep.py` |
| Classical magnitude vs uniform-budget comparison | `rmt_vs_uniform.py` |
| Magnitude pre-fine-tune helpers | `run_finetune_magnitude.py` (and `_v3.py`, `_v4.py`) |
| Magnitude pre-fine-tune for a specific timm checkpoint | `run_finetune_magnitude_model_exec.py` |
| Magnitude pre-fine-tune queue driver | `run_finetune_magnitude_model_exec_queue.py` |
| End-to-end fine-tune pipeline | `run_finetune_pipeline.py` (and `_v2.py`) |

### Classical RMT pruning (Appendix “Classical RMT”)

| Paper method | File |
|---|---|
| Classical RMT cycle implementation | `prune.py` |
| Theory-driven RMT prune (negative control) | `theory_pruning.py` |
| Per-layer RMT diagnostics + randomness null tests | `randomness_diagnostic.py` |

### Spectral Edge Budgeting (SEB / "S+" with Haar bulk model)

| Paper method | File |
|---|---|
| Haar/MP optimizer for the SEB hyperparameters \((\beta, s_\mathrm{d}, p)\) | `haar_optuna.py` |
| Refined Haar optimizer (later sweep iterations) | `haar_optuna_refined.py` |
| Haar small-z search (low-z regime) | `haar_small_z.py` |
| Hyperparameter grid search v1 | `hp_search.py` |
| Hyperparameter grid search v2 | `hp_search_v2.py` |
| Iterative 5%-step comparison | `iterative_5pct_compare.py` |
| Iterative “growing a” schedule | `iterative_growing_a.py` |
| SV-decides-when-to-stop iterative variant | `iterative_sv_decides.py` |
| Sweep analyzer | `analyze_sweep.py` |
| SEB hyperparameter long-grid driver | `pruning_hyperparam_grid_search.py` |
| Pruning-method comparison sweep | `pruning_method_comparison_sweep.py` |

### Spectral Edge Reallocation (SER) ablations

| Paper method | File |
|---|---|
| Removed-matrix audit (v5 cycle of SER) | `run_removed_matrix_audit_v5_model_exec.py` |
| Removed-matrix audit (v8 cycle of SER, current) | `run_removed_matrix_audit_v8_model_exec.py` |
| Resume a partial SER run | `remote_resume_suffix.py` |
| Status snapshot | `remote_status_once.py` |
| Resume supervisor | `remote_suffix_supervisor.py` |

### Singular-value preprocessing tests (Appendix “SV preprocessing”)

| Paper method | File |
|---|---|
| Spectral denoise prune-only test | `spectral_denoise_test.py` |
| SV pruning test | `sv_pruning_test.py` |
| SV theory probe | `sv_theory_test.py` |
| SV hyperparameter grid | `sv_hp_grid.py` |
| SV power-law grid | `sv_power_grid.py` |
| SV threshold grid | `sv_threshold_grid.py` |

### Adaptive RMT control (Section "Budget allocation")

The `adaptive_rmt/` package implements layer-aware RMT budget allocation used by the queue runners.

| Module | Role |
|---|---|
| `config.py` | All knobs (sparsity schedule, FT epochs, restore budgets, layer typing). |
| `controller.py` | Top-level cycle controller: prune → reinsert → fine-tune. |
| `prune.py` | Prune step (calls into `pod_src.RMT` and `pod_src.pruning`). |
| `signals.py` | Per-layer RMT signals (\(\sigma_+\), Hill \(\alpha\), MP score). |
| `rmt_diagnostics.py` | Randomness null tests, MP fit, bulk metrics. |
| `data.py` | Train / val DataLoader plumbing. |
| `finetune.py` | One-cycle frozen-mask fine-tune. |
| `model_utils.py` | timm load / save / mask helpers. |

### Per-architecture queue config

`model_queue_runs/queue_*/queue_guard.py` — one file per pod queue (a, b, c, d, e, f). These are watchdog scripts that the orchestrator calls; they decide whether to launch the next model in their bucket.

### Utility / inherited from the prior RMT-ViT repository

`src/` mirrors the `src/` directory of the prior *“Efficient Pruning of Vision Transformers using Random Matrix Theory”* code, which lives in a separate public repository: **https://github.com/yspennstate/RMT_pruning_ViT** (this repo). The two repositories are deliberately layered:

- The prior repo is the **single-architecture (ViT-Base) implementation** of the original Marchenko–Pastur–based pruning algorithm (Algorithm 3 of the prior paper). It is small, focused, and intended for someone who wants to reproduce the ViT-Base figure in 1–2 commands.
- The current repo is the **multi-architecture / multi-protocol successor.** It re-uses the prior repo’s Marchenko–Pastur, splittable-layer, and validation infrastructure unchanged, and builds the new protocols (SER, Hybrid Magnitude–SER, drop-threshold variant, SEB / S+) on top of that foundation.

The six modules in `src/` are used as a library by every new method:

| Module | Role | Status vs. prior repo |
|---|---|---|
| `RMT.py` | Marchenko–Pastur fit, \(\sigma_+\) edge, Tracy–Widom helpers. | Verbatim copy. |
| `SplittableLayers.py` | Linear/Conv layers that can be split into bulk + signal. | Verbatim copy. |
| `pruning.py` | Low-level prune / reinsert primitives. | Extended with reservoir / reinsert helpers used by SER (~ +50 lines, no behavioural change to the original primitives). |
| `training.py` | Standard training / fine-tune loop. | Verbatim copy. |
| `utils.py` | Misc helpers. | Verbatim copy. |
| `validation.py` | ImageNet validation pass. | Extended with the multi-resolution / mass-validation paths used by the multi-architecture sweep (~ +45 lines). The HuggingFace token, which was hard-coded in the prior repo, is now read from `$HF_TOKEN`. |

`prune.py` and `fine_tune.py` at the repository root are bit-identical with the prior repo (433 and 100 lines respectively). They provide a one-command reproduction of the **original** ViT-Base figure (the same plot reproduced in the prior repo's README), which serves as the baseline curve in Table 1 of the new paper.

### Helpers

| File | Role |
|---|---|
| `build_model_rmt_cache.py` | Pre-compute per-layer SVD cache for a timm checkpoint (one-time cost, then reused by every method). |
| `download_train.py`, `download_val_only.py` | Download ImageNet-1k from HuggingFace into `$HF_CACHE`. Require `HF_TOKEN`. |
| `fine_tune.py` | Stand-alone fine-tune (used as a smoke test). |
| `parse_log_to_json.py` | Convert the runners’ stdout logs into the cycle-by-cycle JSONs that populate Table 2. |
| `direct_run_watchdog.py` | Watchdog that restarts a stalled run. |
| `queue_continue_on_complete.py`, `queue_switch_after_current.py`, `queue_watchdog.py` | Queue management helpers. |
| `scripts/*.bat`, `scripts/*.sh` | One-shot launchers used during development. The `.bat` files target Windows pods, the `.sh` files target Linux. |

---

## Quick start (single architecture, result method)

```bash
# 1. Install
python -m pip install -r requirements.txt

# 2. Set HuggingFace token (any token with read access)
export HF_TOKEN=hf_yourTokenHere

# 3. Where to store ImageNet & cache
export HF_HUB_CACHE=/path/to/hf_cache
export RMT_OPTUNA_RUN=$(pwd)/optuna_run     # results go here

# 4. Pre-compute the per-layer RMT cache for ViT-B/16
python build_model_rmt_cache.py --timm-checkpoint vit_base_patch16_224.augreg2_in21k_ft_in1k

# 5. Run the Hybrid Magnitude–SER protocol
python hybrid_mag20_then_v8_model.py \
    --timm-checkpoint vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --target-sparsities 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 \
    --output-dir $RMT_OPTUNA_RUN/randomness_audit_results_my_run
```

The runner writes one JSON per cycle into the output directory; concatenate them with `python parse_log_to_json.py` to get the row of Table 2 for that architecture.

The same protocol has been validated on the following architectures, covering plain transformers, hierarchical / windowed transformers, hybrid models, and convolutional networks:

- **ViT** — ViT-Small/16, ViT-B/16, ViT-B/16/384, ViT-Large/16
- **DeiT** — DeiT-Tiny, DeiT-Small, DeiT-Base
- **Swin** — Swin-Tiny, Swin-Small, Swin-Base/384
- **ConvNeXt / ConvNeXtV2** — ConvNeXt-Base, ConvNeXtV2-Base
- **Hiera** — Hiera-Base+
- **ResNet** — ResNet18, ResNet34, ResNet50 (`tv_in1k`), ResNet50d, ResNet101d

All baselines are pulled from `timm` by their canonical checkpoint name (e.g. `deit_small_patch16_224.fb_in1k`, `swin_small_patch4_window7_224.ms_in1k`, `convnext_base.fb_in22k_ft_in1k`, `hiera_base_plus_224.mae_in1k_ft_in1k`). To reproduce **all rows** of Table 2 in the paper, point the queue runner (`model_queue_runner.py`) at the timm checkpoints above one at a time, or in parallel on separate GPUs.

## Reproducing the drop-threshold variant rows

```bash
python hybrid_mag_until_drop_then_v8_model.py \
    --timm-checkpoint resnet50.tv_in1k \
    --drop-threshold-pp 0.7 \
    --output-dir $RMT_OPTUNA_RUN/randomness_audit_results_my_drop_run
```

The runner writes a `magnitude_until_drop_meta.json` recording the transition sparsity (where the 0.7 pp drop is exceeded) and an extended results.json with `source_method=classical_magnitude_until_drop` for stage-1 cycles and the SER-default tag for stage-2 cycles.

## Compute requirements

Each model in Table 2 was run on a single A40 (48 GB) Runpod instance. ViT-Large/16 runs on the same A40 with reduced batch size; everything else fits at the queue's default batch. ImageNet-1k is loaded from HuggingFace; the validated cycles in the paper used the standard 1.28M-image train set and 50K-image validation set.

A complete s=0.05 → 0.70 schedule (14 cycles, including stage-1 magnitude prefix and stage-2 SER) takes roughly:

| Architecture | Wall time on 1× A40 |
|---|---|
| DeiT-Tiny, ResNet18, ResNet34 | 6–8 hours |
| DeiT-Small, Swin-Tiny, ResNet50, ResNet50d | 12–16 hours |
| DeiT-Base, ConvNeXt-Base, Hiera-Base+, ResNet101d | 24–36 hours |
| ViT-B/16, Swin-Small | 36–48 hours |
| ViT-B/16/384, ViT-Large/16 | 60–96 hours |

## Repository layout

```
.
├── README.md                       — this file
├── requirements.txt                — Python dependencies
├── LICENSE                         — MIT
├── src/                            — all method implementations
├── adaptive_rmt/               — adaptive RMT package (layer-aware budget)
├── src/                    — RMT/pruning utilities (re-used from prior repo)
├── model_queue_runs/           — per-queue watchdog configs
├── hybrid_mag20_then_v8*.py    — Hybrid Magnitude–SER (result)
├── hybrid_mag_until_drop*.py   — Drop-threshold variant
├── magnitude_rmt_sweep.py      — Classical magnitude baseline sweep
├── prune.py                    — Classical RMT (cycle implementation)
├── theory_pruning.py           — Theory-driven RMT prune (negative control)
├── randomness_diagnostic.py    — RMT randomness null tests
├── haar_*.py                   — SEB / S+ Haar bulk optimizer
├── hp_search*.py               — SEB hyperparameter grid search
├── iterative_*.py              — iterative prune-restore variants
├── sv_*.py, spectral_denoise_test.py — SV preprocessing tests
├── run_finetune_*.py           — magnitude / pipeline fine-tune drivers
├── run_removed_matrix_audit_*  — SER ablation cycles
├── ...                             — utility scripts (build cache, parse logs, watchdogs)
├── scripts/                        — one-shot .bat / .sh launchers
└── configs/                        — (placeholder; runtime configs live alongside the runners)
```

## Citation

Please cite the paper when using this code:

```bibtex
@article{berlyand2026pruning,
  title  = {Pruning Deep Neural Networks via Random Matrix Theory},
  author = {Berlyand, Leonid and Bourdais, Theo and Owhadi, Houman and Shmalo, Yitzchak},
  year   = {2026},
  note   = {Manuscript title: ``Pruning Deep Neural Networks via the Marchenko--Pastur Distribution''},
}
```

## License

MIT. See `LICENSE`.

## Notes on credentials

This repository is a clean public mirror of the live research code. All HuggingFace tokens, pod IPs, SSH keys, and cluster paths have been stripped. Any path that previously pointed into the pod filesystem (`/workspace/rmt_vit_pruning/optuna_run/...`) now reads from the `$RMT_OPTUNA_RUN` and `$RMT_CACHE` environment variables, with sensible defaults that work in a fresh checkout.

Report any leaked secret in the repository history immediately. The maintainers will rotate the credential and force-push a sanitized history.

## Data and checkpoint availability

Checkpoints and per-run evidence files are released here:

**Google Drive folder:** <https://drive.google.com/drive/folders/1mm990SHAHlYdISHxirvMRdVQEAjpIxDd>

The released Drive bundles include:
- The 17 Hybrid Magnitude-SER sparsification checkpoints for the multi-architecture sweep: per-architecture grids over $s\in[0.05, 0.70]$, about 230 post-FT `.pt` files, 68,935,892,958 bytes.
- FLOP-model checkpoints where available plus `final_eval.json` logs for the bundled `tab:param_to_flop_followup` rows: ViT-L 8:16 dense+perm 85.33%, ViT-B 6:12 SER+alpha=0.5 83.74%, and ResNet50/50d/101d 8:16 dense+perm 75.87/78.57/80.92%.
- SER source checkpoints at $s\!=\!0.35$ (ViT-B, ViT-L, ConvNeXt-Base, ResNet50/50d/101d/152d).
- 282-cell certificate-audit CSV/JSON files, mask-statistics JSON snapshots, and per-run training logs.

See `data/release_manifest.json` for the machine-readable mapping. It records zip-relative evidence paths and checkpoint availability, including that the ViT-L 8:16 dense+perm row includes `final_eval.json`, `student_pre_ft.pt`, `student_ep1.pt`, `student_ep2.pt`, and recipe/logs, but not final post-FT weights.

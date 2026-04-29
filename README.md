# RMT-Based Pruning in Deep Learning

Reproduction code for the paper **"Spectral Edge Reallocation: Random Matrix Theory–Guided Pruning of Vision Transformers and Convolutional Networks."**

The repository contains every method evaluated in the paper:

- **Classical magnitude pruning** — global unstructured baseline.
- **Classical RMT pruning** — Marchenko–Pastur (MP) edge fitting + sub-edge bulk denoising (negative control).
- **Spectral Edge Budgeting (SEB / S+)** — no-fine-tuning, layerwise magnitude budget allocated by the fitted MP edge \(\sigma_+(\ell)\) and Hill exponent \(\alpha_\ell\).
- **Spectral Edge Reallocation (SER)** — main RMT prune-restore method: prune beyond the target, form a reservoir, reinsert an RMT-ranked budget, run a short frozen-mask fine-tuning phase.
- **Hybrid Magnitude–SER** — exact global magnitude pruning through \(s = 0.20\), then SER for \(s \ge 0.25\). This is the headline protocol of Tables 1 and 2 in the paper.
- **Drop-threshold variant** — stage-1 magnitude continues until post–FT top-1 drops by more than 0.7 percentage points, then stage-2 (SER) begins. Two flavors:
  - **From-scratch**: stage-1 starts from the dense model.
  - **Seeded**: stage-1 starts from the canonical \(s=0.20\) checkpoint and continues magnitude past it.

Code currently used to produce the validated checkpoints in the paper is preserved as-is — file names retain their internal ("v8", "until_drop", etc.) lineage, but the table below maps every file to the paper-facing method name.

---

## File map (paper method → code)

### Pipeline entry points

| Paper method | File | Notes |
|---|---|---|
| **Hybrid Magnitude–SER**, single architecture | `src/hybrid_mag20_then_v8.py` | Run for one architecture, full s=0.05–0.70 schedule. |
| **Hybrid Magnitude–SER**, single model with `--model` arg | `src/hybrid_mag20_then_v8_model.py` | Same protocol, parameterized by timm checkpoint name. |
| **Hybrid Magnitude–SER**, multi-arch queue runner | `src/hybrid_mag20_then_v8_model_queue.py` | Used to drive the multi-architecture rows of Table 2 (one process per pod queue). |
| **Drop-threshold variant**, single model | `src/hybrid_mag_until_drop_then_v8_model.py` | Stops stage-1 when post-FT top-1 drops > 0.7 pp. |
| **Drop-threshold variant**, queue runner | `src/hybrid_mag_until_drop_then_v8_model_queue.py` | Multi-arch driver for the variant. |
| **Multi-arch orchestrator** (calls the queue runners) | `src/model_queue_runner.py` | Top-level loop that selects the next model from the per-queue config. |
| **Multi-arch launcher** (interactive entry) | `src/start_model_queue.py` | Spawns or resumes a queue. |

### Classical magnitude pruning (Appendix “S+ method”)

| Paper method | File |
|---|---|
| Classical magnitude pruning sweep | `src/magnitude_rmt_sweep.py` |
| Classical magnitude + RMT comparison sweep | `src/rmt_magnitude_sweep.py` |
| Classical magnitude vs uniform-budget comparison | `src/rmt_vs_uniform.py` |
| Magnitude pre-fine-tune helpers | `src/run_finetune_magnitude.py` (and `_v3.py`, `_v4.py`) |
| Magnitude pre-fine-tune for a specific timm checkpoint | `src/run_finetune_magnitude_model_exec.py` |
| Magnitude pre-fine-tune queue driver | `src/run_finetune_magnitude_model_exec_queue.py` |
| End-to-end fine-tune pipeline | `src/run_finetune_pipeline.py` (and `_v2.py`) |

### Classical RMT pruning (Appendix “Classical RMT”)

| Paper method | File |
|---|---|
| Classical RMT cycle implementation | `src/prune.py` |
| Theory-driven RMT prune (negative control) | `src/theory_pruning.py` |
| Per-layer RMT diagnostics + randomness null tests | `src/randomness_diagnostic.py` |

### Spectral Edge Budgeting (SEB / "S+" with Haar bulk model)

| Paper method | File |
|---|---|
| Haar/MP optimizer for the SEB hyperparameters \((\beta, s_\mathrm{d}, p)\) | `src/haar_optuna.py` |
| Refined Haar optimizer (later sweep iterations) | `src/haar_optuna_refined.py` |
| Haar small-z search (low-z regime) | `src/haar_small_z.py` |
| Hyperparameter grid search v1 | `src/hp_search.py` |
| Hyperparameter grid search v2 | `src/hp_search_v2.py` |
| Iterative 5%-step comparison | `src/iterative_5pct_compare.py` |
| Iterative “growing a” schedule | `src/iterative_growing_a.py` |
| SV-decides-when-to-stop iterative variant | `src/iterative_sv_decides.py` |
| Sweep analyzer | `src/analyze_sweep.py` |
| Long sweep driver | `src/overnight_grid_search.py` |
| Alt sweep driver | `src/overnight_sweep.py` |

### Spectral Edge Reallocation (SER) ablations

| Paper method | File |
|---|---|
| Removed-matrix audit (v5 cycle of SER) | `src/run_removed_matrix_audit_v5_model_exec.py` |
| Removed-matrix audit (v8 cycle of SER, current) | `src/run_removed_matrix_audit_v8_model_exec.py` |
| Resume a partial SER run | `src/remote_resume_suffix.py` |
| Status snapshot | `src/remote_status_once.py` |
| Resume supervisor | `src/remote_suffix_supervisor.py` |

### Singular-value preprocessing tests (Appendix “SV preprocessing”)

| Paper method | File |
|---|---|
| Spectral denoise prune-only test | `src/spectral_denoise_test.py` |
| SV pruning test | `src/sv_pruning_test.py` |
| SV theory probe | `src/sv_theory_test.py` |
| SV hyperparameter grid | `src/sv_hp_grid.py` |
| SV power-law grid | `src/sv_power_grid.py` |
| SV threshold grid | `src/sv_threshold_grid.py` |

### Adaptive RMT control (Section "Budget allocation")

The `src/adaptive_rmt/` package implements layer-aware RMT budget allocation used by the queue runners.

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

`src/model_queue_runs/queue_*/queue_guard.py` — one file per pod queue (a, b, c, d, e, f). These are watchdog scripts that the orchestrator calls; they decide whether to launch the next model in their bucket.

### Utility / inherited from the prior RMT-ViT repository

`src/pod_src/` mirrors the `src/` directory of the earlier *“Efficient Pruning of Vision Transformers using Random Matrix Theory”* code (separate repo: https://github.com/yspennstate/RMT_pruning_ViT). It is used as a library by the new methods above:

| Module | Role |
|---|---|
| `RMT.py` | Marchenko–Pastur fit, σ+ edge, Tracy-Widom helpers. |
| `SplittableLayers.py` | Linear/Conv layers that can be split into bulk + signal. |
| `pruning.py` | Low-level prune / reinsert primitives. |
| `training.py` | Standard training / fine-tune loop. |
| `utils.py` | Misc helpers. |
| `validation.py` | ImageNet validation pass. |

### Helpers

| File | Role |
|---|---|
| `src/build_model_rmt_cache.py` | Pre-compute per-layer SVD cache for a timm checkpoint (one-time cost, then reused by every method). |
| `src/download_train.py`, `src/download_val_only.py` | Download ImageNet-1k from HuggingFace into `$HF_CACHE`. Require `HF_TOKEN`. |
| `src/fine_tune.py` | Stand-alone fine-tune (used as a smoke test). |
| `src/parse_log_to_json.py` | Convert the runners’ stdout logs into the cycle-by-cycle JSONs that populate Table 2. |
| `src/direct_run_watchdog.py` | Watchdog that restarts a stalled run. |
| `src/queue_continue_on_complete.py`, `src/queue_switch_after_current.py`, `src/queue_watchdog.py` | Queue management helpers. |
| `scripts/*.bat`, `scripts/*.sh` | One-shot launchers used during development. The `.bat` files target Windows pods, the `.sh` files target Linux. |

---

## Quick start (single architecture, headline method)

```bash
# 1. Install
python -m pip install -r requirements.txt

# 2. Set HuggingFace token (any token with read access)
export HF_TOKEN=hf_yourTokenHere

# 3. Where to store ImageNet & cache
export HF_HUB_CACHE=/path/to/hf_cache
export RMT_OPTUNA_RUN=$(pwd)/optuna_run     # results go here

# 4. Pre-compute the per-layer RMT cache for ViT-B/16
python src/build_model_rmt_cache.py --timm-checkpoint vit_base_patch16_224.augreg2_in21k_ft_in1k

# 5. Run the headline Hybrid Magnitude–SER protocol
python src/hybrid_mag20_then_v8_model.py \
    --timm-checkpoint vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --target-sparsities 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 \
    --output-dir $RMT_OPTUNA_RUN/randomness_audit_results_my_run
```

The runner writes one JSON per cycle into the output directory; concatenate them with `python src/parse_log_to_json.py` to get the row of Table 2 for that architecture.

To reproduce **all rows** of Table 2, point the queue runners at multiple timm checkpoints — the queues used in the paper live in `src/model_queue_runs/queue_*/queue_guard.py`. The queue mapping for the paper run is:

| Queue | Architecture |
|---|---|
| queue_a | DeiT-Small, ResNet50 (tv_in1k), ResNet50d, ViT-Small/16 |
| queue_b | DeiT-Base, ViT-B/16/384 |
| queue_c | ViT-Large/16 |
| queue_d | Swin-Tiny, Swin-Small |
| queue_e | Swin-Base/384 |
| queue_f | ConvNeXt-Base, ConvNeXtV2-Base |
| queue_h | Hiera-Base+ |
| queue_h_lit | ResNet50 (tv_in1k) replica |
| queue_v11_resnet | ResNet101d |
| direct_deit_followup | ResNet18, ResNet34 |

All baseline architectures are pulled from `timm` by their canonical checkpoint name (e.g. `deit_small_patch16_224.fb_in1k`, `swin_small_patch4_window7_224.ms_in1k`, `convnext_base.fb_in22k_ft_in1k`, `hiera_base_plus_224.mae_in1k_ft_in1k`).

## Reproducing the drop-threshold variant rows

```bash
python src/hybrid_mag_until_drop_then_v8_model.py \
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
│   ├── adaptive_rmt/               — adaptive RMT package (layer-aware budget)
│   ├── pod_src/                    — RMT/pruning utilities (re-used from prior repo)
│   ├── model_queue_runs/           — per-queue watchdog configs
│   ├── hybrid_mag20_then_v8*.py    — Hybrid Magnitude–SER (headline)
│   ├── hybrid_mag_until_drop*.py   — Drop-threshold variant
│   ├── magnitude_rmt_sweep.py      — Classical magnitude baseline sweep
│   ├── prune.py                    — Classical RMT (cycle implementation)
│   ├── theory_pruning.py           — Theory-driven RMT prune (negative control)
│   ├── randomness_diagnostic.py    — RMT randomness null tests
│   ├── haar_*.py                   — SEB / S+ Haar bulk optimizer
│   ├── hp_search*.py               — SEB hyperparameter grid search
│   ├── iterative_*.py              — iterative prune-restore variants
│   ├── sv_*.py, spectral_denoise_test.py — SV preprocessing tests
│   ├── run_finetune_*.py           — magnitude / pipeline fine-tune drivers
│   ├── run_removed_matrix_audit_*  — SER ablation cycles
│   └── ...                         — utility scripts (build cache, parse logs, watchdogs)
├── scripts/                        — one-shot .bat / .sh launchers
└── configs/                        — (placeholder; runtime configs live alongside the runners)
```

## Citation

If you use this code please cite the paper:

```bibtex
@article{ser_pruning_2026,
  title  = {Spectral Edge Reallocation: Random Matrix Theory--Guided Pruning of Vision Transformers and Convolutional Networks},
  author = {Authors},
  year   = {2026},
}
```

## License

MIT. See `LICENSE`.

## Notes on credentials

This repository is a clean public mirror of the live research code. All HuggingFace tokens, pod IPs, SSH keys, and cluster paths have been stripped. Any path that previously pointed into the pod filesystem (`/workspace/rmt_vit_pruning/optuna_run/...`) now reads from the `$RMT_OPTUNA_RUN` and `$RMT_CACHE` environment variables, with sensible defaults that work in a fresh checkout.

If you discover a leaked secret in the repository history, please open an issue immediately — we will rotate the credential and force-push a sanitized history.

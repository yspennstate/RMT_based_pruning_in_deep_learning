# Post-FT checkpoint archive - Google Drive plan

> Release note (2026-05-11): this file is a historical staging plan. The
> authoritative public-release mapping is `data/release_manifest.json`, which
> records the released Drive bundles, zip roots, row evidence, and checkpoint
> availability.

The CAST-2E experimental pipeline produces post-fine-tuning checkpoints that are
**not stored in this Git repository** because each `.pt` file is 100 MB – 1.5 GB
and the total exceeds GitHub's per-repo limits. Instead, they are mirrored to
Google Drive (matching the Hybrid Magnitude–SER 17-model archive at
<https://drive.google.com/drive/folders/1mm990SHAHlYdISHxirvMRdVQEAjpIxDd>) for
external auditability.

## Historical local staging

The local archive at `~/Downloads/cast_2e_resnet_review/checkpoints_for_drive/`
contains symbolic links to every paper-result post-FT checkpoint. Sizes given
are the dereferenced sizes (links resolve to the actual .pt files in
`~/Downloads/cast_2e_resnet_review/github_archive/`).

| Subdirectory | Contents | Approximate size | Paper result number |
|---|---|---|---|
| `vitl_canonical/` | ViT-L 2:4 + ToMe-r=8 epochs 1–3 + final + pre-FT + teacher dense | ~16 GB | 84.37% (Tab. param_to_flop_followup) |
| `resnet50.tv_in1k_8_16/` | ResNet50 8:16 cert+perm epochs 1–3 + final | 392 MB | 75.87% (NEW) |
| `4resnet_2_4_cert_1_resnet50.tv_in1k/` | ResNet50 1×1+3×3 2:4 cert+perm epochs 1–3 + final | ~400 MB | 75.67% |
| `4resnet_2_4_cert_2_resnet50d.ra2_in1k/` | ResNet50d 2:4 cert+perm | ~400 MB | 78.00% |
| `4resnet_2_4_cert_3_resnet101d.ra2_in1k/` | ResNet101d 2:4 cert+perm | ~600 MB | 80.59% |
| `4resnet_2_4_cert_4_resnet152d.ra2_in1k/` | ResNet152d 2:4 cert+perm | ~900 MB | 81.33% |
| `convnextv2_canonical/` | ConvNeXtV2 2:4 cert + free-restore (post-FT + pre-FT) | ~420 MB | 85.47% |
| `convnextv2_d1216/` | ConvNeXtV2 12:16 dense+perm (post-FT) | ~339 MB | 86.35% |
| `convnextv2_d816/` | ConvNeXtV2 8:16 dense+perm (post-FT) | ~339 MB | 85.85% |

**Total staging size: ~20 GB.** Each entry has its `final_eval.json`
(or `results.json`) packaged alongside so readers can match a checkpoint to
its evaluation metric without re-running.

## Historical pending runs at staging time

| Run | Pod | ETA | Will land at |
|---|---|---|---|
| ViT-L 8:16 dense+perm inline FT | Pod 1 (12896) | ~12 hr | `vitl_8_16/` |
| ConvNeXtV2 D48 4:8 dense+perm cert + 3-ep FT | Pod 2 (12632) | ~1.5 hr | `convnextv2_d48/` |
| 4-ResNet 8:16 chain (50d, 101d, 152d) | Pod 3a (12059) | ~21 hr | `resnet*_8_16/` |

At staging time, the cron health check pulled JSON eval files first and pulled
`.pt` files into `checkpoints_for_drive/` in a following pass.

## Historical upload procedure

```bash
# rsync-mirror the staging dir to Google Drive via rclone
rclone copy -v --transfers 4 \
    ~/Downloads/cast_2e_resnet_review/checkpoints_for_drive/ \
    gdrive:RMT_pruning_CAST_2E_checkpoints/
```

The README of each subdirectory contains the post-FT top-1, the cert
configuration, the FT hyperparameters, the train/eval epoch records, and the
exact `git rev-parse HEAD` of the producing run. Together with the
corresponding `cast_2e/code/run_*_ft_inline.py` script, a subdirectory
provides the artifacts needed to reproduce a single paper-result number.

## Provenance audit

For every completed post-FT checkpoint in the staging table above:
- `final_eval.json` records: pre-FT top-1, teacher top-1, post-FT top-1,
  delta_pre_to_post_pp, delta_vs_teacher_pp, ckpt_avg_sparsity, n_layers_with_mask,
  per-epoch (train_loss, val_top1, elapsed_s)
- `student_pre_ft.pt` records the projected state-dict + cert metadata before FT
- `student_ep1.pt`, `student_ep2.pt`, `student_ep3.pt` are the per-epoch snapshots
- `student_final.pt`, where present, is the post-FT model (top-1 = post_ft_top1 from the JSON)

This is the same provenance schema as the Hybrid Magnitude–SER 17-model archive.

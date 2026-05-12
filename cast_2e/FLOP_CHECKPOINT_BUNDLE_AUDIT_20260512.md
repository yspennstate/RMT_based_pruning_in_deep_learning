# FLOP Checkpoint Bundle Audit

Audit date: 2026-05-12.

Public Drive folder inspected:
`https://drive.google.com/drive/folders/1mm990SHAHlYdISHxirvMRdVQEAjpIxDd`.

Observed uploaded bundles:

- `rmt_paper_flop_model_data_20260511.zip`, shown by Drive as modified 2026-05-11 with size 6.22 GB/GiB.
- `rmt_pruning_sparsification_17_model_checkpoints_20260502.zip`, shown by Drive as modified 2026-05-09 with size 64.2 GB/GiB. This appears to correspond to the local `rmt_pruning_paper_17_model_checkpoints_20260502.zip` archive, whose size is 68,935,892,958 bytes.

The FLOP bundle opens locally and contains loadable checkpoints for the ViT-B 6:12 row, the ViT-L 8:16 epoch-2 row, and the ResNet 8:16 final rows. It also contains the expected `final_eval.json` files for those rows.

Important caveats:

- The uploaded FLOP bundle does not contain the canonical native-2:4 ViT-B, ViT-L, DeiT-B/S/T, or ConvNeXtV2 final checkpoints.
- The ResNet 2:4-chain checkpoint files inside the uploaded FLOP bundle under `pod1_4resnet_2_4_chain/.../student_pre_ft.pt` and `checkpoints/latest_step.pt` are not loadable by `torch.load` on this machine.
- Valid final ResNet 2:4 checkpoints do exist locally under `C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\...\epoch3.pt`.
- The ViT-L 8:16 dense+perm final epoch-3 checkpoint remains unavailable locally; the bundle has `final_eval.json`, pre-FT weights, and epoch-1/epoch-2 checkpoints.
- The ViT-B magnitude 2:4+ToMe final post-FT checkpoint remains unavailable locally; only a stage-1/pre-FT artifact is located.
- ConvNeXtV2 8:16 and 12:16 final local checkpoints load, but the tensor audit records zero sparsity in the saved weights. They should remain accuracy/MAC-accounting rows unless a mask/runtime artifact is supplied.

Created supplemental artifact:

- `C:\Users\owner\Downloads\rmt_paper_flop_missing_checkpoint_supplement_20260512.zip`
- Size: 4,300,989,937 bytes
- SHA-256: `2721a063dfc4d40c13957c6afde22fd6b8071c4c8c0e6e440b6399985669102c`

The supplement contains valid local final checkpoints for the missing canonical native-2:4 rows, valid final ResNet 2:4 checkpoints, ConvNeXtV2 8:16/12:16 accuracy checkpoints, and audit metadata. It is provenance only; it does not change weights or validation results.

Machine-readable audit files:

- `data/flop_checkpoint_bundle_audit_20260513.json`
- `data/flop_checkpoint_bundle_audit_20260513.csv`
- `data/release_manifest.json`

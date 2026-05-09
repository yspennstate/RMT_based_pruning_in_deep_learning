#!/usr/bin/env bash
# Re-run ResNet50 round-1 + round-2 with the fixed project_kn_sparsity.py.
# Original run had compute_cin_permutation kwarg bug → all perm cells failed.

set +e
exec > /workspace/run_logs/pod2_resnet_rerun.log 2>&1

VAL=/workspace/val_eval/val
CALIB=/workspace/val_eval/val
CK_RN50=/workspace/sweep_ckpts/resnet50.tv_in1k_keep_s35_classmag.pt

echo "=== Pod 2 ResNet50 RE-RUN start $(date -u) ==="

# Wait until round-2 ConvNeXtV2-B sweep finishes (it's the slowest, runs last).
# We don't want to fight with it for GPU bandwidth.
# Actually, GPU has 70+ GB free; just run in parallel.

echo "--- ResNet50 round-1 (13 cells, fixed code) ---"
python -u /workspace/code/cert_opt_eval_kn.py \
    --timm-name resnet50.tv_in1k --ser-checkpoint "$CK_RN50" \
    --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
    --output /workspace/run_logs/resnet50_kn_results_fixed.json --num-workers 4 \
    2>&1 | tee /workspace/run_logs/resnet50_kn_run_fixed.log
echo "[R1] exit=$?"

echo "--- ResNet50 round-2 extended (50 cells, fixed code) ---"
python -u /workspace/code/cert_opt_eval_kn_extended.py \
    --timm-name resnet50.tv_in1k --ser-checkpoint "$CK_RN50" \
    --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
    --output /workspace/run_logs/resnet50_kn_extended_fixed.json --num-workers 4 \
    2>&1 | tee /workspace/run_logs/resnet50_kn_extended_fixed.log
echo "[R2] exit=$?"

# Bonus round 4: ConvNeXtV2-B extended after ResNet50 done (fixes any prior
# convnextv2 perm failures from the same bug, since cert_aware_kn_for_conv
# is the path)
CK_CNXTV2=/workspace/sweep_ckpts/convnextv2_base.fcmae_ft_in22k_in1k_keep_s35.pt
echo "--- ConvNeXtV2-B extended re-run (50 cells, fixed code) ---"
python -u /workspace/code/cert_opt_eval_kn_extended.py \
    --timm-name convnextv2_base.fcmae_ft_in22k_in1k --ser-checkpoint "$CK_CNXTV2" \
    --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
    --output /workspace/run_logs/convnextv2_kn_extended_fixed.json --num-workers 4 \
    2>&1 | tee /workspace/run_logs/convnextv2_kn_extended_fixed.log
echo "[R3] exit=$?"

echo "=== Pod 2 ResNet50 RE-RUN done $(date -u) ==="

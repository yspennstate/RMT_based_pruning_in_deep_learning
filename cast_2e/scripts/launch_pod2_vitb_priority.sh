#!/usr/bin/env bash
# Priority ViT-B sweep — generate ckpt + run round-1 + round-2 extended.
# User wanted ViT-B (not ViT-S). Runs in parallel with ResNet50 re-run.

set +e
exec > /workspace/run_logs/pod2_vitb_priority.log 2>&1

VAL=/workspace/val_eval/val
CALIB=/workspace/val_eval/val
CK_VITB=/workspace/sweep_ckpts/vit_base_patch16_224.augreg2_in21k_ft_in1k_keep_s35_classmag.pt

echo "=== Pod 2 ViT-B priority sweep $(date -u) ==="

# 0. Generate ViT-B ckpt if missing
if [ ! -f "$CK_VITB" ]; then
    echo "--- generating ViT-B/16 224 Classical Magnitude SER s=0.35 ckpt ---"
    python -u /workspace/code/quick_prune_vitb224.py \
        --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k --target-sparsity 0.35 \
        --output "$CK_VITB"
fi

# 1. ViT-B round 1 (15 cells)
echo "--- ViT-B round 1 (15 cells) ---"
python -u /workspace/code/cert_opt_eval_vitb_kn.py \
    --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k --ser-checkpoint "$CK_VITB" \
    --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
    --output /workspace/run_logs/vitb_kn_results.json --num-workers 4 --batch-size-val 64 \
    2>&1 | tee /workspace/run_logs/vitb_kn_run.log
echo "[ViT-B R1] exit=$?"

# 2. ViT-B round 2 extended (50 cells)
echo "--- ViT-B round 2 extended (50 cells) ---"
python -u /workspace/code/cert_opt_eval_vitb_kn_extended.py \
    --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k --ser-checkpoint "$CK_VITB" \
    --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
    --output /workspace/run_logs/vitb_kn_extended.json --num-workers 4 --batch-size-val 64 \
    2>&1 | tee /workspace/run_logs/vitb_kn_extended.log
echo "[ViT-B R2] exit=$?"

# 3. Bonus: 5-seed replicate of ViT-B winner cell for variance bound
echo "=== Pod 2 ViT-B priority sweep DONE $(date -u) ==="

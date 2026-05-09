#!/usr/bin/env bash
# Pod 3a comprehensive sweep — ALL 3 archs (ResNet50, ConvNeXtV2-B, ViT-B)
# focused on the user's load-bearing question: does SER s=0.35 source HELP or
# HURT pre-FT preservation vs starting from dense?
#
# 6 sparsity patterns × 2 sources (dense, SER) × 6 α_ser values × ~4 perm/calib
# variants × 5-seed replicates of winners = ~150 cells per arch.

set +e
exec > /workspace/pod3a_comprehensive.log 2>&1
echo "=== Pod 3a comprehensive chain start $(date -u) ==="

# Wait for ImageNet val to be available (relay running in background from local)
while [ "$(ls /workspace/imagenet/val 2>/dev/null | wc -l)" -lt 1000 ]; do
    echo "[$(date -u)] waiting for ImageNet val (currently $(ls /workspace/imagenet/val 2>/dev/null | wc -l) classes)"
    sleep 30
done
echo "[$(date -u)] ImageNet val ready: $(ls /workspace/imagenet/val | wc -l) classes"

VAL=/workspace/imagenet/val
CALIB=/workspace/imagenet/val
mkdir -p /workspace/sweep_ckpts /workspace/run_logs

# Install deps
pip install -q --upgrade timm torchvision 2>&1 | tail -3

# ---- Generate SER ckpts ----
gen_ckpt () {
    local arch=$1; local out=$2; local kind=$3
    if [ ! -f "$out" ]; then
        echo "[ckpt] generating $arch -> $out (kind=$kind)"
        if [ "$kind" == "linear" ]; then
            python -u /workspace/code/quick_prune_vitb224.py \
                --timm-name "$arch" --target-sparsity 0.35 --output "$out" 2>&1 || echo "FAILED $arch"
        else
            python -u /workspace/code/quick_prune_resnet_magnitude.py \
                --timm-name "$arch" --target-sparsity 0.35 --output "$out" 2>&1 || echo "FAILED $arch"
        fi
    else
        echo "$out present"
    fi
}

CK_RN50=/workspace/sweep_ckpts/resnet50.tv_in1k_keep_s35.pt
CK_CNXTV2=/workspace/sweep_ckpts/convnextv2_base.fcmae_ft_in22k_in1k_keep_s35.pt
CK_VITB=/workspace/sweep_ckpts/vit_base_patch16_224.augreg2_in21k_ft_in1k_keep_s35.pt

gen_ckpt "resnet50.tv_in1k"                              "$CK_RN50"   "conv"
gen_ckpt "convnextv2_base.fcmae_ft_in22k_in1k"           "$CK_CNXTV2" "conv"
gen_ckpt "vit_base_patch16_224.augreg2_in21k_ft_in1k"    "$CK_VITB"   "linear"

ls -la /workspace/sweep_ckpts/

# === Phase 1: Extended sweeps for all 3 archs (~50 cells each) ===
echo "=== [Phase 1/3] ResNet50 extended (~50 cells) ==="
python -u /workspace/code/cert_opt_eval_kn_extended.py \
    --timm-name resnet50.tv_in1k --ser-checkpoint "$CK_RN50" \
    --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
    --output /workspace/run_logs/resnet50_kn_extended_pod3a.json --num-workers 4 \
    2>&1 | tee /workspace/run_logs/resnet50_pod3a_p1.log

echo "=== [Phase 1/3] ViT-B extended (~50 cells) ==="
python -u /workspace/code/cert_opt_eval_vitb_kn_extended.py \
    --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k --ser-checkpoint "$CK_VITB" \
    --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
    --output /workspace/run_logs/vitb_kn_extended_pod3a.json --num-workers 4 --batch-size-val 64 \
    2>&1 | tee /workspace/run_logs/vitb_pod3a_p1.log

echo "=== [Phase 1/3] ConvNeXtV2-B extended (~50 cells) ==="
python -u /workspace/code/cert_opt_eval_kn_extended.py \
    --timm-name convnextv2_base.fcmae_ft_in22k_in1k --ser-checkpoint "$CK_CNXTV2" \
    --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
    --output /workspace/run_logs/convnextv2_kn_extended_pod3a.json --num-workers 4 \
    2>&1 | tee /workspace/run_logs/convnextv2_pod3a_p1.log

# === Phase 1.5: ADVANCED cert methods (mixed-sparsity, iterative, robust ℓ_∞) ===
echo "=== [Phase 1.5/3] ResNet50 ADVANCED methods (~14 cells) ==="
python -u /workspace/code/cert_opt_eval_advanced.py \
    --timm-name resnet50.tv_in1k --ser-checkpoint "$CK_RN50" \
    --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
    --output /workspace/run_logs/resnet50_advanced_pod3a.json \
    --mask-save-dir /workspace/run_outputs/masks_resnet50 \
    --num-workers 4 \
    2>&1 | tee /workspace/run_logs/resnet50_advanced_pod3a.log

echo "=== [Phase 1.5/3] ConvNeXtV2-B ADVANCED methods (~14 cells) ==="
python -u /workspace/code/cert_opt_eval_advanced.py \
    --timm-name convnextv2_base.fcmae_ft_in22k_in1k --ser-checkpoint "$CK_CNXTV2" \
    --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
    --output /workspace/run_logs/convnextv2_advanced_pod3a.json \
    --mask-save-dir /workspace/run_outputs/masks_convnextv2 \
    --num-workers 4 \
    2>&1 | tee /workspace/run_logs/convnextv2_advanced_pod3a.log

# === Phase 2: 5-seed replicates of each arch (variance bound on winners) ===
echo "=== [Phase 2/3] 5-seed replicates ==="
for SEED in 1 2 3 4 5; do
    python -u /workspace/code/cert_opt_eval_kn_extended.py \
        --timm-name resnet50.tv_in1k --ser-checkpoint "$CK_RN50" \
        --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
        --output /workspace/run_logs/resnet50_kn_seed${SEED}_pod3a.json --num-workers 4 \
        2>&1 | tee /workspace/run_logs/resnet50_pod3a_seed${SEED}.log

    python -u /workspace/code/cert_opt_eval_vitb_kn_extended.py \
        --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k --ser-checkpoint "$CK_VITB" \
        --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
        --output /workspace/run_logs/vitb_kn_seed${SEED}_pod3a.json --num-workers 4 --batch-size-val 64 \
        2>&1 | tee /workspace/run_logs/vitb_pod3a_seed${SEED}.log
done

# === Phase 3: ConvNeXtV2 replicates (slower; do fewer seeds) ===
echo "=== [Phase 3/3] ConvNeXt replicates ==="
for SEED in 1 2 3; do
    python -u /workspace/code/cert_opt_eval_kn_extended.py \
        --timm-name convnextv2_base.fcmae_ft_in22k_in1k --ser-checkpoint "$CK_CNXTV2" \
        --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
        --output /workspace/run_logs/convnextv2_kn_seed${SEED}_pod3a.json --num-workers 4 \
        2>&1 | tee /workspace/run_logs/convnextv2_pod3a_seed${SEED}.log
done

echo "=== Pod 3a chain DONE $(date -u) ==="
ls -la /workspace/run_logs/*.json

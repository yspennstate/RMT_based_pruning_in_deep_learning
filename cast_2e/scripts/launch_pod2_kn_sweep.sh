#!/usr/bin/env bash
# Comprehensive k:n + dense-vs-SER sweep on Pod 2 IN PARALLEL with ViT-L FT.
# Pod 2 has full val ImageFolder at /workspace/val_eval/val (1000 classes).
# We use that for both --imagenet-val and --imagenet-train-for-calib (cert
# framework calibrates on activations, doesn't need GT labels).
#
# Sweeps:
#   3 archs: vit_small, resnet50, convnextv2_base
#   2 sources: dense, SER s=0.35
#   4 sparsity patterns: 2:4, 4:8, 1:4, 3:4
#   Multiple α_ser, perm variants per pattern (~15 cells per arch)
# Total ~45+ cells, ~3-5 hr.

# Don't bail on individual cell failures
set +e
exec > /workspace/run_logs/pod2_kn_sweep.log 2>&1
echo "=== Pod 2 k:n + dense/SER sweep start $(date -u) ==="

VAL=/workspace/val_eval/val
CALIB=/workspace/val_eval/val   # share — calib doesn't need labels

mkdir -p /workspace/sweep_ckpts /workspace/run_logs /workspace/run_outputs

gen_ckpt () {
    local arch=$1; local out=$2
    if [ ! -f "$out" ]; then
        echo "[ckpt] generating $arch -> $out"
        if [[ "$arch" == vit* ]]; then
            python -u /workspace/code/quick_prune_vitb224.py \
                --timm-name "$arch" --target-sparsity 0.35 --output "$out" \
                || echo "[ckpt] FAILED for $arch"
        else
            python -u /workspace/code/quick_prune_resnet_magnitude.py \
                --timm-name "$arch" --target-sparsity 0.35 --output "$out" \
                || echo "[ckpt] FAILED for $arch"
        fi
    else
        echo "[ckpt] $out already exists"
    fi
}

CK_RN50=/workspace/sweep_ckpts/resnet50.tv_in1k_keep_s35_classmag.pt
CK_VITS=/workspace/sweep_ckpts/vit_small_patch16_224.augreg_in21k_ft_in1k_keep_s35.pt
CK_CNXTV2=/workspace/sweep_ckpts/convnextv2_base.fcmae_ft_in22k_in1k_keep_s35.pt

gen_ckpt "resnet50.tv_in1k"                                "$CK_RN50"
gen_ckpt "vit_small_patch16_224.augreg_in21k_ft_in1k"      "$CK_VITS"
gen_ckpt "convnextv2_base.fcmae_ft_in22k_in1k"             "$CK_CNXTV2"

echo "=== ckpts ==="
ls -la /workspace/sweep_ckpts/

# ---- 1. ResNet50 kn sweep ----
echo "=== [1/3] ResNet50 kn sweep $(date -u) ==="
python -u /workspace/code/cert_opt_eval_kn.py \
    --timm-name resnet50.tv_in1k \
    --ser-checkpoint "$CK_RN50" \
    --imagenet-val "$VAL" \
    --imagenet-train-for-calib "$CALIB" \
    --output /workspace/run_logs/resnet50_kn_results.json \
    --num-workers 4 2>&1 | tee /workspace/run_logs/resnet50_kn_run.log
echo "[1/3] ResNet50 exit=$?"

# ---- 2. ViT-S kn sweep ----
echo "=== [2/3] ViT-S kn sweep $(date -u) ==="
python -u /workspace/code/cert_opt_eval_vitb_kn.py \
    --timm-name vit_small_patch16_224.augreg_in21k_ft_in1k \
    --ser-checkpoint "$CK_VITS" \
    --imagenet-val "$VAL" \
    --imagenet-train-for-calib "$CALIB" \
    --output /workspace/run_logs/vits_kn_results.json \
    --num-workers 4 --batch-size-val 64 2>&1 | tee /workspace/run_logs/vits_kn_run.log
echo "[2/3] ViT-S exit=$?"

# ---- 3. ConvNeXtV2-B kn sweep (Conv2d-based) ----
echo "=== [3/3] ConvNeXtV2-B kn sweep $(date -u) ==="
python -u /workspace/code/cert_opt_eval_kn.py \
    --timm-name convnextv2_base.fcmae_ft_in22k_in1k \
    --ser-checkpoint "$CK_CNXTV2" \
    --imagenet-val "$VAL" \
    --imagenet-train-for-calib "$CALIB" \
    --output /workspace/run_logs/convnextv2_kn_results.json \
    --num-workers 4 2>&1 | tee /workspace/run_logs/convnextv2_kn_run.log
echo "[3/3] ConvNeXtV2-B exit=$?"

echo "=== Pod 2 k:n sweep DONE $(date -u) ==="
ls -la /workspace/run_logs/*_kn_results.json

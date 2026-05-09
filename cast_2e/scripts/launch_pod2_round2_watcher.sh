#!/usr/bin/env bash
# Round-2 watcher: waits for current Pod 2 kn sweep (round 1) to finish, then
# runs the EXTENDED ~50-cell-per-arch sweep on all 3 archs. After round 2,
# runs round 3 (large-calib + ℓ_∞ + α_kd combinations).

set +e
exec > /workspace/run_logs/pod2_round2_watcher.log 2>&1

VAL=/workspace/val_eval/val
CALIB=/workspace/val_eval/val
CK_RN50=/workspace/sweep_ckpts/resnet50.tv_in1k_keep_s35_classmag.pt
CK_VITS=/workspace/sweep_ckpts/vit_small_patch16_224.augreg_in21k_ft_in1k_keep_s35.pt
CK_CNXTV2=/workspace/sweep_ckpts/convnextv2_base.fcmae_ft_in22k_in1k_keep_s35.pt

echo "=== Round-2 watcher start $(date -u) ==="
# Wait for round 1 sweep to finish
while pgrep -f "cert_opt_eval_kn.py\|cert_opt_eval_vitb_kn.py" > /dev/null; do
    if ! pgrep -f "cert_opt_eval_kn_extended\|cert_opt_eval_vitb_kn_extended" > /dev/null; then
        sleep 60
    else
        break  # extended already running
    fi
done

echo "=== round 1 done (or extended already running); starting round 2 $(date -u) ==="

# === Round 2: extended kn sweeps ===
echo "--- [R2-1/3] ResNet50 extended ---"
python -u /workspace/code/cert_opt_eval_kn_extended.py \
    --timm-name resnet50.tv_in1k --ser-checkpoint "$CK_RN50" \
    --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
    --output /workspace/run_logs/resnet50_kn_extended.json --num-workers 4 \
    2>&1 | tee /workspace/run_logs/resnet50_kn_extended.log
echo "[R2-1/3] exit=$?"

echo "--- [R2-2/3] ViT-S extended ---"
python -u /workspace/code/cert_opt_eval_vitb_kn_extended.py \
    --timm-name vit_small_patch16_224.augreg_in21k_ft_in1k --ser-checkpoint "$CK_VITS" \
    --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
    --output /workspace/run_logs/vits_kn_extended.json --num-workers 4 --batch-size-val 64 \
    2>&1 | tee /workspace/run_logs/vits_kn_extended.log
echo "[R2-2/3] exit=$?"

echo "--- [R2-3/3] ConvNeXtV2-B extended ---"
python -u /workspace/code/cert_opt_eval_kn_extended.py \
    --timm-name convnextv2_base.fcmae_ft_in22k_in1k --ser-checkpoint "$CK_CNXTV2" \
    --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
    --output /workspace/run_logs/convnextv2_kn_extended.json --num-workers 4 \
    2>&1 | tee /workspace/run_logs/convnextv2_kn_extended.log
echo "[R2-3/3] exit=$?"

echo "=== Round 2 DONE $(date -u) ==="

# === Round 3: ViT-B/16 224 (the canonical paper model) extended sweep ===
# We never ran ViT-B with the dense-vs-SER comparison; do that now for the paper.
CK_VITB=/workspace/sweep_ckpts/vit_base_patch16_224.augreg2_in21k_ft_in1k_keep_s35_classmag.pt
if [ ! -f "$CK_VITB" ]; then
    python -u /workspace/code/quick_prune_vitb224.py \
        --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k --target-sparsity 0.35 \
        --output "$CK_VITB" 2>&1 || echo "vit_base ckpt gen failed"
fi
echo "--- [R3-1/2] ViT-B/16 224 extended ---"
python -u /workspace/code/cert_opt_eval_vitb_kn_extended.py \
    --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k --ser-checkpoint "$CK_VITB" \
    --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
    --output /workspace/run_logs/vitb_kn_extended.json --num-workers 4 --batch-size-val 64 \
    2>&1 | tee /workspace/run_logs/vitb_kn_extended.log
echo "[R3-1/2] exit=$?"

# === Round 3 part 2: deit_base — another ViT family for cross-arch coverage ===
CK_DEITB=/workspace/sweep_ckpts/deit_base_patch16_224_keep_s35.pt
if [ ! -f "$CK_DEITB" ]; then
    python -u /workspace/code/quick_prune_vitb224.py \
        --timm-name deit_base_patch16_224.fb_in1k --target-sparsity 0.35 \
        --output "$CK_DEITB" 2>&1 || echo "deit_base ckpt gen failed"
fi
echo "--- [R3-2/2] DeiT-Base extended ---"
python -u /workspace/code/cert_opt_eval_vitb_kn_extended.py \
    --timm-name deit_base_patch16_224.fb_in1k --ser-checkpoint "$CK_DEITB" \
    --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
    --output /workspace/run_logs/deitb_kn_extended.json --num-workers 4 --batch-size-val 64 \
    2>&1 | tee /workspace/run_logs/deitb_kn_extended.log
echo "[R3-2/2] exit=$?"

echo "=== Round 3 DONE $(date -u) ==="
ls -la /workspace/run_logs/*_kn_*.json
echo "=== Pod 2 watcher chain DONE $(date -u) ==="

#!/usr/bin/env bash
# After ViT-B D612 SER+alpha=0.5 inline FT finishes on Pod 1:
#   start ViT-L 8:16 SER+alpha=0.5 inline FT (best ViT method on ViT-L)
set -e

while ! grep -q "FT COMPLETE" /workspace/run_logs/vitb_ft_inline_D612_ser_a05.log 2>/dev/null; do
    if [[ -f /tmp/proceed_pod1_vitl ]]; then break; fi
    sleep 60
done
echo "[$(date -u +%FT%TZ)] ViT-B FT done starting ViT-L 8:16 INLINE FT"

cd /workspace/code
mkdir -p /workspace/run_outputs/vitl_ft_inline/D816_ser_a05_best

python -u run_vitb_ft_inline.py \
    --timm-name vit_large_patch16_224.augreg_in21k_ft_in1k \
    --ser-checkpoint /workspace/sweep_ckpts/vit_large_patch16_224.augreg_in21k_ft_in1k_keep_s35.pt \
    --imagenet-train /workspace/imagenet/train \
    --imagenet-val /workspace/imagenet/val \
    --output-dir /workspace/run_outputs/vitl_ft_inline/D816_ser_a05_best \
    --k 8 --n 16 --source ser --alpha-ser 0.5 \
    --pipeline linear \
    --epochs 3 --batch 16 --lr 1e-5 --weight-decay 0.01 --label-smoothing 0.1 \
    --distill-temp 2.0 --distill-alpha 0.5 --warmup-steps 1000 \
    --num-workers 8 --log-every 200 --save-every-epoch \
    > /workspace/run_logs/vitl_ft_inline_D816_ser_a05.log 2>&1
echo "[$(date -u +%FT%TZ)] Pod 1 ViT-L 8:16 FT complete"

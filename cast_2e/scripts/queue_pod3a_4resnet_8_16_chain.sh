#!/usr/bin/env bash
# Pod 3a 4-ResNet 8:16 cert+perm FT chain.
# Waits for ImageNet train to finish landing (1000 dirs, ~140 GB) via S3 sync.
# Then runs 4 sequential 3-ep distill FT runs on:
#   resnet50.tv_in1k (smallest, ~3-4 hr)
#   resnet50d.ra2_in1k  (~4-5 hr)
#   resnet101d.ra2_in1k (~6-7 hr)
#   resnet152d.ra2_in1k (~8-10 hr)
# Method: 8:16 dense+perm cert (best 50% sparse method from sweep, pre-FT 0.6556 on resnet50.tv).
set -e

# Wait for train data to land (S3 sync watcher should land it within ~30 min of S3 upload completing)
while true; do
    n=$(ls /workspace/imagenet/train 2>/dev/null | wc -l)
    SIZE=$(du -sb /workspace/imagenet/train 2>/dev/null | awk '{print $1}')
    if [[ "$n" -ge 1000 && "$SIZE" -ge 130000000000 ]]; then
        echo "[$(date -u +%FT%TZ)] train landed: $n dirs, $SIZE bytes — starting 4-ResNet FT chain"
        break
    fi
    if [[ -f /tmp/proceed_pod3a_4resnet ]]; then break; fi
    sleep 60
done

cd /workspace/code

# Check we have all 4 SER ckpts; if not, pull from Pod 1 via S3
for s in resnet50.tv_in1k resnet50d.ra2_in1k resnet101d.ra2_in1k resnet152d.ra2_in1k; do
    if [[ ! -f /workspace/sweep_ckpts/${s}_keep_s35.pt && ! -f /workspace/sweep_ckpts/${s}_keep_s35_classmag.pt ]]; then
        echo "[$(date -u +%FT%TZ)] Missing SER ckpt for $s — fetching from S3 (uploaded earlier from Pod 1)..."
        # Try variants
        aws s3 cp s3://cast-resnet-973584726484/cast2e_ser_ckpts/${s}_keep_s35.pt /workspace/sweep_ckpts/ --profile tradingQQQ 2>/dev/null || true
        aws s3 cp s3://cast-resnet-973584726484/cast2e_ser_ckpts/${s}_keep_s35_classmag.pt /workspace/sweep_ckpts/ --profile tradingQQQ 2>/dev/null || true
    fi
done

# 4-ResNet sequential FT
for entry in "resnet50.tv_in1k:resnet50.tv_in1k_keep_s35.pt" \
             "resnet50d.ra2_in1k:resnet50d.ra2_in1k_keep_s35.pt" \
             "resnet101d.ra2_in1k:resnet101d.ra2_in1k_keep_s35.pt" \
             "resnet152d.ra2_in1k:resnet152d.ra2_in1k_keep_s35_classmag.pt"; do
    timm_name="${entry%:*}"
    ser_ckpt_name="${entry#*:}"
    OUT=/workspace/run_outputs/4resnet_8_16_${timm_name}
    mkdir -p $OUT

    SER_PATH=/workspace/sweep_ckpts/${ser_ckpt_name}
    if [[ ! -f "$SER_PATH" ]]; then
        # Try classmag variant
        SER_ALT=$(echo "$SER_PATH" | sed 's/_keep_s35.pt/_keep_s35_classmag.pt/')
        if [[ -f "$SER_ALT" ]]; then
            SER_PATH=$SER_ALT
        else
            echo "[$(date -u +%FT%TZ)] No SER ckpt for $timm_name, using dense source only"
            SER_PATH=/workspace/sweep_ckpts/resnet50.tv_in1k_keep_s35.pt  # placeholder
        fi
    fi

    echo "=== [$(date -u +%FT%TZ)] FT $timm_name with 8:16 dense+perm cert ==="
    python -u /workspace/code/run_resnet_ft_inline.py \
        --timm-name "$timm_name" \
        --ser-checkpoint "$SER_PATH" \
        --imagenet-train /workspace/imagenet/train \
        --imagenet-val /workspace/imagenet/val \
        --output-dir "$OUT" \
        --k 8 --n 16 --source dense --alpha-ser 0.0 \
        --epochs 3 --batch 256 --lr 1e-3 --weight-decay 1e-4 --momentum 0.9 \
        --label-smoothing 0.1 --distill-temp 2.0 --distill-alpha 0.5 \
        --warmup-steps 500 --num-workers 8 --log-every 100 --save-every-epoch \
        > /workspace/run_logs/8_16_${timm_name}.log 2>&1
    echo "[$(date -u +%FT%TZ)] FT $timm_name complete"
done

echo "[$(date -u +%FT%TZ)] === 4-ResNet 8:16 FT chain ALL DONE ==="

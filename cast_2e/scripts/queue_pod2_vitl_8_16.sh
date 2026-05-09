#!/usr/bin/env bash
# Pod 2 master queue (replaces _v2):
#   1. Finish remaining 6:12 -> 2:4 speedup benchmarks for ViT-B ckpts
#   2. Run NEW ViT-L 8:16 SER+alpha=0.5 inline FT (best ViT method applied to ViT-L)
#
# 8:16 chosen because ViT-L Cin=1024, 4096 are divisible by 16 but not by 12.
# 8:16 gives 50% structured sparsity with more pattern flexibility per group
# than 2:4 -- the canonical CAST recipe used 2:4 + ToMe r=8 and got 84.37%.
set -e

cd /workspace/code

echo "=== STEP 1/2: ViT-B 6:12 -> 2:4 projection speedup benchmarks ==="
mkdir -p /workspace/run_outputs/benchmarks_speedup

for ckpt_label in D612_ser_a05 D612_dense_perm D48_dense_perm D816_dense_perm S1216_ser_a05; do
    out_json=/workspace/run_outputs/benchmarks_speedup/${ckpt_label}_projection_speedup.json
    if [[ -f "${out_json}" ]]; then
        echo "  ${ckpt_label} already done, skipping"
        continue
    fi
    if [[ "${ckpt_label}" == "D612_ser_a05" ]] || [[ "${ckpt_label}" == "D612_dense_perm" ]]; then
        ckpt_path=/workspace/staging_ckpts/vitb/${ckpt_label}.pt
    else
        ckpt_path=/workspace/run_outputs/ckpts_all/vitb/${ckpt_label}.pt
    fi
    if [[ -f "${ckpt_path}" ]]; then
        echo "--- benchmarking ${ckpt_label} ---"
        python -u benchmark_6_12_to_2_4_projection.py \
            --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k \
            --student-ckpt "${ckpt_path}" \
            --output "${out_json}" \
            --batch 128 --warmup 20 --iters 100 \
            > /workspace/run_logs/proj_bench_${ckpt_label}.log 2>&1
        echo "  [$(date -u +%FT%TZ)] ${ckpt_label} done"
    else
        echo "  ckpt ${ckpt_label} not found at ${ckpt_path}, skipping"
    fi
done

echo "=== STEP 2/2: ViT-L 8:16 SER+alpha=0.5 INLINE FT (best method on ViT-L) ==="
mkdir -p /workspace/run_outputs/vitl_ft_inline/D816_ser_a05_best

# Use smaller batch (16) for ViT-L due to memory; LR scaled accordingly.
# Match canonical ViT-L FT recipe: lr=1e-5, weight_decay=0.01, warmup=1000, label_smooth=0.1.
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
echo "[$(date -u +%FT%TZ)] Pod 2 ViT-L 8:16 FT complete"

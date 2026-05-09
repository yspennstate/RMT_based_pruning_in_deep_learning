#!/usr/bin/env bash
# After ViT-L finishes:
#   1. Run 6:12 -> 2:4 projection benchmark on saved 50%-sparse ckpts (paper speedup number)
#      NOTE: throughput measurement is valid even if accuracy of saved ckpt is wrong
#      due to missing perm-hook on reload, since we only time the GEMM kernel.
#   2. Run D612_dense_perm 3-ep inline FT (companion to D612_ser on Pod 1)
set -e

while ! grep -qE "step=16014[0-9]/160145|=== ViT-L FT" /workspace/run_logs/vitl_outer2.log 2>/dev/null; do
    if [[ -f /tmp/proceed_pod2_post_vitl ]]; then break; fi
    sleep 60
done
echo "[$(date -u +%FT%TZ)] ViT-L done starting Pod 2 post-ViT-L chain"

cd /workspace/code

echo "=== STEP 1/2: 6:12 -> 2:4 projection speedup benchmarks ==="
mkdir -p /workspace/run_outputs/benchmarks_speedup

for ckpt_label in D612_ser_a05 D612_dense_perm D48_dense_perm D816_dense_perm S1216_ser_a05; do
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
            --output /workspace/run_outputs/benchmarks_speedup/${ckpt_label}_projection_speedup.json \
            --batch 128 --warmup 20 --iters 100 \
            > /workspace/run_logs/proj_bench_${ckpt_label}.log 2>&1
        echo "  [$(date -u +%FT%TZ)] ${ckpt_label} done"
    else
        echo "  ckpt ${ckpt_label} not found at ${ckpt_path}, skipping"
    fi
done

echo "=== STEP 2/2: ViT-B D612_dense_perm 3-ep INLINE FT ==="
mkdir -p /workspace/run_outputs/vitb_ft_inline/D612_dense_perm
python -u run_vitb_ft_inline.py \
    --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --ser-checkpoint /workspace/sweep_ckpts/vit_base_patch16_224.augreg2_in21k_ft_in1k_keep_s35_classmag.pt \
    --imagenet-train /workspace/imagenet/train \
    --imagenet-val /workspace/imagenet/val \
    --output-dir /workspace/run_outputs/vitb_ft_inline/D612_dense_perm \
    --k 6 --n 12 --source dense --alpha-ser 0.0 \
    --pipeline linear \
    --epochs 3 --batch 256 --lr 1e-4 --weight-decay 0.05 --label-smoothing 0.1 \
    --distill-temp 2.0 --distill-alpha 0.5 --warmup-steps 500 \
    --num-workers 8 --log-every 100 --save-every-epoch \
    > /workspace/run_logs/vitb_ft_inline_D612_dense_perm.log 2>&1
echo "[$(date -u +%FT%TZ)] Pod 2 chain complete"

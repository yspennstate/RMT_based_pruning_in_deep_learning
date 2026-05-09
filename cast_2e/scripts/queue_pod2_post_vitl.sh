#!/usr/bin/env bash
# After ViT-L finishes:
#   1. Run 6:12 -> 2:4 projection benchmark on 50%-sparse ckpts (paper speedup number)
#   2. Run D612_dense_perm 3-ep FT (companion to D612_ser_alpha=0.5 on Pod 1)
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

echo "=== STEP 2/2: ViT-B D612_dense_perm 3-ep FT ==="
python -u run_vitb_ft_from_ckpt.py \
    --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --student-ckpt /workspace/staging_ckpts/vitb/D612_dense_perm.pt \
    --imagenet-train /workspace/imagenet/train \
    --imagenet-val /workspace/imagenet/val \
    --output-dir /workspace/run_outputs/vitb_ft/D612_dense_perm \
    --epochs 3 --batch 256 --lr 5e-5 --distill-temp 2.0 --distill-alpha 0.5 \
    --num-workers 8 --log-every 100 --save-every-epoch \
    > /workspace/run_logs/vitb_ft_D612_dense_perm.log 2>&1
echo "[$(date -u +%FT%TZ)] Pod 2 chain complete"

#!/usr/bin/env bash
# pod1_best_2hr_chain.sh — 2 hour Pod 1 chain. Runs ONLY winning pre-FT
# methods on resnet50 / vit-b / convnext, saves every ckpt + mask, then
# benchmarks each with PyTorch's native 2:4 sparse kernel for actual
# wall-clock speedup. 4:8/8:16 fall back to dense kernel + theoretical
# FLOP reduction (need cuSparseLt for hardware speedup).
set -e
set -o pipefail

cd /workspace/code

OUT=/workspace/run_outputs/best_pod1
LOG=/workspace/run_logs
CKPT=/workspace/sweep_ckpts
IMVAL=/workspace/imagenet/val
IMCALIB=/workspace/imagenet/val
mkdir -p $OUT $LOG

echo "=== POD 1 BEST-METHODS 2HR CHAIN $(date -u +%FT%TZ) ==="

# ───────── ResNet50 ─────────
echo "[1/3] ResNet50 best-methods sweep..."
python -u cert_opt_eval_best.py \
    --timm-name resnet50.tv_in1k \
    --ser-checkpoint $CKPT/resnet50.tv_in1k_keep_s35.pt \
    --imagenet-val $IMVAL --imagenet-train-for-calib $IMCALIB \
    --output $OUT/resnet50_best_results.json \
    --mask-save-dir $OUT/masks/resnet50 \
    --ckpt-save-all-dir $OUT/ckpts/resnet50 \
    --pipeline conv --arch-key resnet --num-workers 4 \
    2>&1 | tee $LOG/pod1_best_resnet50.log
echo "  [1/3] ResNet50 sweep DONE."

echo "[1b/3] Benchmark ResNet50 ckpts..."
python -u benchmark_all_ckpts.py \
    --timm-name resnet50.tv_in1k \
    --ckpts-dir $OUT/ckpts/resnet50 \
    --output $OUT/benchmarks/resnet50_bench.json \
    --batch 128 --warmup 20 --iters 100 \
    2>&1 | tee $LOG/pod1_best_resnet50_bench.log
echo "  [1b/3] ResNet50 benchmark DONE."

# ───────── ViT-B ─────────
echo "[2/3] ViT-B best-methods sweep..."
python -u cert_opt_eval_best.py \
    --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --ser-checkpoint $CKPT/vit_base_patch16_224.augreg2_in21k_ft_in1k_keep_s35_classmag.pt \
    --imagenet-val $IMVAL --imagenet-train-for-calib $IMCALIB \
    --output $OUT/vitb_best_results.json \
    --mask-save-dir $OUT/masks/vitb \
    --ckpt-save-all-dir $OUT/ckpts/vitb \
    --pipeline linear --arch-key vitb --num-workers 4 \
    2>&1 | tee $LOG/pod1_best_vitb.log
echo "  [2/3] ViT-B sweep DONE."

echo "[2b/3] Benchmark ViT-B ckpts..."
python -u benchmark_all_ckpts.py \
    --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --ckpts-dir $OUT/ckpts/vitb \
    --output $OUT/benchmarks/vitb_bench.json \
    --batch 128 --warmup 20 --iters 100 \
    2>&1 | tee $LOG/pod1_best_vitb_bench.log
echo "  [2b/3] ViT-B benchmark DONE."

# ───────── ConvNeXt ─────────
# Note: timm name on Pod 1 sweep_ckpts dir is convnext_base.fb_in22k_ft_in1k
echo "[3/3] ConvNeXt-Base best-methods sweep..."
python -u cert_opt_eval_best.py \
    --timm-name convnext_base.fb_in22k_ft_in1k \
    --ser-checkpoint $CKPT/convnext_base.fb_in22k_ft_in1k_keep_s35.pt \
    --imagenet-val $IMVAL --imagenet-train-for-calib $IMCALIB \
    --output $OUT/convnext_best_results.json \
    --mask-save-dir $OUT/masks/convnext \
    --ckpt-save-all-dir $OUT/ckpts/convnext \
    --pipeline linear --arch-key convnext --num-workers 4 \
    2>&1 | tee $LOG/pod1_best_convnext.log
echo "  [3/3] ConvNeXt sweep DONE."

echo "[3b/3] Benchmark ConvNeXt ckpts..."
python -u benchmark_all_ckpts.py \
    --timm-name convnext_base.fb_in22k_ft_in1k \
    --ckpts-dir $OUT/ckpts/convnext \
    --output $OUT/benchmarks/convnext_bench.json \
    --batch 128 --warmup 20 --iters 100 \
    2>&1 | tee $LOG/pod1_best_convnext_bench.log
echo "  [3b/3] ConvNeXt benchmark DONE."

# Summary
echo ""
echo "=== POD 1 BEST 2HR CHAIN COMPLETE $(date -u +%FT%TZ) ==="
echo "Results in $OUT:"
ls -la $OUT/
echo ""
echo "Per-cell ckpts:"
ls -la $OUT/ckpts/*/
echo ""
echo "Benchmarks:"
ls -la $OUT/benchmarks/

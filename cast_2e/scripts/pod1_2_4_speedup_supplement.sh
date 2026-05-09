#!/usr/bin/env bash
# pod1_2_4_speedup_supplement.sh — runs a 2:4 cell on each of resnet50/vitb/convnext
# so we have ACTUAL hardware speedup data via PyTorch's to_sparse_semi_structured.
# This complements the 4:8/8:16/12:16 best-methods sweep which can only report
# theoretical FLOP reduction.
set -e
set -o pipefail

cd /workspace/code
OUT=/workspace/run_outputs/best_pod1
LOG=/workspace/run_logs

# Minimal sweep — single 2:4 cell per arch (dense source, perm, no α_ser)
cat > /tmp/cells_24.py <<'EOF'
# Patch CELLS_BY_ARCH to single 2:4 cell per arch
import sys
sys.path.insert(0, "/workspace/code")
import cert_opt_eval_best as ceb
ceb.CELLS_BY_ARCH = {
    "resnet":   [("D24_dense_perm", 2, 4, "dense", True, 0.0)],
    "vitb":     [("D24_dense_perm", 2, 4, "dense", True, 0.0)],
    "convnext": [("D24_dense_perm", 2, 4, "dense", True, 0.0)],
}
ceb.main()
EOF

echo "=== POD 1 2:4 SPEEDUP SUPPLEMENT $(date -u +%FT%TZ) ==="

# ResNet50 2:4
echo "[1/3] ResNet50 2:4..."
python -u /tmp/cells_24.py \
    --timm-name resnet50.tv_in1k \
    --ser-checkpoint /workspace/sweep_ckpts/resnet50.tv_in1k_keep_s35.pt \
    --imagenet-val /workspace/imagenet/val --imagenet-train-for-calib /workspace/imagenet/val \
    --output $OUT/resnet50_24_results.json \
    --mask-save-dir $OUT/masks/resnet50 \
    --ckpt-save-all-dir $OUT/ckpts/resnet50 \
    --pipeline conv --arch-key resnet --num-workers 4 \
    2>&1 | tee $LOG/pod1_24_resnet50.log

python -u benchmark_all_ckpts.py \
    --timm-name resnet50.tv_in1k \
    --ckpts-dir $OUT/ckpts/resnet50 \
    --output $OUT/benchmarks/resnet50_bench_with_24.json \
    --batch 128 --warmup 20 --iters 100 \
    2>&1 | tee $LOG/pod1_24_resnet50_bench.log

# ViT-B 2:4
echo "[2/3] ViT-B 2:4..."
python -u /tmp/cells_24.py \
    --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --ser-checkpoint /workspace/sweep_ckpts/vit_base_patch16_224.augreg2_in21k_ft_in1k_keep_s35_classmag.pt \
    --imagenet-val /workspace/imagenet/val --imagenet-train-for-calib /workspace/imagenet/val \
    --output $OUT/vitb_24_results.json \
    --mask-save-dir $OUT/masks/vitb \
    --ckpt-save-all-dir $OUT/ckpts/vitb \
    --pipeline linear --arch-key vitb --num-workers 4 \
    2>&1 | tee $LOG/pod1_24_vitb.log

python -u benchmark_all_ckpts.py \
    --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --ckpts-dir $OUT/ckpts/vitb \
    --output $OUT/benchmarks/vitb_bench_with_24.json \
    --batch 128 --warmup 20 --iters 100 \
    2>&1 | tee $LOG/pod1_24_vitb_bench.log

# ConvNeXt 2:4
echo "[3/3] ConvNeXt 2:4..."
python -u /tmp/cells_24.py \
    --timm-name convnext_base.fb_in22k_ft_in1k \
    --ser-checkpoint /workspace/sweep_ckpts/convnext_base.fb_in22k_ft_in1k_keep_s35.pt \
    --imagenet-val /workspace/imagenet/val --imagenet-train-for-calib /workspace/imagenet/val \
    --output $OUT/convnext_24_results.json \
    --mask-save-dir $OUT/masks/convnext \
    --ckpt-save-all-dir $OUT/ckpts/convnext \
    --pipeline linear --arch-key convnext --num-workers 4 \
    2>&1 | tee $LOG/pod1_24_convnext.log

python -u benchmark_all_ckpts.py \
    --timm-name convnext_base.fb_in22k_ft_in1k \
    --ckpts-dir $OUT/ckpts/convnext \
    --output $OUT/benchmarks/convnext_bench_with_24.json \
    --batch 128 --warmup 20 --iters 100 \
    2>&1 | tee $LOG/pod1_24_convnext_bench.log

echo ""
echo "=== 2:4 SPEEDUP SUPPLEMENT DONE $(date -u +%FT%TZ) ==="
ls -la $OUT/benchmarks/

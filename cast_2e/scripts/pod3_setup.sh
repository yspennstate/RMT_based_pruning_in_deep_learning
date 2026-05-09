#!/usr/bin/env bash
# pod3_setup.sh — Pod 3 bootstrap. Run this on a fresh A100 80GB pod once SSH is
# established. Assumes ImageNet is at /workspace/imagenet/{train,val} (either
# from attached volume or from prior parquet→ImageFolder).

set -euo pipefail

mkdir -p /workspace/code /workspace/sweep_ckpts /workspace/run_logs /workspace/run_outputs

# ---- 1. Verify ImageNet is reachable ----
if [ ! -d /workspace/imagenet/val ] || [ ! -d /workspace/imagenet/train ]; then
    echo "ERROR: /workspace/imagenet/{train,val} not found; attach the EU-SE-1 volume mbjoetvn22"
    exit 1
fi
echo "ImageNet OK: $(ls /workspace/imagenet/val | wc -l) val classes"

# ---- 2. Python env ----
pip install -q --upgrade timm torchvision 2>&1 | tail -3

# ---- 3. Generate ResNet50 + ViT-B SER ckpts if missing ----
if [ ! -f /workspace/sweep_ckpts/resnet50.tv_in1k_keep_s35.pt ]; then
    # Use Classical Magnitude as a stand-in for ResNet (Hybrid Mag-SER ckpt is
    # external; gap ~0.75 pp per Table 2). For the dense-vs-SER ablation, the
    # Classical Magnitude version is sufficient.
    echo "Generating resnet50 Classical Magnitude s=0.35 ckpt..."
    python -u /workspace/code/quick_prune_resnet_magnitude.py \
        --timm-name resnet50.tv_in1k --target-sparsity 0.35 \
        --output /workspace/sweep_ckpts/resnet50.tv_in1k_keep_s35.pt
fi

if [ ! -f /workspace/sweep_ckpts/vit_base_patch16_224.augreg2_in21k_ft_in1k_keep_s35_classmag.pt ]; then
    echo "Generating ViT-B/16 224 Classical Magnitude s=0.35 ckpt..."
    python -u /workspace/code/quick_prune_vitb224.py \
        --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k --target-sparsity 0.35 \
        --output /workspace/sweep_ckpts/vit_base_patch16_224.augreg2_in21k_ft_in1k_keep_s35_classmag.pt
fi

echo "Pod 3 bootstrap done."

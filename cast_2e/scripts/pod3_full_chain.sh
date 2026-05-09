#!/usr/bin/env bash
# Pod 3 full chain: bootstrap (download val, install deps, generate ckpts) +
# comprehensive 10-hour kn sweep over ResNet50, ViT-B, ConvNeXtV2-B (and DeiT-B)
# both dense and SER source, ~6 sparsity patterns × ~6 α_ser values × replicates.
#
# Args:
#   $1 = pod role: "conv" runs ResNet50 + ConvNeXtV2-B
#                  "linear" runs ViT-B + DeiT-B
# Usage: bash pod3_full_chain.sh conv

set +e
ROLE=${1:-conv}
exec > /workspace/pod3_chain_${ROLE}.log 2>&1
echo "=== Pod 3 chain start, role=$ROLE $(date -u) ==="

# ---- Setup paths ----
mkdir -p /workspace/code /workspace/sweep_ckpts /workspace/run_logs /workspace/imagenet
cd /workspace

# ---- Install Python deps ----
echo "--- installing deps ---"
pip install -q --upgrade timm torchvision pyarrow pillow 2>&1 | tail -3

# ---- Download ImageNet val from HF (50K images, ~6.7 GB) ----
if [ ! -d /workspace/imagenet/val ] || [ "$(ls /workspace/imagenet/val 2>/dev/null | wc -l)" -lt 1000 ]; then
    echo "--- downloading + converting ImageNet val ---"
    python -c "
from huggingface_hub import snapshot_download
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
snapshot_download(repo_id='ILSVRC/imagenet-1k', repo_type='dataset',
                  allow_patterns=['data/val_images*.tar.gz', 'imagenet_2012_validation_synset_labels.txt'],
                  local_dir='/workspace/hf_imagenet', max_workers=8)
" 2>&1 | tail -5

    # Untar val
    cd /workspace/hf_imagenet/data 2>/dev/null
    mkdir -p /workspace/imagenet/val_tmp
    for tar in val_images*.tar.gz; do
        echo "untar $tar..."
        tar xzf "$tar" -C /workspace/imagenet/val_tmp 2>&1 | tail -3
    done

    # Use timm's labels file or torch's standard mapping. Quickest: structure
    # the val into class-id folders using the synset labels file.
    python << 'PYEOF'
import os, shutil
synset_path = '/workspace/hf_imagenet/imagenet_2012_validation_synset_labels.txt'
if not os.path.exists(synset_path):
    print("no synset labels — falling back to flat val (won't have class folders)")
else:
    with open(synset_path) as f:
        syns = [line.strip() for line in f]
    val_tmp = '/workspace/imagenet/val_tmp'
    val_out = '/workspace/imagenet/val'
    os.makedirs(val_out, exist_ok=True)
    files = sorted(os.listdir(val_tmp))
    print(f"got {len(files)} val files, {len(syns)} synsets")
    for fname, syn in zip(files, syns):
        cls_dir = os.path.join(val_out, syn)
        os.makedirs(cls_dir, exist_ok=True)
        os.rename(os.path.join(val_tmp, fname), os.path.join(cls_dir, fname))
    shutil.rmtree(val_tmp, ignore_errors=True)
    print(f"done — {len(os.listdir(val_out))} class folders in {val_out}")
PYEOF
    cd /workspace
fi

ls /workspace/imagenet/val 2>&1 | head -3
echo "Val class count: $(ls /workspace/imagenet/val 2>/dev/null | wc -l)"
VAL=/workspace/imagenet/val
CALIB=/workspace/imagenet/val

# ---- Generate SER ckpts ----
gen_ckpt () {
    local arch=$1; local out=$2
    if [ ! -f "$out" ]; then
        if [[ "$arch" == vit* ]] || [[ "$arch" == deit* ]]; then
            python -u /workspace/code/quick_prune_vitb224.py \
                --timm-name "$arch" --target-sparsity 0.35 --output "$out" 2>&1 || echo "FAILED $arch"
        else
            python -u /workspace/code/quick_prune_resnet_magnitude.py \
                --timm-name "$arch" --target-sparsity 0.35 --output "$out" 2>&1 || echo "FAILED $arch"
        fi
    else
        echo "$out present"
    fi
}

if [ "$ROLE" == "conv" ]; then
    CK_RN50=/workspace/sweep_ckpts/resnet50.tv_in1k_keep_s35_classmag.pt
    CK_CNXTV2=/workspace/sweep_ckpts/convnextv2_base.fcmae_ft_in22k_in1k_keep_s35.pt
    gen_ckpt "resnet50.tv_in1k"                          "$CK_RN50"
    gen_ckpt "convnextv2_base.fcmae_ft_in22k_in1k"       "$CK_CNXTV2"

    echo "=== [conv-1] ResNet50 extended ==="
    python -u /workspace/code/cert_opt_eval_kn_extended.py \
        --timm-name resnet50.tv_in1k --ser-checkpoint "$CK_RN50" \
        --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
        --output /workspace/run_logs/resnet50_kn_extended.json --num-workers 4 \
        2>&1 | tee /workspace/run_logs/resnet50_kn_run.log
    echo "[conv-1] exit=$?"

    echo "=== [conv-2] ConvNeXtV2-B extended ==="
    python -u /workspace/code/cert_opt_eval_kn_extended.py \
        --timm-name convnextv2_base.fcmae_ft_in22k_in1k --ser-checkpoint "$CK_CNXTV2" \
        --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
        --output /workspace/run_logs/convnextv2_kn_extended.json --num-workers 4 \
        2>&1 | tee /workspace/run_logs/convnextv2_kn_run.log
    echo "[conv-2] exit=$?"

    # Round-2: extra replicates + calib variants on ResNet50
    echo "=== [conv-3] ResNet50 ROUND 2 — replicate sweep ==="
    for SEED in 1 2 3 4 5; do
        python -u /workspace/code/cert_opt_eval_kn_extended.py \
            --timm-name resnet50.tv_in1k --ser-checkpoint "$CK_RN50" \
            --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
            --output /workspace/run_logs/resnet50_kn_extended_seed${SEED}.json --num-workers 4 \
            2>&1 | tee /workspace/run_logs/resnet50_kn_seed${SEED}.log
    done

elif [ "$ROLE" == "linear" ]; then
    CK_VITB=/workspace/sweep_ckpts/vit_base_patch16_224.augreg2_in21k_ft_in1k_keep_s35_classmag.pt
    CK_DEITB=/workspace/sweep_ckpts/deit_base_patch16_224_keep_s35.pt
    CK_DEITS=/workspace/sweep_ckpts/deit_small_patch16_224_keep_s35.pt
    gen_ckpt "vit_base_patch16_224.augreg2_in21k_ft_in1k"  "$CK_VITB"
    gen_ckpt "deit_base_patch16_224.fb_in1k"               "$CK_DEITB"

    echo "=== [linear-1] ViT-B extended ==="
    python -u /workspace/code/cert_opt_eval_vitb_kn_extended.py \
        --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k --ser-checkpoint "$CK_VITB" \
        --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
        --output /workspace/run_logs/vitb_kn_extended.json --num-workers 4 --batch-size-val 64 \
        2>&1 | tee /workspace/run_logs/vitb_kn_run.log
    echo "[linear-1] exit=$?"

    echo "=== [linear-2] DeiT-B extended ==="
    python -u /workspace/code/cert_opt_eval_vitb_kn_extended.py \
        --timm-name deit_base_patch16_224.fb_in1k --ser-checkpoint "$CK_DEITB" \
        --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
        --output /workspace/run_logs/deitb_kn_extended.json --num-workers 4 --batch-size-val 64 \
        2>&1 | tee /workspace/run_logs/deitb_kn_run.log
    echo "[linear-2] exit=$?"

    echo "=== [linear-3] ViT-B ROUND 2 — replicate sweep ==="
    for SEED in 1 2 3 4 5; do
        python -u /workspace/code/cert_opt_eval_vitb_kn_extended.py \
            --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k --ser-checkpoint "$CK_VITB" \
            --imagenet-val "$VAL" --imagenet-train-for-calib "$CALIB" \
            --output /workspace/run_logs/vitb_kn_extended_seed${SEED}.json --num-workers 4 --batch-size-val 64 \
            2>&1 | tee /workspace/run_logs/vitb_kn_seed${SEED}.log
    done
fi

echo "=== Pod 3 chain DONE $(date -u) ==="
ls -la /workspace/run_logs/*.json

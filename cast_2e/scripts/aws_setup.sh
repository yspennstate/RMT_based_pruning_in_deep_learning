#!/usr/bin/env bash
# aws_setup.sh - bootstrap a fresh DLAMI g4dn/g6 spot instance for ResNet CAST-2E.
#
# Run as ubuntu user from user-data via:
#   sudo -i -u ubuntu bash -c "S3_BUCKET=... S3_BACKUP=... bash /tmp/setup.sh"
#
# Steps:
#   1. Activate the DLAMI conda env so 'pip' resolves
#   2. Install timm + datasets + huggingface_hub
#   3. Fetch HF_TOKEN from s3://$S3_BUCKET/secrets/hf_token.txt (IAM-gated)
#   4. Download imagenet-1k parquet shards from HF (parallel via xargs)
#   5. Materialize parquet -> ImageFolder /workspace/imagenet/{train,val}
#   6. Pull code + SER ckpts from S3
# Total wall-clock: ~75-100 min (ImageNet download dominates)

set -euo pipefail
echo "[$(date -u +%H:%M:%S)] === setup start ==="

# ---- 1. Activate DLAMI env (system Python or conda) ----
# DLAMI Ubuntu 24.04 PyTorch image: pytorch is pre-installed in /opt/pytorch venv.
# /opt/pytorch/bin/python is the right interpreter; pip is /opt/pytorch/bin/pip.
if [ -x /opt/pytorch/bin/pip ]; then
    PIP=/opt/pytorch/bin/pip
    PY=/opt/pytorch/bin/python
elif [ -x /opt/conda/bin/pip ]; then
    PIP=/opt/conda/bin/pip
    PY=/opt/conda/bin/python
else
    # fallback: ensure-pip on the system python
    sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip
    PIP="python3 -m pip"
    PY=python3
fi
echo "[$(date -u +%H:%M:%S)] using PY=$PY  PIP=$PIP"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ---- 2. Workspace + extra Python deps ----
WORKSPACE=/workspace
sudo mkdir -p $WORKSPACE
sudo chown ubuntu:ubuntu $WORKSPACE
mkdir -p $WORKSPACE/code $WORKSPACE/sweep_ckpts $WORKSPACE/imagenet $WORKSPACE/run_logs

echo "[$(date -u +%H:%M:%S)] === installing deps ==="
$PIP install --quiet --upgrade pip
$PIP install --quiet timm==1.0.13 huggingface_hub pyarrow pillow tqdm pyyaml

# Make timm + python visible to subsequent invocations
echo "export PATH=$(dirname $PY):\$PATH" >> $HOME/.bashrc
export PATH="$(dirname $PY):$PATH"

# ---- 3. S3 buckets ----
S3_BUCKET="${S3_BUCKET:-cast-resnet-973584726484}"
S3_BACKUP="${S3_BACKUP:-cast-resnet-backup-973584726484}"

# ---- 4. Pull code + ckpts from S3 ----
echo "[$(date -u +%H:%M:%S)] === pulling code + SER ckpts ==="
aws s3 sync s3://$S3_BUCKET/cast_2e_resnet_review/ $WORKSPACE/code/ --quiet
chmod +x $WORKSPACE/code/*.sh
aws s3 sync s3://$S3_BUCKET/sweep_ckpts/ $WORKSPACE/sweep_ckpts/ --quiet
ls -lah $WORKSPACE/sweep_ckpts/

# ---- 5. HF token + ImageNet download ----
echo "[$(date -u +%H:%M:%S)] === fetching HF token ==="
HF_TOKEN=$(aws s3 cp s3://$S3_BUCKET/secrets/hf_token.txt - 2>/dev/null | tr -d '\r\n ')
if [ -z "$HF_TOKEN" ]; then
    echo "FATAL: hf_token.txt missing in s3://$S3_BUCKET/secrets/"
    exit 1
fi
export HF_TOKEN
echo "  HF token loaded (${#HF_TOKEN} chars)"

# Where the parquet shards land. Use the dedicated EBS root (200 GB).
PARQUET_DIR=$WORKSPACE/parquet_cache
mkdir -p $PARQUET_DIR
REVISION=49e2ee26f3810fb5a7536bbf732a7b07389a47b5

echo "[$(date -u +%H:%M:%S)] === downloading imagenet-1k parquet (308 shards, ~150 GB, parallel) ==="
# Build URL list: 294 train + 14 val. Skip test (we don't need it).
{
    for i in $(seq -f "%05g" 0 293); do
        echo "https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/$REVISION/data/train-$i-of-00294.parquet"
    done
    for i in $(seq -f "%05g" 0 13); do
        echo "https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/$REVISION/data/validation-$i-of-00014.parquet"
    done
} > /tmp/parquet_urls.txt
TOTAL=$(wc -l < /tmp/parquet_urls.txt)
echo "  $TOTAL urls queued"

# 8-way parallel download with curl + retries; resumable via -C -
download_one() {
    url="$1"
    fname=$(basename "$url")
    out="$PARQUET_DIR/$fname"
    if [ -f "$out" ] && [ "$(stat -c%s "$out")" -gt 1000000 ]; then
        echo "  SKIP $fname"
        return 0
    fi
    for try in 1 2 3; do
        if curl -sSL --fail --retry 3 --retry-delay 5 -H "Authorization: Bearer $HF_TOKEN" -o "$out" "$url"; then
            sz=$(stat -c%s "$out")
            echo "  OK   $fname  ${sz} bytes"
            return 0
        fi
        echo "  retry $try $fname"
        rm -f "$out"
        sleep 5
    done
    echo "  FAIL $fname (3 tries)"
    return 1
}
export -f download_one
export PARQUET_DIR HF_TOKEN

cat /tmp/parquet_urls.txt | xargs -n1 -P8 -I{} bash -c 'download_one "$@"' _ {} 2>&1 | tee $WORKSPACE/run_logs/parquet_download.log | grep -E "OK|FAIL|retry" | tail -20

DOWNLOADED=$(ls -1 $PARQUET_DIR | wc -l)
echo "[$(date -u +%H:%M:%S)] downloaded $DOWNLOADED / $TOTAL shards"
if [ "$DOWNLOADED" -lt "$TOTAL" ]; then
    echo "FATAL: only $DOWNLOADED of $TOTAL parquet shards downloaded; aborting."
    exit 1
fi

# ---- 6. Materialize parquet -> ImageFolder ----
echo "[$(date -u +%H:%M:%S)] === materializing ImageFolder (this is the slow step ~30 min) ==="
$PY $WORKSPACE/code/parquet_to_imagefolder.py \
    --parquet-dir $PARQUET_DIR \
    --output-root $WORKSPACE/imagenet \
    --train-shards 294 \
    --val-shards 14 \
    --num-workers 4 2>&1 | tee $WORKSPACE/run_logs/extract.log

# Sanity check (SIGPIPE-safe: temporarily relax pipefail for the head pipes)
set +o pipefail
TRAIN_COUNT=$(find $WORKSPACE/imagenet/train -type f -name "*.JPEG" 2>/dev/null | head -2000 | wc -l)
VAL_COUNT=$(find $WORKSPACE/imagenet/val   -type f -name "*.JPEG" 2>/dev/null | wc -l)
set -o pipefail
echo "[$(date -u +%H:%M:%S)] ImageFolder ready: train >= $TRAIN_COUNT (sampled), val = $VAL_COUNT"
if [ "$VAL_COUNT" -lt 49000 ]; then
    echo "FATAL: val set has $VAL_COUNT images, expected ~50000"
    exit 1
fi

# Free disk: parquet shards no longer needed
echo "[$(date -u +%H:%M:%S)] freeing parquet cache ($(du -sh $PARQUET_DIR | cut -f1))"
rm -rf $PARQUET_DIR

# ---- 7. Output bucket sanity check ----
aws s3 ls s3://$S3_BACKUP/ >/dev/null 2>&1 || aws s3 mb s3://$S3_BACKUP --region us-east-1

# ---- 8. Ready ----
echo "[$(date -u +%H:%M:%S)] === ready ==="
echo "  workspace: $WORKSPACE"
echo "  code:      $WORKSPACE/code"
echo "  ckpts:     $WORKSPACE/sweep_ckpts"
echo "  imagenet:  $WORKSPACE/imagenet  (val=$VAL_COUNT JPEGs)"
echo "  S3 input:  s3://$S3_BUCKET"
echo "  S3 output: s3://$S3_BACKUP"
df -h /

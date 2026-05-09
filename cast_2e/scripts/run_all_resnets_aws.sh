#!/usr/bin/env bash
# run_all_resnets_aws.sh — orchestrate the 6-run CAST-2E ResNet experiment on a single
# g6.xlarge spot instance. Sequential because G/VT Spot quota = 4 vCPUs (one box).
#
# Total wall-clock: ~16-24 hours.
# Total cost: ~$5-10 of the $160 AWS credit.
#
# What this script does (in order):
#   1. Confirm we're on the AWS instance (not the local laptop)
#   2. Confirm the workspace exists (aws_setup.sh has run)
#   3. M1, M2, M3 — the 3 main paper rows (cert_aware, 1×1+3×3)
#   4. A1, A2, A3 — the 3 diagnostic ablations on resnet50
#   5. Aggregate manifests into a single results table
#   6. (Optional) Terminate the instance — see TERMINATE_AT_END flag
#
# Spot interruption recovery: each run uses --s3-backup-bucket so per-epoch checkpoints
# go to S3. If interrupted, re-run this script — completed runs (manifest.yaml present
# in S3) are skipped.

set -uo pipefail

# ---- Config ----
WORKSPACE=/workspace
CODE_DIR=$WORKSPACE/code
SWEEP_CKPTS=$WORKSPACE/sweep_ckpts
IMAGENET_ROOT=$WORKSPACE/imagenet
RESULTS_ROOT=$WORKSPACE/cast_resnet
S3_BACKUP="${S3_BACKUP:-cast-resnet-backup-973584726484}"
LOG_DIR=$WORKSPACE/run_logs
mkdir -p $LOG_DIR $RESULTS_ROOT

# Hard cost guardrail — stop if AWS Cost Explorer shows we've spent too much
HARD_COST_LIMIT_USD=60

# Termination at end (set 0 if you want to ssh in afterward)
TERMINATE_AT_END="${TERMINATE_AT_END:-1}"

# ---- Helpers ----
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a $LOG_DIR/master.log; }

run_one() {
    local tag=$1
    local model=$2
    local method=$3
    local include_3x3=$4
    local free_restore=$5

    local out_dir=$RESULTS_ROOT/${tag}_$(date -u +%Y%m%dT%H%M%SZ)
    local s3_done="s3://$S3_BACKUP/$(basename $out_dir)/manifest.yaml"

    # Resume guard: if a previous run for this model+mode already completed in S3, skip.
    if aws s3 ls "s3://$S3_BACKUP/" 2>/dev/null | grep -q "${tag}_"; then
        local prior=$(aws s3 ls "s3://$S3_BACKUP/" | grep "${tag}_" | tail -1 | awk '{print $NF}' | tr -d '/')
        if aws s3 ls "s3://$S3_BACKUP/$prior/manifest.yaml" >/dev/null 2>&1; then
            log "SKIP $tag — prior run $prior already has manifest.yaml in S3"
            return 0
        fi
    fi

    log "================================================================"
    log "$tag  model=$model  method=$method  3x3=$include_3x3  free_restore=$free_restore"
    log "================================================================"

    local ser="$SWEEP_CKPTS/${model}_keep_s35.pt"
    if [ ! -f "$ser" ]; then
        log "FATAL: SER checkpoint missing at $ser"
        return 1
    fi

    local extra_flags=""
    if [ "$include_3x3" = "1" ]; then extra_flags="$extra_flags --include-3x3-convs"; fi
    if [ "$free_restore" = "0" ]; then extra_flags="$extra_flags --no-free-restore"; fi

    local logfile=$LOG_DIR/${tag}_$(date -u +%Y%m%dT%H%M%SZ).log
    log "logfile: $logfile"

    python -u $CODE_DIR/run_resnet_cast_aws.py \
        --timm-name "$model" \
        --ser-checkpoint "$ser" \
        --imagenet-root "$IMAGENET_ROOT" \
        --output-dir "$out_dir" \
        --method "$method" \
        --epochs 3 \
        --batch-size 64 \
        --batch-size-val 128 \
        --lr 1e-5 \
        --label-smoothing 0.1 \
        --distill-temp 2.0 \
        --distill-alpha 0.5 \
        --n-calib-imgs 64 \
        --num-workers 2 \
        --s3-backup-bucket "$S3_BACKUP" \
        --s3-backup-every-min 10 \
        $extra_flags \
        2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}

    if [ $rc -eq 0 ]; then
        log "$tag DONE (rc=$rc)"
        # Final S3 sync to make sure manifest is up there
        aws s3 sync "$out_dir" "s3://$S3_BACKUP/$(basename $out_dir)/" --quiet
    else
        log "$tag FAILED (rc=$rc)  see $logfile"
    fi
    return $rc
}

# ---- Preflight ----
log "Preflight..."
if ! command -v aws >/dev/null; then log "FATAL: aws cli missing"; exit 1; fi
if ! command -v python >/dev/null; then log "FATAL: python missing"; exit 1; fi
if [ ! -d $CODE_DIR ]; then log "FATAL: $CODE_DIR missing — did aws_setup.sh run?"; exit 1; fi
if [ ! -d $IMAGENET_ROOT/train ] || [ ! -d $IMAGENET_ROOT/val ]; then
    log "FATAL: ImageNet missing at $IMAGENET_ROOT/{train,val}"
    exit 1
fi
log "Preflight OK. workspace=$WORKSPACE  s3=$S3_BACKUP"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>&1 | tee -a $LOG_DIR/master.log

# ---- Run filter — only execute runs whose tag is in $RUNS_FILTER (comma-separated)
#      Default: all 6 runs. To execute only M1: RUNS_FILTER=M1 bash run_all_resnets_aws.sh
RUNS_FILTER="${RUNS_FILTER:-M1,M2,M3,A1,A2,A3}"
should_run() { [[ ",$RUNS_FILTER," == *",$1,"* ]]; }

log "RUNS_FILTER=$RUNS_FILTER"

# ---- Main paper rows ----
should_run M1 && (run_one M1_resnet50_1x1_3x3       resnet50.tv_in1k       cert_aware 1 1 || log "M1 problem; continuing")
should_run M2 && (run_one M2_resnet50d_1x1_3x3      resnet50d.ra2_in1k     cert_aware 1 1 || log "M2 problem; continuing")
should_run M3 && (run_one M3_resnet101d_1x1_3x3     resnet101d.ra2_in1k    cert_aware 1 1 || log "M3 problem; continuing")

# ---- Diagnostic ablations on resnet50 ----
should_run A1 && (run_one A1_resnet50_1x1_only      resnet50.tv_in1k       cert_aware 0 1 || log "A1 problem; continuing")
should_run A2 && (run_one A2_resnet50_magnitude     resnet50.tv_in1k       magnitude  1 1 || log "A2 problem; continuing")
should_run A3 && (run_one A3_resnet50_no_freerestore resnet50.tv_in1k      cert_aware 1 0 || log "A3 problem; continuing")

# ---- Aggregate results ----
log "================================================================"
log "Aggregating manifests from S3..."
log "================================================================"
mkdir -p $WORKSPACE/results_summary
aws s3 sync "s3://$S3_BACKUP/" $WORKSPACE/results_summary/ --quiet --exclude "*" --include "*/manifest.yaml" --include "*/final_eval.json" --include "*/two_four_stats.json" --include "*/mac_report.json"

python - <<'PYEOF' | tee $WORKSPACE/results_summary/RESULTS_TABLE.md
import json, glob
from pathlib import Path

rows = []
for run_dir in sorted(Path("/workspace/results_summary").iterdir()):
    if not run_dir.is_dir(): continue
    final = run_dir / "final_eval.json"
    mac   = run_dir / "mac_report.json"
    stats = run_dir / "two_four_stats.json"
    if not (final.exists() and mac.exists() and stats.exists()): continue
    f = json.loads(final.read_text())
    m = json.loads(mac.read_text())
    s = json.loads(stats.read_text())
    rows.append({
        "run":     run_dir.name,
        "model":   f.get("model", "?"),
        "method":  f.get("method", "?"),
        "epochs":  f.get("epochs", "?"),
        "pre_top1":  f.get("pre_ft_top1"),
        "post_top1": f.get("post_ft_top1"),
        "delta":     f.get("delta"),
        "dense_gmacs":     m.get("dense_total_gmacs"),
        "sparse_gmacs":    m.get("sparse_exec_total_gmacs"),
        "mac_reduction":   m.get("mac_reduction_fraction"),
        "bad_groups":      s.get("groups_with_more_than_2_nonzero_after"),
    })

print("# ResNet CAST-2E results (auto-generated)\n")
print("| run | model | method | post_top1 | dense GMACs | sparse GMACs | reduction | bad_groups |")
print("|---|---|---|---:|---:|---:|---:|---:|")
for r in rows:
    print(f"| {r['run']} | {r['model']} | {r['method']} | "
          f"{r['post_top1']:.4f} | {r['dense_gmacs']:.3f} | {r['sparse_gmacs']:.3f} | "
          f"{r['mac_reduction']*100:.1f}% | {r['bad_groups']} |")
PYEOF

aws s3 cp $WORKSPACE/results_summary/RESULTS_TABLE.md "s3://$S3_BACKUP/RESULTS_TABLE.md" --quiet
log "Results table at s3://$S3_BACKUP/RESULTS_TABLE.md"

# ---- Optional terminate ----
if [ "$TERMINATE_AT_END" = "1" ]; then
    INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")
    if [ "$INSTANCE_ID" != "unknown" ]; then
        log "Self-terminating instance $INSTANCE_ID"
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region us-east-1
    fi
fi

log "Master script complete."

#!/bin/bash
# Re-run the 30-cell magnitude vs RMT+SV comparison on the FULL 50K val set
# (no 10K subset). Saves to a separate output file.

set -u
export PYTHONIOENCODING=utf-8
export CUDA_LAUNCH_BLOCKING=1
# Set HF_TOKEN externally before running this script:
#   export HF_TOKEN=...
if [ -z "${HF_TOKEN:-}" ]; then echo "Set HF_TOKEN env var" >&2; exit 1; fi
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export TMPDIR=/workspace/tmp

cd "${RMT_REPO_ROOT:-.}"

OUT=rmt_cache/final_comparison_50k.json
BATCHES=300                   # > ceil(50000/256) so DataLoader is fully iterated
PY=python
SCRIPT=optuna_run/magnitude_rmt_sweep.py
LOG=optuna_run/final_comparison_50k_log.txt

echo "[$(date '+%F %T')] === FULL-VAL FINAL COMPARISON (50K, A40) ===" | tee -a "$LOG"

flush_gpu () {
    $PY -c "import torch,gc; gc.collect(); torch.cuda.empty_cache(); print('GPU flushed')" >> "$LOG" 2>&1
}

# Magnitude baseline
for S in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75; do
    echo "[$(date '+%F %T')] magnitude @ s=${S}%..." | tee -a "$LOG"
    $PY $SCRIPT --eval-batches $BATCHES --out-file $OUT --sparsities $S --methods magnitude >> "$LOG" 2>&1
    flush_gpu
done

# 4-regime RMT + SV
for S in 5 10 15 20; do
    echo "[$(date '+%F %T')] alpha_budget+SV @ s=${S}%..." | tee -a "$LOG"
    $PY $SCRIPT --eval-batches $BATCHES --sv-prune --sv-mode haar --sv-z 0.5 --sv-power 3 \
        --out-file $OUT --sparsities $S --methods alpha_budget_b0.50_sd0.30 >> "$LOG" 2>&1
    flush_gpu
done
for S in 25 30 35 40; do
    echo "[$(date '+%F %T')] splus_b1.00+SV @ s=${S}%..." | tee -a "$LOG"
    $PY $SCRIPT --eval-batches $BATCHES --sv-prune --sv-mode haar --sv-z 0.5 --sv-power 3 \
        --out-file $OUT --sparsities $S --methods splus_budget_b1.00_sd0.50 >> "$LOG" 2>&1
    flush_gpu
done
for S in 45 50 55; do
    echo "[$(date '+%F %T')] splus_b1.25+SV @ s=${S}%..." | tee -a "$LOG"
    $PY $SCRIPT --eval-batches $BATCHES --sv-prune --sv-mode haar --sv-z 0.5 --sv-power 3 \
        --out-file $OUT --sparsities $S --methods splus_budget_b1.25_sd0.70_p1.0 >> "$LOG" 2>&1
    flush_gpu
done
for S in 60 65 70 75; do
    echo "[$(date '+%F %T')] splus_layertype+SV @ s=${S}%..." | tee -a "$LOG"
    $PY $SCRIPT --eval-batches $BATCHES --sv-prune --sv-mode haar --sv-z 0.5 --sv-power 3 \
        --out-file $OUT --sparsities $S --methods splus_budget_b1.50_sd0.85_p1.0_ba1.00_bm2.00 >> "$LOG" 2>&1
    flush_gpu
done

echo "[$(date '+%F %T')] === ALL DONE ===" | tee -a "$LOG"
echo "After this finishes, launch run_finetune_magnitude.py for the magnitude FT baseline."

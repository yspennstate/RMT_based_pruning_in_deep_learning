#!/bin/bash
# Linux equivalent of run_final_comparison.bat
# Magnitude vs RMT+SV (4-regime) comparison, one cell per python invocation.
# Resume-safe: final_comparison.json already has 8 cells from the crashed run.

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

OUT=rmt_cache/final_comparison.json
BATCHES=50
PY=python
SCRIPT=optuna_run/magnitude_rmt_sweep.py
LOG=optuna_run/final_comparison_log.txt

echo "[$(date '+%F %T')] === FINAL COMPARISON (Linux/A40, per-cell isolation) ===" | tee -a "$LOG"

flush_gpu () {
    $PY -c "import torch,gc; gc.collect(); torch.cuda.empty_cache(); print('GPU flushed')" >> "$LOG" 2>&1
}

# Magnitude baseline — 15 sparsities
for S in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75; do
    echo "[$(date '+%F %T')] magnitude @ s=${S}%..." | tee -a "$LOG"
    $PY $SCRIPT --eval-batches $BATCHES --out-file $OUT --sparsities $S --methods magnitude >> "$LOG" 2>&1
    flush_gpu
done

echo "[$(date '+%F %T')] Magnitude done, cooling 30s..." | tee -a "$LOG"
sleep 30

# 4-regime RMT + SV  (haar z=0.5, power=3)
# Regime 1 (5-20%): alpha_budget_b0.50_sd0.30
for S in 5 10 15 20; do
    echo "[$(date '+%F %T')] alpha_budget+SV @ s=${S}%..." | tee -a "$LOG"
    $PY $SCRIPT --eval-batches $BATCHES --sv-prune --sv-mode haar --sv-z 0.5 --sv-power 3 \
        --out-file $OUT --sparsities $S --methods alpha_budget_b0.50_sd0.30 >> "$LOG" 2>&1
    flush_gpu
done

# Regime 2 (25-40%): splus_budget_b1.00_sd0.50
for S in 25 30 35 40; do
    echo "[$(date '+%F %T')] splus_b1.00+SV @ s=${S}%..." | tee -a "$LOG"
    $PY $SCRIPT --eval-batches $BATCHES --sv-prune --sv-mode haar --sv-z 0.5 --sv-power 3 \
        --out-file $OUT --sparsities $S --methods splus_budget_b1.00_sd0.50 >> "$LOG" 2>&1
    flush_gpu
done

# Regime 3 (45-55%): splus_budget_b1.25_sd0.70_p1.0
for S in 45 50 55; do
    echo "[$(date '+%F %T')] splus_b1.25+SV @ s=${S}%..." | tee -a "$LOG"
    $PY $SCRIPT --eval-batches $BATCHES --sv-prune --sv-mode haar --sv-z 0.5 --sv-power 3 \
        --out-file $OUT --sparsities $S --methods splus_budget_b1.25_sd0.70_p1.0 >> "$LOG" 2>&1
    flush_gpu
done

# Regime 4 (60-75%): splus_budget_b1.50_sd0.85_p1.0_ba1.00_bm2.00  (layer-type)
for S in 60 65 70 75; do
    echo "[$(date '+%F %T')] splus_layertype+SV @ s=${S}%..." | tee -a "$LOG"
    $PY $SCRIPT --eval-batches $BATCHES --sv-prune --sv-mode haar --sv-z 0.5 --sv-power 3 \
        --out-file $OUT --sparsities $S --methods splus_budget_b1.50_sd0.85_p1.0_ba1.00_bm2.00 >> "$LOG" 2>&1
    flush_gpu
done

echo "[$(date '+%F %T')] === ALL DONE ===" | tee -a "$LOG"

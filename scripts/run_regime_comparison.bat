@echo off
REM Regime-adapted comparison: best splus config per sparsity regime.
REM LOW (s=5-40%%): b1.00_sd0.50 — gentle modulation
REM MID (s=45-55%%): b1.25_sd0.70_p1.0 — moderate, proven
REM HIGH (s=60-75%%): ba1.00_bm2.00 — layer-type, dominant
REM
REM 3 sequential invocations, shared output file with resume.
REM Magnitude runs across all 15 sparsities via resume logic.
REM Total: 30 cells, ~27 min.
set PYTHONIOENCODING=utf-8
cd /d C:\Users\owner\Projects\rmt_pruning_vit
echo [%date% %time%] Starting regime comparison... >> optuna_run\regime_comparison_log.txt

REM === LOW REGIME: s=5-40%%, b1.00_sd0.50 ===
echo [%date% %time%] LOW regime (s=5-40%%, b1.00_sd0.50)... >> optuna_run\regime_comparison_log.txt
C:\Python314\python.exe optuna_run\magnitude_rmt_sweep.py ^
  --eval-batches 250 ^
  --out-file rmt_cache\regime_comparison.json ^
  --sparsities 5 10 15 20 25 30 35 40 ^
  --methods magnitude splus_budget_b1.00_sd0.50 ^
  >> optuna_run\regime_comparison_log.txt 2>&1

REM === MID REGIME: s=45-55%%, b1.25_sd0.70_p1.0 ===
echo [%date% %time%] MID regime (s=45-55%%, b1.25_sd0.70_p1.0)... >> optuna_run\regime_comparison_log.txt
C:\Python314\python.exe optuna_run\magnitude_rmt_sweep.py ^
  --eval-batches 250 ^
  --out-file rmt_cache\regime_comparison.json ^
  --sparsities 45 50 55 ^
  --methods magnitude splus_budget_b1.25_sd0.70_p1.0 ^
  >> optuna_run\regime_comparison_log.txt 2>&1

REM === HIGH REGIME: s=60-75%%, layer-type ba1.00_bm2.00 ===
echo [%date% %time%] HIGH regime (s=60-75%%, ba1.00_bm2.00)... >> optuna_run\regime_comparison_log.txt
C:\Python314\python.exe optuna_run\magnitude_rmt_sweep.py ^
  --eval-batches 250 ^
  --out-file rmt_cache\regime_comparison.json ^
  --sparsities 60 65 70 75 ^
  --methods magnitude splus_budget_b1.50_sd0.85_p1.0_ba1.00_bm2.00 ^
  >> optuna_run\regime_comparison_log.txt 2>&1

echo [%date% %time%] Regime comparison COMPLETE (exit %ERRORLEVEL%) >> optuna_run\regime_comparison_log.txt

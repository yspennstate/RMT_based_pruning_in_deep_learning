@echo off
REM Refined3 sweep — two parts, shared output file.
REM
REM Part A (uniform-beta tight grid): 4 beta x 4 sd x 5 sparsities = 80 cells
REM   beta in {1.30, 1.40, 1.50, 1.60}, sd in {0.80, 0.85, 0.90, 0.95}, p=1.0
REM   sparsities: 55, 60, 62.5, 65, 67.5
REM
REM Part B (layer-type beta at winning sd/p): 3 beta_attn x 3 beta_mlp x 2 sparsities = 18 cells
REM   Fixed: sd=0.85, p=1.0. Base b=1.50 used only for 'other' layers.
REM   sparsities: 60, 65
REM
REM Total: 98 cells, ~75 min wall time at 150-batch screening eval.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d C:\Users\owner\Projects\rmt_pruning_vit
echo [%date% %time%] Starting refined3 Part A (uniform-beta tight grid)... >> optuna_run\splus_refined3_bat_log.txt

C:\Python314\python.exe optuna_run\magnitude_rmt_sweep.py ^
  --eval-batches 150 ^
  --out-file rmt_cache\splus_refined3_results.json ^
  --sparsities 55 60 62.5 65 67.5 ^
  --methods ^
    splus_budget_b1.30_sd0.80_p1.0 ^
    splus_budget_b1.30_sd0.85_p1.0 ^
    splus_budget_b1.30_sd0.90_p1.0 ^
    splus_budget_b1.30_sd0.95_p1.0 ^
    splus_budget_b1.40_sd0.80_p1.0 ^
    splus_budget_b1.40_sd0.85_p1.0 ^
    splus_budget_b1.40_sd0.90_p1.0 ^
    splus_budget_b1.40_sd0.95_p1.0 ^
    splus_budget_b1.50_sd0.80_p1.0 ^
    splus_budget_b1.50_sd0.85_p1.0 ^
    splus_budget_b1.50_sd0.90_p1.0 ^
    splus_budget_b1.50_sd0.95_p1.0 ^
    splus_budget_b1.60_sd0.80_p1.0 ^
    splus_budget_b1.60_sd0.85_p1.0 ^
    splus_budget_b1.60_sd0.90_p1.0 ^
    splus_budget_b1.60_sd0.95_p1.0 ^
  >> optuna_run\splus_refined3_bat_log.txt 2>&1

echo [%date% %time%] Part A done (exit %ERRORLEVEL%), starting Part B (layer-type beta)... >> optuna_run\splus_refined3_bat_log.txt

C:\Python314\python.exe optuna_run\magnitude_rmt_sweep.py ^
  --eval-batches 150 ^
  --out-file rmt_cache\splus_refined3_results.json ^
  --sparsities 60 65 ^
  --methods ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.25_bm1.25 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.25_bm1.50 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.25_bm1.75 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.50_bm1.25 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.50_bm1.50 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.50_bm1.75 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.75_bm1.25 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.75_bm1.50 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.75_bm1.75 ^
  >> optuna_run\splus_refined3_bat_log.txt 2>&1

echo [%date% %time%] Refined3 complete, exit %ERRORLEVEL% >> optuna_run\splus_refined3_bat_log.txt

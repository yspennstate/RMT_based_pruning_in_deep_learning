@echo off
REM Refined2 splus_budget sweep — extends winning directions from refined1.
REM 27 methods (beta 1.25/1.50/1.75 x sd 0.85/0.95/1.05 x p 0.5/0.75/1.0)
REM x 4 sparsities (0.55/0.60/0.65/0.70) = 108 cells.
REM Uses 150-batch screening eval; writes to separate results file.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d C:\Users\owner\Projects\rmt_pruning_vit
echo [%date% %time%] Starting splus_refined2 sweep... >> optuna_run\splus_refined2_bat_log.txt
C:\Python314\python.exe optuna_run\magnitude_rmt_sweep.py ^
  --eval-batches 150 ^
  --out-file rmt_cache\splus_refined2_results.json ^
  --sparsities 55 60 65 70 ^
  --methods ^
    splus_budget_b1.25_sd0.85_p0.5 ^
    splus_budget_b1.25_sd0.85_p0.75 ^
    splus_budget_b1.25_sd0.85_p1.0 ^
    splus_budget_b1.25_sd0.95_p0.5 ^
    splus_budget_b1.25_sd0.95_p0.75 ^
    splus_budget_b1.25_sd0.95_p1.0 ^
    splus_budget_b1.25_sd1.05_p0.5 ^
    splus_budget_b1.25_sd1.05_p0.75 ^
    splus_budget_b1.25_sd1.05_p1.0 ^
    splus_budget_b1.50_sd0.85_p0.5 ^
    splus_budget_b1.50_sd0.85_p0.75 ^
    splus_budget_b1.50_sd0.85_p1.0 ^
    splus_budget_b1.50_sd0.95_p0.5 ^
    splus_budget_b1.50_sd0.95_p0.75 ^
    splus_budget_b1.50_sd0.95_p1.0 ^
    splus_budget_b1.50_sd1.05_p0.5 ^
    splus_budget_b1.50_sd1.05_p0.75 ^
    splus_budget_b1.50_sd1.05_p1.0 ^
    splus_budget_b1.75_sd0.85_p0.5 ^
    splus_budget_b1.75_sd0.85_p0.75 ^
    splus_budget_b1.75_sd0.85_p1.0 ^
    splus_budget_b1.75_sd0.95_p0.5 ^
    splus_budget_b1.75_sd0.95_p0.75 ^
    splus_budget_b1.75_sd0.95_p1.0 ^
    splus_budget_b1.75_sd1.05_p0.5 ^
    splus_budget_b1.75_sd1.05_p0.75 ^
    splus_budget_b1.75_sd1.05_p1.0 ^
  >> optuna_run\splus_refined2_bat_log.txt 2>&1
echo [%date% %time%] Exit code: %ERRORLEVEL% >> optuna_run\splus_refined2_bat_log.txt

@echo off
REM Refined splus_budget sweep — 18 methods x 6 sparsities = 108 cells.
REM Adds decay_power knob (p=1/2/3 — linear/quadratic/cubic RMT decay with sparsity).
REM Uses 150-batch screening eval for ~40% speedup vs 250 batches.
REM Writes to a separate output file so the main 175-cell sweep stays untouched.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d C:\Users\owner\Projects\rmt_pruning_vit
echo [%date% %time%] Starting splus_refined sweep... >> optuna_run\splus_refined_bat_log.txt
C:\Python314\python.exe optuna_run\magnitude_rmt_sweep.py ^
  --eval-batches 150 ^
  --out-file rmt_cache\splus_refined_results.json ^
  --sparsities 50 55 60 62.5 65 70 ^
  --methods ^
    splus_budget_b0.75_sd0.70_p1.0 ^
    splus_budget_b0.75_sd0.70_p2.0 ^
    splus_budget_b0.75_sd0.70_p3.0 ^
    splus_budget_b0.75_sd0.85_p1.0 ^
    splus_budget_b0.75_sd0.85_p2.0 ^
    splus_budget_b0.75_sd0.85_p3.0 ^
    splus_budget_b1.00_sd0.70_p1.0 ^
    splus_budget_b1.00_sd0.70_p2.0 ^
    splus_budget_b1.00_sd0.70_p3.0 ^
    splus_budget_b1.00_sd0.85_p1.0 ^
    splus_budget_b1.00_sd0.85_p2.0 ^
    splus_budget_b1.00_sd0.85_p3.0 ^
    splus_budget_b1.25_sd0.70_p1.0 ^
    splus_budget_b1.25_sd0.70_p2.0 ^
    splus_budget_b1.25_sd0.70_p3.0 ^
    splus_budget_b1.25_sd0.85_p1.0 ^
    splus_budget_b1.25_sd0.85_p2.0 ^
    splus_budget_b1.25_sd0.85_p3.0 ^
  >> optuna_run\splus_refined_bat_log.txt 2>&1
echo [%date% %time%] Exit code: %ERRORLEVEL% >> optuna_run\splus_refined_bat_log.txt

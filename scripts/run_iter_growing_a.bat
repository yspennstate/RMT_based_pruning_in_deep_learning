@echo off
REM Iterative Haar SV + magnitude prune — crash-safe version.
REM 50% GPU cap (set inside script) + 8/16 CPU cores + BelowNormal priority
REM so the live IBKR trading bots stay responsive.
cd /d C:\Users\owner\Projects\rmt_pruning_vit\optuna_run
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=8
set OPENBLAS_NUM_THREADS=8
set NUMEXPR_NUM_THREADS=8
start /belownormal /affinity FF /b /wait python iterative_growing_a.py > iter_growing_a_stdout.log 2>&1

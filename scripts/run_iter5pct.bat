@echo off
cd /d C:\Users\owner\Projects\rmt_pruning_vit\optuna_run
set PYTHONIOENCODING=utf-8
echo === Iterative 5%% prune compare ===
python iterative_5pct_compare.py
echo.
echo === Done ===
pause

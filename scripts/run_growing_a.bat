@echo off
cd /d C:\Users\owner\Projects\rmt_pruning_vit\optuna_run
set PYTHONIOENCODING=utf-8
echo === Strategy A growing prune (no mask) ===
python iterative_growing_a.py
echo.
echo === Done ===
pause

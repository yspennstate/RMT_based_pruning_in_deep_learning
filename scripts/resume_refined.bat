@echo off
cd /d C:\Users\owner\Projects\rmt_pruning_vit\optuna_run
set PYTHONIOENCODING=utf-8
echo === Resuming Phase 2: Refined z+bulk search ===
python haar_optuna_refined.py
echo.
echo === Phase 2 complete ===
pause

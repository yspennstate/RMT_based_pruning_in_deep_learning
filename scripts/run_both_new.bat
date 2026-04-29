@echo off
cd /d C:\Users\owner\Projects\rmt_pruning_vit\optuna_run
set PYTHONIOENCODING=utf-8
echo === Run 1: Growing K%% prune (no mask) ===
python iterative_growing_a.py
echo.
echo === Run 2: SV decides extra kills ===
python iterative_sv_decides.py
echo.
echo === All done ===
pause

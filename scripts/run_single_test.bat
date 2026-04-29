@echo off
cd /d C:\Users\owner\Projects\rmt_pruning_vit
echo Starting magnitude sweep test...
python optuna_run\magnitude_rmt_sweep.py --methods magnitude --sparsities 30
echo.
echo Exit code: %ERRORLEVEL%
pause

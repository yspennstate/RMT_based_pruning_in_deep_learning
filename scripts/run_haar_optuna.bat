@echo off
cd /d C:\Users\owner\Projects\rmt_pruning_vit\optuna_run
set PYTHONIOENCODING=utf-8

echo === Phase 1: 3-method search ===
python haar_optuna.py
if %errorlevel% neq 0 (
    echo Phase 1 failed with error %errorlevel%
    pause
    exit /b 1
)

echo.
echo === Phase 2: Refined z+bulk search ===
python haar_optuna_refined.py

echo.
echo === Both phases complete ===
pause

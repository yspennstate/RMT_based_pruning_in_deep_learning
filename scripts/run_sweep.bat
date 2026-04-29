@echo off
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d C:\Users\owner\Projects\rmt_pruning_vit
echo [%date% %time%] Starting sweep... >> optuna_run\sweep_bat_log.txt
C:\Python314\python.exe optuna_run\magnitude_rmt_sweep.py --sparsities 30 40 50 55 60 65 70 >> optuna_run\sweep_bat_log.txt 2>&1
echo [%date% %time%] Exit code: %ERRORLEVEL% >> optuna_run\sweep_bat_log.txt

@echo off
cd /d C:\Users\owner\Projects\rmt_pruning_vit
start /belownormal /affinity FF /b /wait python optuna_run\magnitude_rmt_sweep.py
pause

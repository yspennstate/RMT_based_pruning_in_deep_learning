@echo off
cd /d "C:\Users\owner\Projects\rmt_pruning_vit\optuna_run"
python -u overnight_grid_search.py > overnight_log2.txt 2>&1
pause

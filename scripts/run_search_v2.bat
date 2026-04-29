@echo off
cd /d "C:\Users\owner\Projects\rmt_pruning_vit\optuna_run"
python -u hp_search_v2.py 10 > hp_search_v2_log.txt 2>&1
pause

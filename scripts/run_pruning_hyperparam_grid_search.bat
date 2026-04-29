@echo off
REM Long unattended hyperparameter grid search for SEB / Spectral Edge Budgeting.
REM Output JSON files land in hp_results/grid_*.json.
cd /d "%~dp0.."
python -u pruning_hyperparam_grid_search.py > pruning_hyperparam_grid_search.log 2>&1
pause

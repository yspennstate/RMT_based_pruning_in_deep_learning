@echo off
REM Long unattended pruning-method comparison sweep:
REM   classical magnitude vs RMT-grounded variants vs layer-adaptive.
REM Output JSON files land in pruning_method_comparison_results.json.
cd /d "%~dp0.."
python -u pruning_method_comparison_sweep.py > pruning_method_comparison_sweep.log 2>&1
pause

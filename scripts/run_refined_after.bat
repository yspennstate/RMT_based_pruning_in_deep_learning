@echo off
cd /d C:\Users\owner\Projects\rmt_pruning_vit\optuna_run
set PYTHONIOENCODING=utf-8

echo Waiting for Phase 1 (PID 6712) to finish...

:wait_loop
tasklist /FI "PID eq 6712" 2>NUL | find "6712" >NUL
if %errorlevel%==0 (
    timeout /t 30 /nobreak >NUL
    goto wait_loop
)

echo Phase 1 done. Starting refined z+bulk search...
echo.
python haar_optuna_refined.py
echo.
echo === Refined search complete ===
pause

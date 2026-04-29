@echo off
REM ══════════════════════════════════════════════════════════════════
REM Backfill missing cells — with nvidia-smi reset between each cell
REM to clear any stale CUDA driver state.
REM ══════════════════════════════════════════════════════════════════
set PYTHONIOENCODING=utf-8
set CUDA_LAUNCH_BLOCKING=1
cd /d C:\Users\owner\Projects\rmt_pruning_vit

set OUT=rmt_cache\final_comparison.json
set BATCHES=1250
set PY=C:\Python314\python.exe
set SCRIPT=optuna_run\magnitude_rmt_sweep.py
set LOG=optuna_run\final_comparison_log.txt

echo [%date% %time%] === BACKFILL START === >> %LOG%

REM ====== Missing magnitude cells ======
for %%S in (25 30 35 40 45 50 75) do (
    echo [%date% %time%] BACKFILL magnitude @ s=%%S%%... >> %LOG%
    nvidia-smi -r >nul 2>&1
    timeout /t 5 /nobreak >nul
    %PY% %SCRIPT% --eval-batches %BATCHES% --out-file %OUT% --sparsities %%S --methods magnitude >> %LOG% 2>&1
    %PY% -c "import torch,gc;gc.collect();torch.cuda.empty_cache();print('GPU flushed')" >> %LOG% 2>&1
    timeout /t 45 /nobreak >nul
)

REM ====== All RMT+SV cells ======

REM alpha_budget + SV (s=5-20%%)
for %%S in (5 10 15 20) do (
    echo [%date% %time%] BACKFILL alpha+SV @ s=%%S%%... >> %LOG%
    nvidia-smi -r >nul 2>&1
    timeout /t 5 /nobreak >nul
    %PY% %SCRIPT% --eval-batches %BATCHES% --sv-prune --sv-mode haar --sv-z 0.5 --sv-power 3 --out-file %OUT% --sparsities %%S --methods alpha_budget_b0.50_sd0.30 >> %LOG% 2>&1
    %PY% -c "import torch,gc;gc.collect();torch.cuda.empty_cache();print('GPU flushed')" >> %LOG% 2>&1
    timeout /t 45 /nobreak >nul
)

REM splus b1.00 + SV (s=25-40%%)
for %%S in (25 30 35 40) do (
    echo [%date% %time%] BACKFILL splus_b1.00+SV @ s=%%S%%... >> %LOG%
    nvidia-smi -r >nul 2>&1
    timeout /t 5 /nobreak >nul
    %PY% %SCRIPT% --eval-batches %BATCHES% --sv-prune --sv-mode haar --sv-z 0.5 --sv-power 3 --out-file %OUT% --sparsities %%S --methods splus_budget_b1.00_sd0.50 >> %LOG% 2>&1
    %PY% -c "import torch,gc;gc.collect();torch.cuda.empty_cache();print('GPU flushed')" >> %LOG% 2>&1
    timeout /t 45 /nobreak >nul
)

REM splus b1.25 + SV (s=45-55%%)
for %%S in (45 50 55) do (
    echo [%date% %time%] BACKFILL splus_b1.25+SV @ s=%%S%%... >> %LOG%
    nvidia-smi -r >nul 2>&1
    timeout /t 5 /nobreak >nul
    %PY% %SCRIPT% --eval-batches %BATCHES% --sv-prune --sv-mode haar --sv-z 0.5 --sv-power 3 --out-file %OUT% --sparsities %%S --methods splus_budget_b1.25_sd0.70_p1.0 >> %LOG% 2>&1
    %PY% -c "import torch,gc;gc.collect();torch.cuda.empty_cache();print('GPU flushed')" >> %LOG% 2>&1
    timeout /t 45 /nobreak >nul
)

REM splus layertype + SV (s=60-75%%)
for %%S in (60 65 70 75) do (
    echo [%date% %time%] BACKFILL splus_lt+SV @ s=%%S%%... >> %LOG%
    nvidia-smi -r >nul 2>&1
    timeout /t 5 /nobreak >nul
    %PY% %SCRIPT% --eval-batches %BATCHES% --sv-prune --sv-mode haar --sv-z 0.5 --sv-power 3 --out-file %OUT% --sparsities %%S --methods splus_budget_b1.50_sd0.85_p1.0_ba1.00_bm2.00 >> %LOG% 2>&1
    %PY% -c "import torch,gc;gc.collect();torch.cuda.empty_cache();print('GPU flushed')" >> %LOG% 2>&1
    timeout /t 45 /nobreak >nul
)

echo [%date% %time%] === BACKFILL DONE === >> %LOG%
echo BACKFILL DONE
pause

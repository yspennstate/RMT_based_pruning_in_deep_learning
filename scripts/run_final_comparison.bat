@echo off
REM ══════════════════════════════════════════════════════════════════
REM Final comparison: Magnitude pruning vs RMT + SV (best regimes)
REM Full 10K validation set (1250 batches), every 5%% sparsity
REM Each sparsity level is a SEPARATE python invocation to get a fresh
REM CUDA context — prevents one CUDA error from killing the whole run.
REM ══════════════════════════════════════════════════════════════════
set PYTHONIOENCODING=utf-8
set CUDA_LAUNCH_BLOCKING=1
cd /d C:\Users\owner\Projects\rmt_pruning_vit

set OUT=rmt_cache\final_comparison.json
set BATCHES=1250
set PY=C:\Python314\python.exe
set SCRIPT=optuna_run\magnitude_rmt_sweep.py
set LOG=optuna_run\final_comparison_log.txt

echo [%date% %time%] === FINAL COMPARISON START (per-cell isolation) === >> %LOG%

REM ====== MAGNITUDE BASELINE — one cell at a time ======
for %%S in (5 10 15 20 25 30 35 40 45 50 55 60 65 70 75) do (
    echo [%date% %time%] magnitude @ s=%%S%%... >> %LOG%
    %PY% %SCRIPT% --eval-batches %BATCHES% --out-file %OUT% --sparsities %%S --methods magnitude >> %LOG% 2>&1
    %PY% -c "import torch,gc;gc.collect();torch.cuda.empty_cache();print('GPU flushed')" >> %LOG% 2>&1
    timeout /t 45 /nobreak >nul
)

echo [%date% %time%] Magnitude baseline done, cooling 60s... >> %LOG%
timeout /t 60 /nobreak >nul

REM ====== RMT + SV — best regime per sparsity band, one cell at a time ======

REM Low sparsity (5-20%%): alpha_budget + SV haar z=0.5
for %%S in (5 10 15 20) do (
    echo [%date% %time%] alpha_budget+SV @ s=%%S%%... >> %LOG%
    %PY% %SCRIPT% --eval-batches %BATCHES% --sv-prune --sv-mode haar --sv-z 0.5 --sv-power 3 --out-file %OUT% --sparsities %%S --methods alpha_budget_b0.50_sd0.30 >> %LOG% 2>&1
    %PY% -c "import torch,gc;gc.collect();torch.cuda.empty_cache();print('GPU flushed')" >> %LOG% 2>&1
    timeout /t 45 /nobreak >nul
)

REM Low-mid (25-40%%): splus b1.00 sd0.50 + SV
for %%S in (25 30 35 40) do (
    echo [%date% %time%] splus_b1.00+SV @ s=%%S%%... >> %LOG%
    %PY% %SCRIPT% --eval-batches %BATCHES% --sv-prune --sv-mode haar --sv-z 0.5 --sv-power 3 --out-file %OUT% --sparsities %%S --methods splus_budget_b1.00_sd0.50 >> %LOG% 2>&1
    %PY% -c "import torch,gc;gc.collect();torch.cuda.empty_cache();print('GPU flushed')" >> %LOG% 2>&1
    timeout /t 45 /nobreak >nul
)

REM Mid (45-55%%): splus b1.25 sd0.70 + SV
for %%S in (45 50 55) do (
    echo [%date% %time%] splus_b1.25+SV @ s=%%S%%... >> %LOG%
    %PY% %SCRIPT% --eval-batches %BATCHES% --sv-prune --sv-mode haar --sv-z 0.5 --sv-power 3 --out-file %OUT% --sparsities %%S --methods splus_budget_b1.25_sd0.70_p1.0 >> %LOG% 2>&1
    %PY% -c "import torch,gc;gc.collect();torch.cuda.empty_cache();print('GPU flushed')" >> %LOG% 2>&1
    timeout /t 45 /nobreak >nul
)

REM High (60-75%%): splus layertype ba1.00 bm2.00 + SV
for %%S in (60 65 70 75) do (
    echo [%date% %time%] splus_layertype+SV @ s=%%S%%... >> %LOG%
    %PY% %SCRIPT% --eval-batches %BATCHES% --sv-prune --sv-mode haar --sv-z 0.5 --sv-power 3 --out-file %OUT% --sparsities %%S --methods splus_budget_b1.50_sd0.85_p1.0_ba1.00_bm2.00 >> %LOG% 2>&1
    %PY% -c "import torch,gc;gc.collect();torch.cuda.empty_cache();print('GPU flushed')" >> %LOG% 2>&1
    timeout /t 45 /nobreak >nul
)

echo [%date% %time%] === ALL DONE === >> %LOG%
echo ALL DONE
pause

@echo off
REM 4-regime comparison: without and with singular-vector sparsification.
REM Part A: 4-regime (alpha + splus + splus-layertype) without SV
REM Part B: same 4-regime WITH SV pruning (power=30, theta=0.00001125)
REM Safety: CUDA_LAUNCH_BLOCKING=1 for synchronous error reporting,
REM         60s inter-regime cool-down to flush GPU state between invocations.
set PYTHONIOENCODING=utf-8
set CUDA_LAUNCH_BLOCKING=1
cd /d C:\Users\owner\Projects\rmt_pruning_vit
echo [%date% %time%] Starting 4-regime comparison (no SV)... >> optuna_run\four_regime_sv_log.txt

REM ====== PART A: 4-regime WITHOUT SV ======

REM Regime 1: alpha at s=5-20%%
C:\Python314\python.exe optuna_run\magnitude_rmt_sweep.py --eval-batches 250 --out-file rmt_cache\four_regime_nosv.json --sparsities 5 10 15 20 --methods magnitude alpha_budget_b0.50_sd0.30 >> optuna_run\four_regime_sv_log.txt 2>&1

echo [%date% %time%] Regime A1 done, cooling 60s... >> optuna_run\four_regime_sv_log.txt
C:\Python314\python.exe -c "import torch,gc;gc.collect();torch.cuda.empty_cache();print('GPU flushed')" >> optuna_run\four_regime_sv_log.txt 2>&1
timeout /t 60 /nobreak >nul

REM Regime 2: splus at s=25-40%%
C:\Python314\python.exe optuna_run\magnitude_rmt_sweep.py --eval-batches 250 --out-file rmt_cache\four_regime_nosv.json --sparsities 25 30 35 40 --methods magnitude splus_budget_b1.00_sd0.50 >> optuna_run\four_regime_sv_log.txt 2>&1

echo [%date% %time%] Regime A2 done, cooling 60s... >> optuna_run\four_regime_sv_log.txt
C:\Python314\python.exe -c "import torch,gc;gc.collect();torch.cuda.empty_cache();print('GPU flushed')" >> optuna_run\four_regime_sv_log.txt 2>&1
timeout /t 60 /nobreak >nul

REM Regime 3: splus at s=45-55%%
C:\Python314\python.exe optuna_run\magnitude_rmt_sweep.py --eval-batches 250 --out-file rmt_cache\four_regime_nosv.json --sparsities 45 50 55 --methods magnitude splus_budget_b1.25_sd0.70_p1.0 >> optuna_run\four_regime_sv_log.txt 2>&1

echo [%date% %time%] Regime A3 done, cooling 60s... >> optuna_run\four_regime_sv_log.txt
C:\Python314\python.exe -c "import torch,gc;gc.collect();torch.cuda.empty_cache();print('GPU flushed')" >> optuna_run\four_regime_sv_log.txt 2>&1
timeout /t 60 /nobreak >nul

REM Regime 4: splus layertype at s=60-75%%
C:\Python314\python.exe optuna_run\magnitude_rmt_sweep.py --eval-batches 250 --out-file rmt_cache\four_regime_nosv.json --sparsities 60 65 70 75 --methods magnitude splus_budget_b1.50_sd0.85_p1.0_ba1.00_bm2.00 >> optuna_run\four_regime_sv_log.txt 2>&1

echo [%date% %time%] Part A done, cooling 60s before Part B (with SV)... >> optuna_run\four_regime_sv_log.txt
C:\Python314\python.exe -c "import torch,gc;gc.collect();torch.cuda.empty_cache();print('GPU flushed')" >> optuna_run\four_regime_sv_log.txt 2>&1
timeout /t 60 /nobreak >nul

REM ====== PART B: 4-regime WITH SV pruning ======

REM Regime 1: alpha + SV at s=5-20%%
C:\Python314\python.exe optuna_run\magnitude_rmt_sweep.py --eval-batches 250 --sv-prune --out-file rmt_cache\four_regime_sv.json --sparsities 5 10 15 20 --methods magnitude alpha_budget_b0.50_sd0.30 >> optuna_run\four_regime_sv_log.txt 2>&1

echo [%date% %time%] Regime B1 done, cooling 60s... >> optuna_run\four_regime_sv_log.txt
C:\Python314\python.exe -c "import torch,gc;gc.collect();torch.cuda.empty_cache();print('GPU flushed')" >> optuna_run\four_regime_sv_log.txt 2>&1
timeout /t 60 /nobreak >nul

REM Regime 2: splus + SV at s=25-40%%
C:\Python314\python.exe optuna_run\magnitude_rmt_sweep.py --eval-batches 250 --sv-prune --out-file rmt_cache\four_regime_sv.json --sparsities 25 30 35 40 --methods magnitude splus_budget_b1.00_sd0.50 >> optuna_run\four_regime_sv_log.txt 2>&1

echo [%date% %time%] Regime B2 done, cooling 60s... >> optuna_run\four_regime_sv_log.txt
C:\Python314\python.exe -c "import torch,gc;gc.collect();torch.cuda.empty_cache();print('GPU flushed')" >> optuna_run\four_regime_sv_log.txt 2>&1
timeout /t 60 /nobreak >nul

REM Regime 3: splus + SV at s=45-55%%
C:\Python314\python.exe optuna_run\magnitude_rmt_sweep.py --eval-batches 250 --sv-prune --out-file rmt_cache\four_regime_sv.json --sparsities 45 50 55 --methods magnitude splus_budget_b1.25_sd0.70_p1.0 >> optuna_run\four_regime_sv_log.txt 2>&1

echo [%date% %time%] Regime B3 done, cooling 60s... >> optuna_run\four_regime_sv_log.txt
C:\Python314\python.exe -c "import torch,gc;gc.collect();torch.cuda.empty_cache();print('GPU flushed')" >> optuna_run\four_regime_sv_log.txt 2>&1
timeout /t 60 /nobreak >nul

REM Regime 4: splus layertype + SV at s=60-75%%
C:\Python314\python.exe optuna_run\magnitude_rmt_sweep.py --eval-batches 250 --sv-prune --out-file rmt_cache\four_regime_sv.json --sparsities 60 65 70 75 --methods magnitude splus_budget_b1.50_sd0.85_p1.0_ba1.00_bm2.00 >> optuna_run\four_regime_sv_log.txt 2>&1

echo [%date% %time%] ALL DONE >> optuna_run\four_regime_sv_log.txt

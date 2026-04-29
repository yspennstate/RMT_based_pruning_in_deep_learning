@echo off
REM ================================================================
REM Refined4 sweep — autonomous overnight run.
REM Waits for refined3 to complete, then runs 3 phases sequentially.
REM Total compute: ~6h + up to ~1.2h waiting = fits under 8h window.
REM
REM Phase 1 (2.4h): Champions re-eval at 250-batch full eval
REM   14 methods x 11 sparsities = 154 cells, definitive leaderboard
REM
REM Phase 2 (2.25h): Ultra-fine beta x sd grid at cliff
REM   9 beta x 5 sd x 4 sparsities = 180 cells, 150-batch screening
REM
REM Phase 3 (1.25h): Full 5x5 layer-type beta grid
REM   25 methods x 4 sparsities = 100 cells, 150-batch screening
REM
REM Output files (separate per phase, atomic per-cell saves):
REM   rmt_cache\splus_refined4_phase1_results.json
REM   rmt_cache\splus_refined4_phase2_results.json
REM   rmt_cache\splus_refined4_phase3_results.json
REM ================================================================
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d C:\Users\owner\Projects\rmt_pruning_vit

echo [%date% %time%] Refined4 launched. Waiting for refined3 to complete... >> optuna_run\splus_refined4_bat_log.txt

REM --- Wait for refined3 to write its completion marker ---
set /a WAIT_COUNT=0
:WAIT_LOOP
ping -n 31 127.0.0.1 >nul
set /a WAIT_COUNT+=1
findstr /c:"Refined3 complete" optuna_run\splus_refined3_bat_log.txt >nul 2>&1
if %ERRORLEVEL% EQU 0 goto REFINED3_DONE
if %WAIT_COUNT% GEQ 150 goto REFINED3_DONE
goto WAIT_LOOP

:REFINED3_DONE
echo [%date% %time%] Proceeding after %WAIT_COUNT% wait cycles (~%WAIT_COUNT%0 sec) >> optuna_run\splus_refined4_bat_log.txt
ping -n 11 127.0.0.1 >nul
echo [%date% %time%] Starting Phase 1: champions re-eval at 250-batch... >> optuna_run\splus_refined4_bat_log.txt

REM ================================================================
REM PHASE 1: Champions re-eval at full 250-batch (~2.4h)
REM 14 methods x 11 sparsities = 154 cells
REM Sparsities: 30, 40, 50, 55, 60, 62.5, 65, 67.5, 70, 72.5, 75
REM ================================================================
C:\Python314\python.exe optuna_run\magnitude_rmt_sweep.py ^
  --eval-batches 250 ^
  --out-file rmt_cache\splus_refined4_phase1_results.json ^
  --sparsities 30 40 50 55 60 62.5 65 67.5 70 72.5 75 ^
  --methods ^
    magnitude ^
    splus_budget_b1.00_sd0.50 ^
    splus_budget_b1.00_sd0.70 ^
    splus_budget_b1.25_sd0.70_p1.0 ^
    splus_budget_b1.25_sd0.85_p1.0 ^
    splus_budget_b1.50_sd0.85_p1.0 ^
    splus_budget_b1.25_sd0.95_p1.0 ^
    splus_budget_b1.50_sd0.95_p1.0 ^
    splus_budget_b1.75_sd0.85_p1.0 ^
    splus_budget_b1.30_sd0.85_p1.0 ^
    splus_budget_b1.30_sd0.90_p1.0 ^
    splus_budget_b1.30_sd0.95_p1.0 ^
    splus_budget_b1.40_sd0.85_p1.0 ^
    splus_budget_b1.40_sd0.90_p1.0 ^
  >> optuna_run\splus_refined4_bat_log.txt 2>&1

echo [%date% %time%] Phase 1 done (exit %ERRORLEVEL%), starting Phase 2: ultra-fine beta x sd grid... >> optuna_run\splus_refined4_bat_log.txt
ping -n 11 127.0.0.1 >nul

REM ================================================================
REM PHASE 2: Ultra-fine beta x sd grid at cliff (~2.25h)
REM 9 beta x 5 sd x 4 sparsities = 180 cells, 150-batch screening
REM ================================================================
C:\Python314\python.exe optuna_run\magnitude_rmt_sweep.py ^
  --eval-batches 150 ^
  --out-file rmt_cache\splus_refined4_phase2_results.json ^
  --sparsities 60 62.5 65 67.5 ^
  --methods ^
    splus_budget_b1.20_sd0.82_p1.0 splus_budget_b1.20_sd0.85_p1.0 splus_budget_b1.20_sd0.88_p1.0 splus_budget_b1.20_sd0.91_p1.0 splus_budget_b1.20_sd0.94_p1.0 ^
    splus_budget_b1.25_sd0.82_p1.0 splus_budget_b1.25_sd0.85_p1.0 splus_budget_b1.25_sd0.88_p1.0 splus_budget_b1.25_sd0.91_p1.0 splus_budget_b1.25_sd0.94_p1.0 ^
    splus_budget_b1.30_sd0.82_p1.0 splus_budget_b1.30_sd0.85_p1.0 splus_budget_b1.30_sd0.88_p1.0 splus_budget_b1.30_sd0.91_p1.0 splus_budget_b1.30_sd0.94_p1.0 ^
    splus_budget_b1.35_sd0.82_p1.0 splus_budget_b1.35_sd0.85_p1.0 splus_budget_b1.35_sd0.88_p1.0 splus_budget_b1.35_sd0.91_p1.0 splus_budget_b1.35_sd0.94_p1.0 ^
    splus_budget_b1.40_sd0.82_p1.0 splus_budget_b1.40_sd0.85_p1.0 splus_budget_b1.40_sd0.88_p1.0 splus_budget_b1.40_sd0.91_p1.0 splus_budget_b1.40_sd0.94_p1.0 ^
    splus_budget_b1.45_sd0.82_p1.0 splus_budget_b1.45_sd0.85_p1.0 splus_budget_b1.45_sd0.88_p1.0 splus_budget_b1.45_sd0.91_p1.0 splus_budget_b1.45_sd0.94_p1.0 ^
    splus_budget_b1.50_sd0.82_p1.0 splus_budget_b1.50_sd0.85_p1.0 splus_budget_b1.50_sd0.88_p1.0 splus_budget_b1.50_sd0.91_p1.0 splus_budget_b1.50_sd0.94_p1.0 ^
    splus_budget_b1.55_sd0.82_p1.0 splus_budget_b1.55_sd0.85_p1.0 splus_budget_b1.55_sd0.88_p1.0 splus_budget_b1.55_sd0.91_p1.0 splus_budget_b1.55_sd0.94_p1.0 ^
    splus_budget_b1.60_sd0.82_p1.0 splus_budget_b1.60_sd0.85_p1.0 splus_budget_b1.60_sd0.88_p1.0 splus_budget_b1.60_sd0.91_p1.0 splus_budget_b1.60_sd0.94_p1.0 ^
  >> optuna_run\splus_refined4_bat_log.txt 2>&1

echo [%date% %time%] Phase 2 done (exit %ERRORLEVEL%), starting Phase 3: layer-type beta 5x5 grid... >> optuna_run\splus_refined4_bat_log.txt
ping -n 11 127.0.0.1 >nul

REM ================================================================
REM PHASE 3: Full layer-type beta 5x5 grid (~1.25h)
REM 25 methods x 4 sparsities = 100 cells, 150-batch screening
REM All at sd=0.85, p=1.0 (current champion point)
REM ================================================================
C:\Python314\python.exe optuna_run\magnitude_rmt_sweep.py ^
  --eval-batches 150 ^
  --out-file rmt_cache\splus_refined4_phase3_results.json ^
  --sparsities 60 62.5 65 67.5 ^
  --methods ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.00_bm1.00 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.00_bm1.25 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.00_bm1.50 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.00_bm1.75 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.00_bm2.00 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.25_bm1.00 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.25_bm1.25 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.25_bm1.50 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.25_bm1.75 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.25_bm2.00 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.50_bm1.00 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.50_bm1.25 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.50_bm1.50 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.50_bm1.75 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.50_bm2.00 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.75_bm1.00 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.75_bm1.25 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.75_bm1.50 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.75_bm1.75 ^
    splus_budget_b1.50_sd0.85_p1.0_ba1.75_bm2.00 ^
    splus_budget_b1.50_sd0.85_p1.0_ba2.00_bm1.00 ^
    splus_budget_b1.50_sd0.85_p1.0_ba2.00_bm1.25 ^
    splus_budget_b1.50_sd0.85_p1.0_ba2.00_bm1.50 ^
    splus_budget_b1.50_sd0.85_p1.0_ba2.00_bm1.75 ^
    splus_budget_b1.50_sd0.85_p1.0_ba2.00_bm2.00 ^
  >> optuna_run\splus_refined4_bat_log.txt 2>&1

echo [%date% %time%] REFINED4 ALL PHASES COMPLETE (last exit %ERRORLEVEL%) >> optuna_run\splus_refined4_bat_log.txt

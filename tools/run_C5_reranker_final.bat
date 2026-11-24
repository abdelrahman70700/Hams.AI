@echo off
REM Step C5: final reranker candidate - low learning rate (2e-6), fp32, denoised negatives, 2 epochs.
call "%~dp0env.bat"
cd /d "%ROOT%"
set "LOG=%ROOT%\tools\logs\run_C5.log"
title Hams.AI step C5 - final reranker candidate
echo [%date% %time%] START > "%LOG%"
"%PY%" scripts\train_reranker.py --lr 2e-6 --warmup 0.1 --no-bf16 --epochs 2 --eval-steps 20 --denoise-margin 1.0 --skip-top 1 --negatives 4 --tag lr2e6_dn >> "%LOG%" 2>&1
echo [%date% %time%] DONE >> "%LOG%"

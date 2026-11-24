@echo off
REM Step C3: reranker fine-tuning with DENOISED hard negatives (base-reranker margin filter, skip top-1),
REM for GTE (primary) and mMiniLM (backup, XLM-R architecture).
call "%~dp0env.bat"
cd /d "%ROOT%"
set "LOG=%ROOT%\tools\logs\run_C3.log"
title Hams.AI step C3 - reranker with denoised negatives
echo [%date% %time%] START > "%LOG%"
"%PY%" scripts\train_reranker.py --denoise-margin 1.0 --skip-top 1 --negatives 4 --lr 1e-5 --warmup 0.1 --epochs 2 --eval-steps 20 --tag dn >> "%LOG%" 2>&1
"%PY%" scripts\train_reranker.py --base mminilm --denoise-margin 1.0 --skip-top 1 --negatives 4 --lr 2e-5 --warmup 0.1 --epochs 2 --eval-steps 20 --tag "" >> "%LOG%" 2>&1
echo [%date% %time%] DONE >> "%LOG%"

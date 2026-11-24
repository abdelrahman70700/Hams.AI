@echo off
REM Step C2: LLM throughput smoke test, then a small reranker fine-tuning sweep (lower LR, step-wise eval).
call "%~dp0env.bat"
cd /d "%ROOT%"
set "LOG=%ROOT%\tools\logs\run_C2.log"
title Hams.AI step C2 - LLM smoke test + reranker sweep
echo [%date% %time%] START > "%LOG%"
call "%ROOT%\tools\smoke_llm.bat"
"%PY%" scripts\train_reranker.py --lr 5e-6 --warmup 0.2 --no-bf16 --epochs 1 --eval-steps 15 --tag lr5e6_fp32 >> "%LOG%" 2>&1
"%PY%" scripts\train_reranker.py --lr 5e-6 --warmup 0.2 --epochs 1 --eval-steps 15 --tag lr5e6_bf16 >> "%LOG%" 2>&1
"%PY%" scripts\train_reranker.py --lr 2e-6 --warmup 0.2 --no-bf16 --epochs 1 --eval-steps 15 --tag lr2e6_fp32 >> "%LOG%" 2>&1
echo [%date% %time%] DONE >> "%LOG%"

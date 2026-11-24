@echo off
REM Step C: re-tune fusion for the fine-tuned encoder (validation), fine-tune the reranker,
REM select fusion on validation, run the held-out test benchmark, calibrate the abstention gate.
call "%~dp0env.bat"
cd /d "%ROOT%"
set "LOG=%ROOT%\tools\logs\run_C.log"
title Hams.AI step C - reranker fine-tuning
echo [%date% %time%] START > "%LOG%"
"%PY%" scripts\prepare_data.py >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\benchmark_retrieval.py --embedder e5-base-ft --split validation --tune --tag tune_ft >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\train_reranker.py >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\benchmark_retrieval.py --embedder e5-base-ft --reranker gte-rerank-ft --split validation --fusion rrf --tag val_rrf >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\benchmark_retrieval.py --embedder e5-base-ft --reranker gte-rerank-ft --split validation --fusion union --tag val_union >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\select_fusion.py >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\benchmark_retrieval.py --embedder e5-base-ft --reranker gte-rerank-ft --tag after_finetuning >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\calibrate_gate.py --embedder e5-base-ft --reranker gte-rerank-ft --write --tag after >> "%LOG%" 2>&1 || goto :fail
REM same final pipeline configuration with the BASE models -> fair "before" numbers
"%PY%" scripts\benchmark_retrieval.py --embedder e5-base --reranker gte-rerank --tag before_finetuning >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\calibrate_gate.py --embedder e5-base --reranker gte-rerank --tag before >> "%LOG%" 2>&1 || goto :fail
echo [%date% %time%] DONE >> "%LOG%"
exit /b 0
:fail
echo [%date% %time%] FAILED >> "%LOG%"
exit /b 1

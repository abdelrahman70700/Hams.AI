@echo off
REM Step A1: deps check, prepare data, BM25 index, e5-base dense index, RRF tuning, "before fine-tuning" baseline.
call "%~dp0env.bat"
cd /d "%ROOT%"
set "LOG=%ROOT%\tools\logs\run_A1.log"
title Hams.AI step A1 - baseline
echo [%date% %time%] START > "%LOG%"
"%PY%" -m pip install -q -r requirements.txt hf_xet >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\prepare_data.py >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\build_index.py --lexical-only --fresh >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\build_index.py --embedder e5-base >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\benchmark_retrieval.py --embedder e5-base --split validation --tune --tag tune_base >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\benchmark_retrieval.py --embedder e5-base --reranker gte-rerank --tag before_finetuning >> "%LOG%" 2>&1 || goto :fail
echo [%date% %time%] DONE >> "%LOG%"
exit /b 0
:fail
echo [%date% %time%] FAILED >> "%LOG%"
exit /b 1

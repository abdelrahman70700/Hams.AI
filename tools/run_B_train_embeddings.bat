@echo off
REM Step B: fine-tune the embedding model, build its index, evaluate retrieval after fine-tuning.
call "%~dp0env.bat"
cd /d "%ROOT%"
set "LOG=%ROOT%\tools\logs\run_B.log"
title Hams.AI step B - embedding fine-tuning
echo [%date% %time%] START > "%LOG%"
"%PY%" scripts\train_embeddings.py >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\build_index.py --embedder e5-base-ft >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\benchmark_retrieval.py --embedder e5-base-ft --reranker gte-rerank --tag after_embedding_ft >> "%LOG%" 2>&1 || goto :fail
echo [%date% %time%] DONE >> "%LOG%"
exit /b 0
:fail
echo [%date% %time%] FAILED >> "%LOG%"
exit /b 1

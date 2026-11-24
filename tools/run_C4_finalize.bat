@echo off
REM Step C4: select the best fine-tuned reranker on validation (-> reranker-ft), select fusion on validation,
REM run the held-out test benchmark, calibrate the gate, re-run "before" with the same final config.
call "%~dp0env.bat"
cd /d "%ROOT%"
set "RR=reranker-ft"
set "LOG=%ROOT%\tools\logs\run_C4.log"
title Hams.AI step C4 - finalize retrieval (%RR%)
echo [%date% %time%] START > "%LOG%"
"%PY%" scripts\select_reranker.py >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\benchmark_retrieval.py --embedder e5-base-ft --reranker %RR% --split validation --fusion rrf --tag val_rrf >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\benchmark_retrieval.py --embedder e5-base-ft --reranker %RR% --split validation --fusion union --tag val_union >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\select_fusion.py >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\benchmark_retrieval.py --embedder e5-base-ft --reranker %RR% --tag after_finetuning >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\calibrate_gate.py --embedder e5-base-ft --reranker %RR% --write --tag after >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\benchmark_retrieval.py --embedder e5-base --reranker gte-rerank --tag before_finetuning >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\calibrate_gate.py --embedder e5-base --reranker gte-rerank --tag before >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\make_report.py >> "%LOG%" 2>&1
echo [%date% %time%] DONE >> "%LOG%"
exit /b 0
:fail
echo [%date% %time%] FAILED >> "%LOG%"
exit /b 1

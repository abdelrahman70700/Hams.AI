@echo off
REM Step A2: zero-shot dense bake-off (model selection evidence) + zero-shot reference rerankers.
call "%~dp0env.bat"
cd /d "%ROOT%"
set "LOG=%ROOT%\tools\logs\run_A2.log"
title Hams.AI step A2 - bake-off
echo [%date% %time%] START > "%LOG%"
"%PY%" scripts\benchmark_retrieval.py --embedders e5-small e5-base gte-base --tag zeroshot_bakeoff >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\benchmark_retrieval.py --embedder e5-base --reranker mminilm --tag zeroshot_mminilm >> "%LOG%" 2>&1 || goto :fail
echo [%date% %time%] DONE >> "%LOG%"
exit /b 0
:fail
echo [%date% %time%] FAILED >> "%LOG%"
exit /b 1

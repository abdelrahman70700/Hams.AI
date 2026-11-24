@echo off
REM Step D2: regenerate data (label fix), re-run retrieval benchmarks, then generation benchmarks after/before
REM with the improved prompt and abstention detection. Starts llama-server if needed.
call "%~dp0env.bat"
cd /d "%ROOT%"
set "LOG=%ROOT%\tools\logs\run_D2.log"
title Hams.AI step D2 - benchmarks
echo [%date% %time%] START > "%LOG%"
"%PY%" scripts\prepare_data.py >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\benchmark_retrieval.py --embedder e5-base-ft --reranker reranker-ft --tag after_finetuning >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\benchmark_retrieval.py --embedder e5-base --reranker gte-rerank --tag before_finetuning >> "%LOG%" 2>&1 || goto :fail
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health -TimeoutSec 2).StatusCode } catch { 0 }" | findstr 200 >nul && goto :llm_up
start "llama-server" cmd /c ""%ROOT%\tools\start_llm.bat""
:wait_llm
timeout /t 3 >nul
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health -TimeoutSec 2).StatusCode } catch { 0 }" | findstr 200 >nul || goto wait_llm
:llm_up
"%PY%" scripts\benchmark_generation.py --tag after --judge >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\benchmark_generation.py --tag before --embedder e5-base --reranker gte-rerank --gate-file results/gate_calibration_before.json --judge >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\make_report.py >> "%LOG%" 2>&1
echo [%date% %time%] DONE >> "%LOG%"
exit /b 0
:fail
echo [%date% %time%] FAILED >> "%LOG%"
exit /b 1

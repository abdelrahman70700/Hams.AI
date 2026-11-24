@echo off
REM Step E: latency benchmarks (in-process, before the API starts so models are loaded once),
REM then start the API and run the concurrency benchmark through HTTP. Leaves llama-server + API running.
call "%~dp0env.bat"
cd /d "%ROOT%"
set "LOG=%ROOT%\tools\logs\run_E.log"
title Hams.AI step E - latency and concurrency
echo [%date% %time%] START > "%LOG%"
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health -TimeoutSec 2).StatusCode } catch { 0 }" | findstr 200 >nul && goto :llm_up
start "llama-server" cmd /c ""%ROOT%\tools\start_llm.bat""
:wait_llm
timeout /t 3 >nul
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health -TimeoutSec 2).StatusCode } catch { 0 }" | findstr 200 >nul || goto wait_llm
:llm_up
"%PY%" scripts\benchmark_latency.py --tag after >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\benchmark_latency.py --tag before --quick --embedder e5-base --reranker gte-rerank >> "%LOG%" 2>&1 || goto :fail
set "HAMS_DEVICE=cpu"
"%PY%" scripts\benchmark_latency.py --tag cpu_only --quick --no-llm --n 40 >> "%LOG%" 2>&1 || goto :fail
set "HAMS_DEVICE="
start "Hams.AI API" cmd /c ""%PY%" demo\app.py > "%ROOT%\tools\logs\api.log" 2>&1"
:wait_api
timeout /t 3 >nul
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8010/health -TimeoutSec 5).Content } catch { 0 }" | findstr chunks >nul || goto wait_api
"%PY%" scripts\benchmark_concurrency.py --levels 1,2,4,8,16 --requests 16 >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\make_report.py >> "%LOG%" 2>&1
echo [%date% %time%] DONE >> "%LOG%"
exit /b 0
:fail
echo [%date% %time%] FAILED >> "%LOG%"
exit /b 1

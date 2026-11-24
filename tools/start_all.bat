@echo off
REM Start the demo: llama-server (if not already running) + the Hams.AI API, then open the web page.
REM   tools\start_all.bat        laptop profile (Qwen3.5-2B, 8k context, 2 parallel slots)
REM   tools\start_all.bat l4     L4 profile (Qwen3.5-4B, 16k context, 8 slots) - model must be downloaded first
call "%~dp0env.bat"
cd /d "%ROOT%"
title Hams.AI launcher
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health -TimeoutSec 2).StatusCode } catch { 0 }" | findstr 200 >nul && goto :llm_up
if /i "%~1"=="l4" (
  start "llama-server" cmd /c ""%ROOT%\tools\start_llm.bat" Qwen3.5-4B-Q4_K_M.gguf 99 16384 8"
) else (
  start "llama-server" cmd /c ""%ROOT%\tools\start_llm.bat" Qwen3.5-2B-Q4_K_M.gguf 99 8192 2"
)
echo Starting the answer model (llama-server) ...
:wait_llm
timeout /t 2 >nul
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health -TimeoutSec 2).StatusCode } catch { 0 }" | findstr 200 >nul || goto wait_llm
:llm_up
echo Answer model is up.
REM restart our own API window so it always runs the current code
taskkill /FI "WINDOWTITLE eq Hams.AI API*" /T /F >nul 2>&1
start "Hams.AI API" /min cmd /c ""%PY%" demo\app.py > "%ROOT%\tools\logs\api.log" 2>&1"
echo Starting the search API (loads the models, about 20-40 seconds) ...
set /a TRIES=0
:wait_api
timeout /t 2 >nul
set /a TRIES+=1
if %TRIES% GEQ 90 (echo The API did not start. See tools\logs\api.log & pause & exit /b 1)
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8010/health -TimeoutSec 5).Content } catch { 0 }" | findstr chunks >nul || goto wait_api
echo.
echo Ready: http://127.0.0.1:8010   (API documentation: http://127.0.0.1:8010/docs)
echo Keep the "llama-server" and "Hams.AI API" windows open while you use the demo.
start "" http://127.0.0.1:8010

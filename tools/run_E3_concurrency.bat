@echo off
REM Step E3: concurrency 1/2/4/8/16 against the API (starts llama-server and the API if needed) + report.
call "%~dp0env.bat"
cd /d "%ROOT%"
set "LOG=%ROOT%\tools\logs\run_E3.log"
title Hams.AI step E3 - concurrency
echo [%date% %time%] START > "%LOG%"
"%PY%" -m pip install -q -r requirements.txt >> "%LOG%" 2>&1 || goto :fail
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health -TimeoutSec 2).StatusCode } catch { 0 }" | findstr 200 >nul && goto :llm_up
start "llama-server" cmd /c ""%ROOT%\tools\start_llm.bat""
:wait_llm
timeout /t 3 >nul
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health -TimeoutSec 2).StatusCode } catch { 0 }" | findstr 200 >nul || goto wait_llm
:llm_up
echo [llm] up >> "%LOG%"
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8010/health -TimeoutSec 2).Content } catch { 0 }" | findstr chunks >nul && goto :api_up
start "Hams.AI API" /min cmd /c ""%PY%" -u demo\app.py > "%ROOT%\tools\logs\api.log" 2>&1"
set /a TRIES=0
:wait_api
timeout /t 3 >nul
set /a TRIES+=1
if %TRIES% GEQ 60 (echo [api] did not start within 3 minutes, see tools\logs\api.log >> "%LOG%" & goto :fail)
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8010/health -TimeoutSec 5).Content } catch { 0 }" | findstr chunks >nul || goto wait_api
:api_up
echo [api] up >> "%LOG%"
"%PY%" scripts\benchmark_concurrency.py --levels 1,2,4,8,16 --requests 16 >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\make_report.py >> "%LOG%" 2>&1
echo [%date% %time%] DONE >> "%LOG%"
exit /b 0
:fail
echo [%date% %time%] FAILED >> "%LOG%"
exit /b 1

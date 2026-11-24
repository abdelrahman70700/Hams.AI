@echo off
REM Film all live demo scenes at 4K (3840x2160): makes sure llama-server and the API are running first.
call "%~dp0env.bat"
cd /d "%ROOT%"
set "LOG=%ROOT%\tools\logs\run_I4.log"
title Hams.AI - filming the live demo at 4K
echo [%date% %time%] START > "%LOG%"
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health -TimeoutSec 2).StatusCode } catch { 0 }" | findstr 200 >nul && goto :llm_up
start "llama-server" cmd /c ""%ROOT%\tools\start_llm.bat""
:wait_llm
timeout /t 3 >nul
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health -TimeoutSec 2).StatusCode } catch { 0 }" | findstr 200 >nul || goto wait_llm
:llm_up
echo [llm] up >> "%LOG%"
taskkill /FI "WINDOWTITLE eq Hams.AI API*" /T /F >> "%LOG%" 2>&1
timeout /t 3 >nul
start "Hams.AI API" /min cmd /c ""%PY%" demo\app.py > "%ROOT%\tools\logs\api.log" 2>&1"
set /a TRIES=0
:wait_api
timeout /t 2 >nul
set /a TRIES+=1
if %TRIES% GEQ 90 (echo [api] did not start >> "%LOG%" & goto :fail)
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8010/health -TimeoutSec 5).Content } catch { 0 }" | findstr chunks >nul || goto wait_api
echo [api] up >> "%LOG%"
"%PY%" scripts\video\footage.py >> "%LOG%" 2>&1 || goto :fail
echo [%date% %time%] DONE >> "%LOG%"
exit /b 0
:fail
echo [%date% %time%] FAILED >> "%LOG%"
exit /b 1

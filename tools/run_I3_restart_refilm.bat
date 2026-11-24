@echo off
REM Restart the API (to load code changes), then re-film selected demo scenes (default: l08_conv).
call "%~dp0env.bat"
cd /d "%ROOT%"
set "LOG=%ROOT%\tools\logs\run_I3.log"
title Hams.AI - restart API and re-film
echo [%date% %time%] START %* > "%LOG%"
taskkill /FI "WINDOWTITLE eq Hams.AI API*" /T /F >> "%LOG%" 2>&1
timeout /t 3 >nul
start "Hams.AI API" /min cmd /c ""%PY%" demo\app.py > "%ROOT%\tools\logs\api.log" 2>&1"
set /a TRIES=0
:wait_api
timeout /t 2 >nul
set /a TRIES+=1
if %TRIES% GEQ 90 (echo [api] did not start >> "%LOG%" & goto :fail)
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8010/health -TimeoutSec 5).Content } catch { 0 }" | findstr chunks >nul || goto wait_api
set "SEGS=%*"
if "%SEGS%"=="" set "SEGS=l08_conv"
"%PY%" scripts\video\footage.py %SEGS% >> "%LOG%" 2>&1 || goto :fail
echo [%date% %time%] DONE >> "%LOG%"
exit /b 0
:fail
echo [%date% %time%] FAILED >> "%LOG%"
exit /b 1

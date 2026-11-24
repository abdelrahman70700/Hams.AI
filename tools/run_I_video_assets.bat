@echo off
REM Demo video assets: English narration (edge-tts neural voice) + live footage at 1920x1080 synced to it.
call "%~dp0env.bat"
cd /d "%ROOT%"
set "LOG=%ROOT%\tools\logs\run_I.log"
title Hams.AI - narration and live footage for the demo video
echo [%date% %time%] START > "%LOG%"
"%PY%" -m pip install -q "edge-tts>=7" playwright imageio-ffmpeg >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\video\tts.py >> "%LOG%" 2>&1 || goto :fail
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8010/health -TimeoutSec 5).Content } catch { 0 }" | findstr chunks >nul || (echo [api] not running - run tools\start_all.bat first >> "%LOG%" & goto :fail)
"%PY%" scripts\video\footage.py >> "%LOG%" 2>&1 || goto :fail
echo [%date% %time%] DONE >> "%LOG%"
exit /b 0
:fail
echo [%date% %time%] FAILED >> "%LOG%"
exit /b 1

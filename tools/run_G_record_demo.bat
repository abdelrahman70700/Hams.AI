@echo off
REM Records the demo video from the live system (Microsoft Edge, automated) -> demo_video\hams_demo.mp4
call "%~dp0env.bat"
cd /d "%ROOT%"
set "LOG=%ROOT%\tools\logs\run_G.log"
title Hams.AI - recording the demo video
echo [%date% %time%] START > "%LOG%"
"%PY%" -m pip install -q playwright imageio-ffmpeg markdown >> "%LOG%" 2>&1 || goto :fail
"%PY%" -m playwright install ffmpeg >> "%LOG%" 2>&1 || goto :fail
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health -TimeoutSec 2).StatusCode } catch { 0 }" | findstr 200 >nul && goto :llm_up
start "llama-server" cmd /c ""%ROOT%\tools\start_llm.bat""
:wait_llm
timeout /t 2 >nul
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health -TimeoutSec 2).StatusCode } catch { 0 }" | findstr 200 >nul || goto wait_llm
:llm_up
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8010/health -TimeoutSec 5).Content } catch { 0 }" | findstr chunks >nul && goto :api_up
start "Hams.AI API" /min cmd /c ""%PY%" demo\app.py > "%ROOT%\tools\logs\api.log" 2>&1"
set /a TRIES=0
:wait_api
timeout /t 2 >nul
set /a TRIES+=1
if %TRIES% GEQ 90 (echo [api] did not start >> "%LOG%" & goto :fail)
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8010/health -TimeoutSec 5).Content } catch { 0 }" | findstr chunks >nul || goto wait_api
:api_up
echo Recording (about 6 minutes, runs in the background - do not close this window) ...
"%PY%" scripts\record_demo.py >> "%LOG%" 2>&1 || goto :fail
"%PY%" -c "import imageio_ffmpeg,subprocess;f=imageio_ffmpeg.get_ffmpeg_exe();subprocess.run([f,'-y','-loglevel','error','-i','demo_video/hams_demo.webm','-c:v','libx264','-crf','20','-preset','veryfast','-pix_fmt','yuv420p','-r','30','-movflags','+faststart','demo_video/hams_demo.mp4'],check=True)" >> "%LOG%" 2>&1 || goto :fail
echo [%date% %time%] DONE >> "%LOG%"
explorer "%ROOT%\demo_video"
exit /b 0
:fail
echo [%date% %time%] FAILED >> "%LOG%"
exit /b 1

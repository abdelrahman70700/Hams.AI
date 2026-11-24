@echo off
REM Final run: re-measure the fine-tuned answers (incl. conversations) with the restored rewrite prompt,
REM refresh error analysis + README, restart the API, test conversations, and record the demo video.
call "%~dp0env.bat"
cd /d "%ROOT%"
set "LOG=%ROOT%\tools\logs\run_H.log"
title Hams.AI - final re-measure and demo recording
echo [%date% %time%] START > "%LOG%"
taskkill /FI "WINDOWTITLE eq Hams.AI API*" /T /F >> "%LOG%" 2>&1
timeout /t 3 >nul
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health -TimeoutSec 2).StatusCode } catch { 0 }" | findstr 200 >nul && goto :llm_up
start "llama-server" cmd /c ""%ROOT%\tools\start_llm.bat""
:wait_llm
timeout /t 3 >nul
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health -TimeoutSec 2).StatusCode } catch { 0 }" | findstr 200 >nul || goto wait_llm
:llm_up
echo [step] generation benchmark (fine-tuned) >> "%LOG%"
"%PY%" scripts\benchmark_generation.py --tag after --judge --no-relevance >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\benchmark_generation.py --tag after --rescore --relevance >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\error_analysis.py --tag after >> "%LOG%" 2>&1 || goto :fail
"%PY%" scripts\make_report.py >> "%LOG%" 2>&1 || goto :fail
echo [step] start API >> "%LOG%"
start "Hams.AI API" /min cmd /c ""%PY%" demo\app.py > "%ROOT%\tools\logs\api.log" 2>&1"
set /a TRIES=0
:wait_api
timeout /t 2 >nul
set /a TRIES+=1
if %TRIES% GEQ 90 (echo [api] did not start >> "%LOG%" & goto :fail)
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8010/health -TimeoutSec 5).Content } catch { 0 }" | findstr chunks >nul || goto wait_api
echo [step] probe conversations >> "%LOG%"
"%PY%" scripts\probe_conversations.py > "%ROOT%\tools\logs\probe_conv.log" 2>&1
echo [step] record demo >> "%LOG%"
"%PY%" scripts\record_demo.py >> "%LOG%" 2>&1 || goto :fail
"%PY%" -c "import imageio_ffmpeg,subprocess;f=imageio_ffmpeg.get_ffmpeg_exe();subprocess.run([f,'-y','-loglevel','error','-i','demo_video/hams_demo.webm','-c:v','libx264','-crf','20','-preset','veryfast','-pix_fmt','yuv420p','-r','30','-movflags','+faststart','demo_video/hams_demo_final.mp4'],check=True)" >> "%LOG%" 2>&1 || goto :fail
echo [%date% %time%] DONE >> "%LOG%"
exit /b 0
:fail
echo [%date% %time%] FAILED >> "%LOG%"
exit /b 1

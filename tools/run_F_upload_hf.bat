@echo off
REM Step F: upload the two fine-tuned models (with model cards) to Hugging Face as public model repositories.
REM Log in first with tools\hf_login.bat.   Usage: tools\run_F_upload_hf.bat [hf-username]
call "%~dp0env.bat"
cd /d "%ROOT%"
set "HFUSER=%~1"
if "%HFUSER%"=="" set "HFUSER=AbdelrahmanHassan"
set "LOG=%ROOT%\tools\logs\run_F.log"
title Hams.AI step F - upload to Hugging Face as %HFUSER%
echo [%date% %time%] START user=%HFUSER% > "%LOG%"
"%PY%" scripts\upload_to_hf.py --user %HFUSER% >> "%LOG%" 2>&1 || goto :fail
echo [%date% %time%] DONE >> "%LOG%"
type "%LOG%"
pause
exit /b 0
:fail
echo [%date% %time%] FAILED >> "%LOG%"
type "%LOG%"
pause
exit /b 1

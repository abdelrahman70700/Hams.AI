@echo off
REM Re-film selected live scenes for the demo video, e.g.:  run_I2_refilm.bat l08_conv
call "%~dp0env.bat"
cd /d "%ROOT%"
set "LOG=%ROOT%\tools\logs\run_I2.log"
title Hams.AI - re-filming demo scenes
echo [%date% %time%] START %* > "%LOG%"
set "SEGS=%*"
if "%SEGS%"=="" set "SEGS=l08_conv"
"%PY%" scripts\video\footage.py %SEGS% >> "%LOG%" 2>&1 || goto :fail
echo [%date% %time%] DONE >> "%LOG%"
exit /b 0
:fail
echo [%date% %time%] FAILED >> "%LOG%"
exit /b 1

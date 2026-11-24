@echo off
REM Builds D:\Hams.AI_delivery with only what is handed in:
REM   github\        the repository (code, data, results, docs) - no environment, models, indexes or logs
REM   huggingface\   the two fine-tuned models, each with its model card, ready to upload
call "%~dp0env.bat"
cd /d "%ROOT%"
set "OUT=%ROOT%_delivery"
REM optional: tools\make_delivery.bat <hf-username>  (used in the model cards' usage examples)
set "HFUSER=%~1"
if "%HFUSER%"=="" set "HFUSER=AbdelrahmanHassan"
title Hams.AI - building the delivery folder
echo Copying the repository to %OUT%\github ...
if exist "%OUT%\github" rmdir /s /q "%OUT%\github"
robocopy "%ROOT%" "%OUT%\github" /E /NFL /NDL /NJH /NJS /NP ^
  /XD "%ROOT%\.env" "%ROOT%\.cache" "%ROOT%\bin" "%ROOT%\models" "%ROOT%\data\index" "%ROOT%\data\corpus\uploads" "%ROOT%\tools\logs" "%ROOT%\demo_video" __pycache__ .pytest_cache ^
  /XF hams_sync.tgz *.vmold* *.new *.pyc ^
      env_report.txt env_report2.txt check_env.bat check_env2.bat check_env.ps1 check_download.bat ^
      run_A_baselines.bat run_D_generation.bat run_D3_remeasure.bat run_D4_remeasure_after.bat ^
      run_E2_cpu_concurrency.bat run_E4_latency_fair.bat download_encoders.bat download_llm.bat ^
      sessions.sqlite retrieval_before_finetuning_defaultcfg.json clean_delivery.bat
if %ERRORLEVEL% GEQ 8 (echo Copy failed. & pause & exit /b 1)
echo Exporting the two fine-tuned models to %OUT%\huggingface ...
"%PY%" scripts\upload_to_hf.py --user %HFUSER% --export "%OUT%\huggingface" || (echo Export failed. & pause & exit /b 1)
echo.
echo Done: %OUT%
echo   github\       push this folder to your GitHub repository
echo   huggingface\  upload each subfolder as a model repository (or run tools\run_F_upload_hf.bat)
explorer "%OUT%"

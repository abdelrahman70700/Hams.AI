@echo off
setlocal
chcp 65001 >nul
call "%~dp0env.bat"
if not exist "%ROOT%\tools\logs" mkdir "%ROOT%\tools\logs"
set "LOG=%ROOT%\tools\logs\setup.log"
set "STATUS=%ROOT%\tools\logs\setup_status.txt"
echo RUNNING > "%STATUS%"
echo [%date% %time%] setup start > "%LOG%"
title Hams.AI setup - do not close

echo === 1/6 conda env (python 3.11, conda-forge) ===
if not exist "%PY%" (
  call F:\conda\condabin\conda.bat create -y -p "%ROOT%\.env" -c conda-forge --override-channels python=3.11 pip >> "%LOG%" 2>&1
)
if not exist "%PY%" ( echo FAILED_CONDA > "%STATUS%" & goto :eof )
echo [%date% %time%] conda ok >> "%LOG%"

echo === 2/6 torch CUDA 12.8 ===
"%PY%" -m pip install --upgrade pip >> "%LOG%" 2>&1
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu128 >> "%LOG%" 2>&1
echo [%date% %time%] torch step done >> "%LOG%"

echo === 3/6 project requirements ===
"%PY%" -m pip install -r "%ROOT%\requirements.txt" >> "%LOG%" 2>&1
echo [%date% %time%] requirements step done >> "%LOG%"

echo === 4/6 verify ===
"%PY%" "%ROOT%\tools\verify_env.py" > "%ROOT%\tools\logs\verify_env.txt" 2>&1
echo [%date% %time%] verify done >> "%LOG%"

echo === 5/6 llama.cpp CUDA build ===
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\tools\get_llama_cpp.ps1" -Dest "%LLAMA_DIR%" >> "%LOG%" 2>&1
echo [%date% %time%] llama.cpp step done >> "%LOG%"

echo === 6/6 models (GGUF + encoders) ===
"%PY%" "%ROOT%\tools\download_models.py" >> "%LOG%" 2>&1
echo [%date% %time%] models step done >> "%LOG%"

echo DONE > "%STATUS%"
echo [%date% %time%] setup finished >> "%LOG%"
echo Setup finished. You can close this window.
endlocal

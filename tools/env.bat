@echo off
REM Shared environment for every project script. Keeps all caches on D: (C: is nearly full).
set "ROOT=D:\Hams.AI"
set "PY=%ROOT%\.env\python.exe"
set "HF_HOME=%ROOT%\.cache\hf"
set "PIP_CACHE_DIR=%ROOT%\.cache\pip"
set "TORCH_HOME=%ROOT%\.cache\torch"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "LLAMA_DIR=%ROOT%\bin\llama"
set "MODELS_DIR=%ROOT%\models"
set "PATH=%ROOT%\.env;%ROOT%\.env\Scripts;%LLAMA_DIR%;%PATH%"

@echo off
call "%~dp0env.bat"
title Hams.AI - downloading llama.cpp
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\tools\get_llama_cpp.ps1" -Dest "%LLAMA_DIR%" > "%ROOT%\tools\logs\llama_setup.log" 2>&1
echo done >> "%ROOT%\tools\logs\llama_setup.log"

@echo off
REM Starts llama.cpp's OpenAI-compatible server on http://127.0.0.1:8080
REM   start_llm.bat                 -> dev default (Qwen3.5-2B Q4_K_M, all layers on GPU)
REM   start_llm.bat <file.gguf> [ngl] [ctx] [parallel]
call "%~dp0env.bat"
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=Qwen3.5-2B-Q4_K_M.gguf"
set "NGL=%~2"
if "%NGL%"=="" set "NGL=99"
set "CTX=%~3"
if "%CTX%"=="" set "CTX=8192"
set "NP=%~4"
if "%NP%"=="" set "NP=2"
title llama-server %MODEL%
"%LLAMA_DIR%\llama-server.exe" -m "%MODELS_DIR%\gguf\%MODEL%" -ngl %NGL% -c %CTX% -np %NP% --host 127.0.0.1 --port 8080 --jinja -fa on -ctk q8_0 -ctv q8_0 --cache-reuse 256 --metrics --alias qwen > "%ROOT%\tools\logs\llama_server.log" 2>&1

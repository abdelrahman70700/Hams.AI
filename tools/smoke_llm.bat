@echo off
REM Smoke test: llama-bench throughput for the dev LLM on this GPU (prompt processing + generation speed).
call "%~dp0env.bat"
cd /d "%ROOT%"
set "LOG=%ROOT%\tools\logs\smoke_llm.log"
title Hams.AI - LLM smoke test
echo [%date% %time%] START > "%LOG%"
"%LLAMA_DIR%\llama-server.exe" --version >> "%LOG%" 2>&1
"%LLAMA_DIR%\llama-bench.exe" -m "%MODELS_DIR%\gguf\Qwen3.5-2B-Q4_K_M.gguf" -ngl 99 -fa 1 -p 512,1536 -n 128 -r 2 -o md >> "%LOG%" 2>&1
echo [%date% %time%] DONE >> "%LOG%"

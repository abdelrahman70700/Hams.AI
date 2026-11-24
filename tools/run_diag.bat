@echo off
REM Diagnostic for the reranker fine-tuning collapse (read-only: loads the base model, prints logits checks).
call "%~dp0env.bat"
cd /d "%ROOT%"
title Hams.AI - reranker diagnostic
"%PY%" tools\diag_reranker.py > "%ROOT%\tools\logs\diag_reranker.log" 2>&1

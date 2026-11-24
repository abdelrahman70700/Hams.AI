@echo off
REM Log in to Hugging Face for this project. Paste your token when asked (it is hidden while you type/paste).
REM The token is stored by huggingface_hub in this project's HF cache (.cache\hf), never in the code.
call "%~dp0env.bat"
cd /d "%ROOT%"
title Hugging Face login
echo Create a token with WRITE permission at https://huggingface.co/settings/tokens
echo Then paste it below (right-click to paste; nothing is shown while pasting) and press Enter.
echo.
"%PY%" -c "from huggingface_hub import login, whoami; login(add_to_git_credential=False); print('Logged in as', whoami()['name'])"
echo.
pause

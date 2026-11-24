@echo off
REM Tests candidate conversations against the running API (needs start_all.bat running).
call "%~dp0env.bat"
cd /d "%ROOT%"
title Hams.AI - testing conversations
"%PY%" scripts\probe_conversations.py > "%ROOT%\tools\logs\probe_conv.log" 2>&1
echo DONE >> "%ROOT%\tools\logs\probe_conv.log"

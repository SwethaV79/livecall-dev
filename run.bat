@echo off
REM Start backend server
start cmd /k "cd /d %~dp0backend && python app.py"
REM Start frontend server
start cmd /k "cd /d %~dp0.. && npm run dev"
REM Start prompt.py script
start cmd /k "cd /d %~dp0backend && python prompt.py"
pause

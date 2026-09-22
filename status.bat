@echo off
rem Shows collection progress. Double-click this file any time (the collector can keep running).
cd /d "%~dp0"
python collector.py status
pause

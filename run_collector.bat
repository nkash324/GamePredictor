@echo off
rem Starts the LoL data collector. Double-click this file.
rem Stop it with Ctrl+C (or just close the window).
cd /d "%~dp0"
title LoL data collector
:loop
python collector.py
echo.
echo The collector stopped. It will restart in 60 seconds.
echo Close this window (or press Ctrl+C) to keep it stopped.
timeout /t 60
goto loop

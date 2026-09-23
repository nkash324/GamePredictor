@echo off
rem Updates everything: new balance data -> training tables -> baselines -> model.
rem Double-click this file. Safe to run while the collector is running.
cd /d "%~dp0"
title LoL model update
python run_pipeline.py %*
if %errorlevel%==0 if exist report.html start "" report.html
echo.
pause

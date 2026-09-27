@echo off
rem Sends the latest model results (the results folder) to GitHub, so they can be pulled
rem on the other computer. Only the small result files - no database, no games table.
cd /d "%~dp0"
title Share results
git add results
git commit -m "Results %date% %time:~0,5%"
git push
echo.
pause

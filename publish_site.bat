@echo off
rem Builds the PatchCast website from the latest results and uploads it to GitHub Pages.
rem Run update_model.bat first so the results are fresh.
rem The website lives in the PatchCastSite folder next to this project folder.
cd /d "%~dp0"
title Publish PatchCast
python build_site.py %*
if errorlevel 1 (
  echo.
  echo Building the site failed - see the message above.
  pause
  exit /b 1
)
cd /d "%~dp0..\PatchCastSite"
if not exist ".git" (
  echo.
  echo The PatchCastSite folder isn't linked to GitHub yet.
  echo Open PATCHCAST_SITE_SETUP.md in the GamePredictor folder and follow "One-time setup", then run this again.
  echo You can still preview the site now by opening PatchCastSite\index.html.
  pause
  exit /b 0
)
git add -A
git commit -m "Update site %date% %time:~0,5%"
git push
echo.
echo Done. The site updates online within a minute or two.
pause

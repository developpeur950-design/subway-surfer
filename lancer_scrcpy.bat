@echo off
REM Affiche le telephone (Subway Surfers City) en haut a gauche de lecran du PC.
REM Double-cliquez sur ce fichier. Ne deplacez plus la fenetre apres la calibration.
"%~dp0scrcpy\scrcpy.exe" --max-size 1024 --max-fps 30 --stay-awake --window-x 0 --window-y 0 --window-title "SubwayCity" %*
if errorlevel 1 pause

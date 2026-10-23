@echo off
REM Affiche le telephone SANS CABLE (Wi-Fi). Le PC et le telephone doivent etre sur le meme reseau.
REM 1re fois : branchez le cable, lancez ce fichier, puis debranchez quand la fenetre apparait.
REM Ensuite : lancez-le directement (si ladresse du telephone na pas change).
REM Sans jamais de cable (Android 11+) : voir le README, section "Sans cable".
set SCR=%~dp0scrcpy
if exist "%~dp0calib\ip_telephone.txt" (
  set /p IP=<"%~dp0calib\ip_telephone.txt"
)
if defined IP (
  "%SCR%\adb.exe" connect %IP%
  "%SCR%\scrcpy.exe" --tcpip=%IP% --max-size 1024 --max-fps 30 --video-bit-rate 4M --window-x 0 --window-y 0 --window-title "SubwayCity" %*
) else (
  "%SCR%\scrcpy.exe" --tcpip --max-size 1024 --max-fps 30 --video-bit-rate 4M --window-x 0 --window-y 0 --window-title "SubwayCity" %*
)
if errorlevel 1 pause

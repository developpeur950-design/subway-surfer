@echo off
REM Affiche le telephone (Subway Surfers City) sur le PC en haut a gauche de l'ecran.
REM Placez ce fichier dans le dossier de scrcpy, ou ajoutez scrcpy au PATH.
REM Ne deplacez plus la fenetre apres la calibration.
scrcpy.exe --max-size 1024 --max-fps 30 --stay-awake --window-x 0 --window-y 0 --window-title "SubwayCity" %*
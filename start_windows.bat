@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERREUR] L'environnement virtuel .venv est absent.
  echo Lancez d'abord setup_windows.bat pour installer le projet.
  echo.
  pause
  exit /b 1
)

echo ========================================================
echo CarePilot - Demarrage du serveur
echo ========================================================
echo URL de l'application : http://127.0.0.1:5000
echo Pour arreter le serveur : faites Ctrl + C dans cette fenetre
echo ========================================================
echo.

:: Nettoyer un eventuel ancien processus sur le port 5000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5000" ^| findstr "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

:: Ouvre l'URL dans le navigateur par defaut apres un leger delai
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:5000"

call ".venv\Scripts\activate.bat"
python app.py
pause

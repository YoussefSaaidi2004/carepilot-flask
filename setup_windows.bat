@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
echo ========================================================
echo CarePilot - Installation locale Windows
echo ========================================================

set "PYTHON="

:: 1. Verifier les versions Python prises en charge (de la plus recente a la plus ancienne)
py -3.14 --version >nul 2>&1 && set "PYTHON=py -3.14"
if not defined PYTHON (
  py -3.13 --version >nul 2>&1 && set "PYTHON=py -3.13"
)
if not defined PYTHON (
  py -3.12 --version >nul 2>&1 && set "PYTHON=py -3.12"
)
if not defined PYTHON (
  py -3.11 --version >nul 2>&1 && set "PYTHON=py -3.11"
)

:: 2. Verifier dans LocalAppData (installations standard utilisateur)
if not defined PYTHON (
  if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
  ) else if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
  ) else if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
  ) else if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
  )
)

:: 3. Verifier la commande python du PATH
if not defined PYTHON (
  python -c "import sys; exit(0 if sys.version_info[0]==3 and sys.version_info[1] in (11,12,13,14) else 1)" >nul 2>&1
  if not errorlevel 1 set "PYTHON=python"
)

if not defined PYTHON (
  echo [ERREUR] Python 3.11, 3.12, 3.13 ou 3.14 (64-bit) est requis.
  echo Installez Python depuis https://www.python.org/downloads/
  goto :error
)

echo Python detecte : %PYTHON%
echo.

if not exist ".venv\Scripts\python.exe" (
  echo Creation de l'environnement virtuel .venv...
  %PYTHON% -m venv .venv
  if errorlevel 1 goto :error
)

call ".venv\Scripts\activate.bat"
echo Mise a jour de pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :error

echo Installation des dependances...
pip install -r requirements.txt
if errorlevel 1 goto :error

if not exist ".env" (
  for /f "delims=" %%K in ('python -c "import secrets; print(secrets.token_urlsafe(48))"') do set "SECRET_KEY=%%K"
  >".env" echo SECRET_KEY=!SECRET_KEY!
  >>".env" echo HOST=127.0.0.1
  >>".env" echo PORT=5000
  >>".env" echo FLASK_DEBUG=0
  >>".env" echo COOKIE_SECURE=0
  echo Configuration .env creee avec une cle secrete aleatoire.
)

echo.
echo Verification des modeles machine learning (smoke test)...
python smoke_test.py
if errorlevel 1 goto :error

echo.
echo Voulez-vous creer un compte administrateur personnalise maintenant ? (O/N)
set /p CREATE_ADM="Votre choix (defaut N) : "
if /i "%CREATE_ADM%"=="O" (
  flask --app app create-admin
) else (
  echo Compte administrateur ignore. L'application utilisera la session locale par defaut.
)

echo.
echo ========================================================
echo Installation terminee avec succes !
echo Vous pouvez maintenant lancer l'application avec :
echo   start_windows.bat
echo ========================================================
pause
exit /b 0

:error
echo.
echo [ERREUR] Une etape a echoue. Lisez le message precedent et verifiez Python / Internet / dependances.
pause
exit /b 1

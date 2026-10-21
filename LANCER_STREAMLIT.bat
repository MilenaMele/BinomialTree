@echo off
setlocal
rem Lanceur autonome : ne modifie ni PATH ni la politique PowerShell.
rem Toujours travailler dans le dossier contenant ce fichier.
cd /d "%~dp0"
if errorlevel 1 goto dossier_invalide

rem Utiliser en priorite le Python 3.13 installe dans le profil Windows.
set "PROJECT_PYTHON_COMMAND="%LOCALAPPDATA%\Programs\Python\Python313\python.exe""
if not exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" goto essayer_python
%PROJECT_PYTHON_COMMAND% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 goto python_trouve

:essayer_python
rem Tester le programme : un alias Microsoft Store ne suffit pas.
set "PROJECT_PYTHON_COMMAND=python"
%PROJECT_PYTHON_COMMAND% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 goto python_trouve

:essayer_py
set "PROJECT_PYTHON_COMMAND=py -3"
%PROJECT_PYTHON_COMMAND% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 goto python_trouve

echo Python 3.10 ou plus recent est introuvable.
echo Installer Python depuis https://www.python.org/downloads/windows/
echo puis relancer ce fichier. Ne pas utiliser le raccourci du Microsoft Store.
set "PROJECT_EXIT_CODE=1"
goto echec

:python_trouve
echo Python utilise :
%PROJECT_PYTHON_COMMAND% -c "import sys; print(sys.executable)"
echo Installation / verification des dependances...
%PROJECT_PYTHON_COMMAND% -m pip install -r requirements.txt
set "PROJECT_EXIT_CODE=%ERRORLEVEL%"
if not "%PROJECT_EXIT_CODE%"=="0" goto installation_echouee

echo Ouverture de Streamlit. Garder cette fenetre ouverte ; Ctrl+C pour arreter.
%PROJECT_PYTHON_COMMAND% -m streamlit run app.py
set "PROJECT_EXIT_CODE=%ERRORLEVEL%"
if not "%PROJECT_EXIT_CODE%"=="0" goto application_echouee
exit /b 0

:installation_echouee
rem Conserver a la fois le message de pip et son code de sortie original.
echo L'installation a echoue avec le code %PROJECT_EXIT_CODE%.
echo Corriger l'erreur de pip affichee ci-dessus, puis relancer ce fichier.
goto echec

:application_echouee
echo Streamlit s'est termine avec le code %PROJECT_EXIT_CODE%.
echo Voir le message d'erreur ci-dessus.
goto echec

:dossier_invalide
echo Impossible d'ouvrir le dossier du projet.
set "PROJECT_EXIT_CODE=1"

:echec
rem Permettre la lecture de l'erreur meme apres un double-clic.
pause
exit /b %PROJECT_EXIT_CODE%

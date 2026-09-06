@echo off
REM Start script for Motion Tracker (Windows).
REM Creates/uses a virtual environment, installs dependencies, and launches the app.

setlocal
cd /d "%~dp0"

set "VENV_DIR=.venv"

REM Create the virtual environment if it does not exist.
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment in %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Error: failed to create virtual environment. Install Python 3.10+.
        exit /b 1
    )
)

REM Activate the virtual environment.
call "%VENV_DIR%\Scripts\activate.bat"

REM Install / update dependencies.
echo Installing dependencies ...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

REM Launch the application.
echo Starting Motion Tracker ...
python -m motion_tracker %*

endlocal

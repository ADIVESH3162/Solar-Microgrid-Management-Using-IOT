@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo    AI-Based Solar Microgrid Management System - Quickstart Launcher
echo =====================================================================
echo.

:: 1. Check Python installation
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    where py >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Python is not installed or not in PATH!
        echo Please install Python 3.10+ from https://python.org and add it to PATH.
        pause
        exit /b 1
    ) else (
        set PYTHON_CMD=py
    )
) else (
    set PYTHON_CMD=python
)

echo [OK] Using Python command: %PYTHON_CMD%
%PYTHON_CMD% --version

:: 2. Check / Create Virtual Environment
if not exist ".venv" (
    echo [INFO] Creating Python virtual environment in .venv...
    %PYTHON_CMD% -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

:: 3. Activate Virtual Environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    set RUN_PY=python
) else (
    set RUN_PY=%PYTHON_CMD%
)

:: 4. Install / Verify Dependencies
echo [INFO] Verifying required Python packages...
%RUN_PY% -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Package installation failed. Check internet connection.
    pause
    exit /b 1
)

:: 5. Ensure Data and Models Exist
if not exist "data\solar_dataset_sample.csv" (
    echo [INFO] Generating baseline sample and historical datasets...
    %RUN_PY% scripts\generate_sample_data.py
)

if not exist "models\solar_power_rf_model.joblib" (
    echo [INFO] Training initial baseline machine learning model...
    %RUN_PY% src\train_model.py
)

:: 6. Launch Streamlit Dashboard
echo.
echo =====================================================================
echo    Starting Streamlit Dashboard...
echo    Access the UI at: http://localhost:8501
echo =====================================================================
echo.
%RUN_PY% -m streamlit run app.py

pause


@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo    Solar Microgrid System - Machine Learning Model Trainer
echo =====================================================================
echo.

where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    set RUN_PY=py
) else (
    set RUN_PY=python
)

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    set RUN_PY=python
)

echo [INFO] Training Random Forest Regressor on solar dataset...
%RUN_PY% src\train_model.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Model training completed successfully!
    echo Serialized model saved to: models\solar_power_rf_model.joblib
    echo Metrics recorded in: models\model_metrics.json
) else (
    echo.
    echo [ERROR] Model training encountered an error.
)

pause


@echo off
REM =====================================================================
REM Quick Setup Script for Age Prediction Service
REM Installs dependencies and verifies configuration
REM =====================================================================

echo.
echo ======================================================================
echo AGE PREDICTION SERVICE - SETUP SCRIPT
echo ======================================================================
echo.

REM Step 1: Check Python
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8+
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
) else (
    for /f "tokens=*" %%A in ('python --version') do echo         %%A
)
echo.

REM Step 2: Check Node.js
echo [2/6] Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js not found. Please install Node.js 14+
    echo Download from: https://nodejs.org/
    pause
    exit /b 1
) else (
    for /f "tokens=*" %%A in ('node --version') do echo         %%A
)
echo.

REM Step 3: Install Python dependencies
echo [3/6] Installing Python dependencies...
cd backend
python -m pip install tensorflow flask requests --quiet
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
) else (
    echo         ✓ tensorflow installed
    echo         ✓ flask installed
    echo         ✓ requests installed
)
echo.

REM Step 4: Install Node.js dependencies
echo [4/6] Installing Node.js dependencies...
call npm install --quiet
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Node.js dependencies
    pause
    exit /b 1
) else (
    echo         ✓ npm packages installed
)
echo.

REM Step 5: Verify model file
echo [5/6] Verifying trained model...
if exist "..\ai\models\age_resnet50_best.keras" (
    for /f %%A in ('..\ai\models\age_resnet50_best.keras') do set "size=%%~zA"
    echo         ✓ Model found: age_resnet50_best.keras
    echo         ✓ File size: %size% bytes
) else (
    echo WARNING: Model file not found at: ..\ai\models\age_resnet50_best.keras
    echo         This is required for the service to run
    echo         Please ensure the model has been trained
)
echo.

REM Step 6: Verify test images
echo [6/6] Verifying test dataset...
if exist "..\ai\dataset\age_prediction\test\001" (
    echo         ✓ Test dataset found
    echo.
) else (
    echo WARNING: Test dataset not found
    echo         This is needed for testing
)
echo.

REM Success message
echo ======================================================================
echo ✓ SETUP COMPLETE
echo ======================================================================
echo.
echo Next steps:
echo.
echo Terminal 1: Start Python age prediction service
echo   cd backend
echo   python age_prediction_service.py
echo.
echo Terminal 2: Start Node.js backend (in same directory)
echo   node server.js
echo.
echo Terminal 3: (Optional) Test the API
echo   python test_api.py
echo.
echo Terminal 4: (Optional) Start frontend
echo   cd frontend
echo   npm run dev
echo.
echo ======================================================================
echo.

pause

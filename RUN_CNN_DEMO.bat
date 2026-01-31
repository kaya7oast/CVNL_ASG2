@echo off
echo ================================================================
echo CHANGI AEROVISION - CNN DEMO - AUTOMATIC SETUP
echo ================================================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo [1/3] Creating virtual environment...
    python -m venv .venv
    echo     Virtual environment created successfully!
    echo.
) else (
    echo [1/3] Virtual environment already exists
    echo.
)

REM Activate virtual environment
echo [2/3] Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install dependencies
echo [3/3] Installing dependencies...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo     Dependencies installed successfully!
echo.

echo ================================================================
echo SETUP COMPLETE! Starting CNN Demo...
echo ================================================================
echo.
echo The web interface will open at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

REM Run the demo
python CNN_demo.py

pause

@echo off
echo 🚀 Starting Nifty 50 Stock Tracker Setup...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ✗ Python not found! Please install Python first.
    pause
    exit /b 1
)
echo ✓ Python found

REM Create project directory
echo 📁 Creating project structure...
mkdir nifty50-tracker\app>nul
mkdir nifty50-tracker\frontend>nul  
mkdir nifty50-tracker\data>nul
mkdir nifty50-tracker\ml_models>nul
mkdir nifty50-tracker\tests>nul
cd nifty50-tracker

REM Create virtual environment
echo 🔧 Creating virtual environment...
python -m venv venv

echo ✨ Virtual environment created!
echo Please run the following commands:
echo.
echo   venv\Scripts\activate
echo   pip install -r requirements.txt
echo   python setup_verification_test.py
echo.
echo 🎉 Setup foundation ready! Follow the detailed guide for complete setup.
pause

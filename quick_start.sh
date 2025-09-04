#!/bin/bash
# Quick Start Script for Nifty 50 Stock Tracker

echo "🚀 Starting Nifty 50 Stock Tracker Setup..."

# Check if Python is installed
if command -v python3 &> /dev/null; then
    echo "✓ Python3 found"
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    echo "✓ Python found"  
    PYTHON_CMD=python
else
    echo "✗ Python not found! Please install Python first."
    exit 1
fi

# Create project directory
echo "📁 Creating project structure..."
mkdir -p nifty50-tracker/{app,frontend,data,ml_models,tests}
cd nifty50-tracker

# Create virtual environment
echo "🔧 Creating virtual environment..."
$PYTHON_CMD -m venv venv

# Activate virtual environment (instructions)
echo "✨ Virtual environment created!"
echo "Please run the following commands:"
echo ""

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "  venv\Scripts\activate"
else
    echo "  source venv/bin/activate"
fi

echo "  pip install -r requirements.txt"
echo "  python setup_verification_test.py"
echo ""
echo "🎉 Setup foundation ready! Follow the detailed guide for complete setup."
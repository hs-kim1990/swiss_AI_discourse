@echo off
echo Starting Swiss AI Discourse Dashboard...
python run.py
if errorlevel 1 (
    echo.
    echo Python not found. Please install Python 3 from https://python.org
    pause
)

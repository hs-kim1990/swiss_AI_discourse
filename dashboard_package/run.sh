#!/bin/bash
echo "Starting Swiss AI Discourse Dashboard..."
if command -v python3 &>/dev/null; then
    python3 run.py
elif command -v python &>/dev/null; then
    python run.py
else
    echo "Python not found. Please install Python 3."
    exit 1
fi

#!/bin/bash
# Wrapper script to run process_pgn.py with the correct Python
# This ensures we use the venv Python, not conda's Python

cd "$(dirname "$0")"

# Use the venv Python directly (full path to avoid conda interference)
VENV_PYTHON="/Users/linhong/Documents/Github/chesshacks/venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: venv Python not found at $VENV_PYTHON"
    echo "Please check your venv path."
    exit 1
fi

echo "Using Python: $VENV_PYTHON"
echo "Python version: $($VENV_PYTHON --version)"
echo ""

# Run the script
exec "$VENV_PYTHON" process_pgn.py "$@"

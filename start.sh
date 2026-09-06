#!/usr/bin/env bash
#
# Start script for Motion Tracker (macOS / Linux).
# Creates/uses a virtual environment, installs dependencies, and launches the app.
#
set -euo pipefail

# Resolve the directory this script lives in, so it works from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"

# Pick a Python interpreter. On macOS prefer Homebrew Python 3.11 (system Tk is too old).
if [ -x "$VENV_DIR/bin/python" ]; then
    PYTHON="$VENV_DIR/bin/python"
elif [ -x "/opt/homebrew/bin/python3.11" ]; then
    PYTHON="/opt/homebrew/bin/python3.11"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    echo "Error: no suitable Python interpreter found. Install Python 3.10+." >&2
    exit 1
fi

# Create the virtual environment if it does not exist.
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# Activate the virtual environment.
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Install / update dependencies.
echo "Installing dependencies ..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

# Launch the application.
echo "Starting Motion Tracker ..."
exec python -m motion_tracker "$@"

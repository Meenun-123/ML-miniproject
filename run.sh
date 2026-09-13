#!/usr/bin/env bash
# Chess Mate Vision - Single Command Launcher
# Starts both FastAPI and Vite with a single command

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

exec "$PYTHON" run.py "$@"

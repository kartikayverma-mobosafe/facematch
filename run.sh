#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    if command -v apt >/dev/null 2>&1; then
        sudo apt update
        sudo apt install -y python3 python3-venv python3-pip build-essential libgl1 libglib2.0-0
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y python3 python3-pip gcc gcc-c++ mesa-libGL
    fi

    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

.venv/bin/python compare_faces.py "$@"

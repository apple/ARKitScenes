#!/bin/bash

CURRENT_DIR="${PWD}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=== Creating virtual environment ==="
python3 -m venv "${SCRIPT_DIR}/venv"

echo "=== Activating virtual environment ==="
source "${SCRIPT_DIR}/venv/bin/activate"

echo "=== Setting up python path ==="
VENV_PYTHON_SITE=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
echo "${CURRENT_DIR}" > "${VENV_PYTHON_SITE}/packages.pth"

echo "=== Installing requirements ==="
pip install -r "${SCRIPT_DIR}/requirements.txt"

echo "Done!"
cd "${CURRENT_DIR}"
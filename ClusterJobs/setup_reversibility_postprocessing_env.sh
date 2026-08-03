#!/bin/bash
# Create a Linux Python environment for the postprocessing jobs.

set -euo pipefail

VENV_PATH="${1:-$HOME/simulation/reversibility-postprocess-venv}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_PATH}"
fi

"${VENV_PATH}/bin/python" -m pip install --upgrade pip
"${VENV_PATH}/bin/python" -m pip install numpy pandas meshio sympy
"${VENV_PATH}/bin/python" -c \
    "import meshio, numpy, pandas, sympy; print('postprocessing environment is ready')"
echo "Use ${VENV_PATH}/bin/python for the Slurm jobs."

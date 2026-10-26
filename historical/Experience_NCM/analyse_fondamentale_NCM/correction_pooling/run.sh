#!/bin/zsh
set -eu
cd "${0:A:h}"
export PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" MPLCONFIGDIR="$PWD/.cache/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PYTHON=../.venv/bin/python
$PYTHON -u scripts/experiments.py > logs/experiments.log 2>&1
$PYTHON -u scripts/compensation.py > logs/compensation.log 2>&1
$PYTHON -u scripts/publish.py > logs/publish.log 2>&1
$PYTHON -u scripts/verify.py --finalize > logs/verify.log 2>&1

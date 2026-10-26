#!/bin/zsh
set -eu
cd "${0:A:h}"
export PYTHONPYCACHEPREFIX="$PWD/.cache/pycache"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PYTHON=../.venv/bin/python
$PYTHON -u scripts/prepare.py > logs/prepare.log 2>&1
$PYTHON -u scripts/prepare_corpora.py > logs/prepare_corpora.log 2>&1
$PYTHON -u scripts/sudoku_levels.py > logs/sudoku_levels.log 2>&1
$PYTHON -u scripts/text_audit.py > logs/text_audit.log 2>&1
$PYTHON -u scripts/publish.py > logs/publish.log 2>&1
$PYTHON -u scripts/verify.py --finalize > logs/verify.log 2>&1

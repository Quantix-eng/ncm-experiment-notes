#!/bin/sh
set -eu
cd "$(dirname "$0")"
export PYTHONPYCACHEPREFIX="$PWD/.cache/pycache"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$TMPDIR" logs
../.venv/bin/python scripts/assets.py > logs/assets.log 2>&1
../.venv/bin/python scripts/crossed.py > logs/crossed.log 2>&1
../.venv/bin/python scripts/text_generation.py > logs/text_generation.log 2>&1
../.venv/bin/python scripts/publish.py > logs/publish.log 2>&1
../.venv/bin/python scripts/verify.py > logs/verify.log 2>&1

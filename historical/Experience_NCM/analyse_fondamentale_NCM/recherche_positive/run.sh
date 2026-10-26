#!/bin/sh
set -eu
cd "$(dirname "$0")"
export PYTHONPYCACHEPREFIX="$PWD/.cache/pycache"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export TMPDIR="$PWD/.cache/tmp"
export TEXMFVAR="$PWD/.cache/tex-var"
export TEXMFCONFIG="$PWD/.cache/tex-config"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR" logs
PY=../.venv/bin/python
# Rebuild and audit the delivered evidence without replacing historical measurements.
# Full computational rerun: "$PY" scripts/reproduce.py --destination ../reproduction_alphabet
"$PY" scripts/publish.py > logs/publication.log 2>&1
"$PY" scripts/pdf_qa.py > logs/pdf_qa.log 2>&1
"$PY" scripts/verify.py > logs/verification.log 2>&1

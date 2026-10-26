#!/bin/zsh
set -eu
cd "${0:A:h}"
export PYTHONPYCACHEPREFIX="$PWD/.cache/pycache"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p results logs figures .cache
.venv/bin/python -u scripts/prepare.py > logs/prepare.log 2>&1
.venv/bin/python -u scripts/audit_overlap.py > logs/audit_overlap.log 2>&1
.venv/bin/python -u scripts/smoke_checks.py > logs/smoke_checks.log 2>&1
.venv/bin/python -u scripts/upstream_parity.py > logs/upstream_parity.log 2>&1
.venv/bin/python -u scripts/extract.py hard > logs/extract_hard.log 2>&1
.venv/bin/python -u scripts/extract.py easy > logs/extract_easy.log 2>&1
.venv/bin/python -u scripts/extract.py medium > logs/extract_medium.log 2>&1
.venv/bin/python -u scripts/tokenizer_errors.py > logs/tokenizer_errors.log 2>&1
.venv/bin/python -u scripts/analyze.py > logs/analyze.log 2>&1
.venv/bin/python -u scripts/experiments.py > logs/experiments.log 2>&1
.venv/bin/python -u scripts/constant_level.py > logs/constant_level.log 2>&1
.venv/bin/python -u scripts/report.py > logs/report.log 2>&1
.venv/bin/python -u scripts/verify.py > logs/verify.log 2>&1

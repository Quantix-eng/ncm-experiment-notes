#!/bin/zsh
set -eu
cd "${0:A:h}"
export PYTHONPYCACHEPREFIX="$PWD/.cache/pycache"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PYTHON=../.venv/bin/python
$PYTHON -u scripts/runtime.py > logs/runtime_parity.log 2>&1
$PYTHON -u scripts/diagnose_level5.py > logs/diagnose_level5.log 2>&1
$PYTHON -u scripts/pooling_algebra.py > logs/pooling_algebra.log 2>&1
$PYTHON -u scripts/archive_centers.py > logs/archive_centers.log 2>&1
$PYTHON -u scripts/dropout_control.py > logs/dropout_control.log 2>&1
$PYTHON -u scripts/level5_interventions.py > logs/level5_interventions.log 2>&1
$PYTHON -u scripts/extract_revision.py > logs/extract_revision.log 2>&1
$PYTHON -u scripts/probes.py > logs/probes.log 2>&1
$PYTHON -u scripts/conditional.py > logs/conditional.log 2>&1
$PYTHON -u scripts/steer_revision.py > logs/steer_revision.log 2>&1
$PYTHON -u scripts/steer_revision.py --conditional > logs/steer_revision_conditional.log 2>&1
$PYTHON -u scripts/report.py > logs/report.log 2>&1
$PYTHON -u scripts/verify.py > logs/verify.log 2>&1

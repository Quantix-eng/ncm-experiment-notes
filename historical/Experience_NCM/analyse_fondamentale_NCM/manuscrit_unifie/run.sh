#!/bin/sh
set -eu
cd "$(dirname "$0")"
export PYTHONPYCACHEPREFIX="$PWD/.cache/pycache"
python3 tools/assemble.py
python3 tools/claims.py
python3 tools/build.py
python3 tools/check.py
python3 tools/package.py

# Shared code and setup

The experiment folders own their scientific scripts. This directory contains the parts reused across studies:

- `backends/`: the pinned NCM implementations, including the separate native and audit adapters. They are retained separately because the historical studies used different copies.
- `sudoku/`: Sudoku invariants and solver helpers.
- `ioi/`: the GPT-2 intervention engine used by experiments 03 and 08.
- `probes/`: feature and fitting utilities used by experiments 12–14.
- `toy/`: the toy-model engine used by experiments 18 and 20.
- `runner.py`: execution support for each experiment’s local `run.py`.
- `layout.json`: a mapping from current files to the relative paths expected by existing scientific scripts.

## Dependencies

Recorded-data checks use Python 3.10+ and its standard library. For model execution, use a separate Python 3.11 or 3.12 environment and choose a requirements file:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements/ncm.txt
.venv/bin/python -m common.download_sudoku
```

Run these commands from the repository root. The downloader verifies pinned official Sudoku files and writes to `assets/`. Use its `--difficulty` option for the other checkpoints. `requirements/language.txt` adds the language-model stack. InterpBench additionally needs its pinned third-party sources; the experiment’s input metadata records those revisions. Fresh dependency installations have not all been tested.

## Why an execution mapping remains

Moving the files alone would break the original scripts’ relative paths and local imports. The runner copies computational files and inputs into an isolated `runs/<name>/workspace/` and executes them there. That compatibility layout is an internal execution detail; editable sources remain in each experiment’s `src/` or here in `common/`.

New runs do not inherit the target study’s recorded outcome files or its old protocol seals. English execution notes replace former report paths used for document fingerprints. Existing numerical gates and integrity checks remain active. A relocated run is not the unchanged historical freeze.

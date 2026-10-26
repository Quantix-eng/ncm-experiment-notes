# NCM Mac — Sudoku

A CPU FP32 port of [NCM Sudoku](https://github.com/HazyResearch/ncm) for Apple Silicon. It runs the official checkpoints without changing the weights.

On a MacBook Pro M1 (8 GB), inference took **39.62 seconds versus 94.88 seconds** for the original implementation: **2.39× faster** over 384 boards, excluding loading and warmup. A separate check of all **6,000 official test boards** found no bitwise differences in codes, final reconstructions, logits or answers. Results are in [validation.json](validation.json).

The main change is caching the puzzle context across generation levels and reusing the reconstruction accumulator. The port also fixes aliasing in intermediate-state snapshots. Inference keeps the original FP32 arithmetic and decoding rules, without quantization or answer repair.

## Run

Tested with macOS 14.0, Python 3.13.1 ARM64 and PyTorch 2.8.0.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install .
.venv/bin/python scripts/download_sudoku.py
.venv/bin/ncm-sudoku --difficulty easy --n 32 --output results.json
```

The downloader fetches approximately 489 MB of pinned, integrity-checked assets. Choose `easy`, `medium` or `hard`; `--n 0` runs all 2,000 boards of that difficulty. Defaults are batch 8 and four CPU threads. For long runs, `--workers 4` enables multiprocessing; the exhaustive validation below covers the default single-process mode.

## Verify

```sh
git clone https://github.com/HazyResearch/ncm.git upstream-ncm
git -C upstream-ncm checkout c70274273f489abc73832b2b98584a8483b6b203
.venv/bin/python scripts/validate_sudoku.py --n 0 --batch-size 8 --threads 4 --output parity.json
```

The script compares against the unmodified upstream code using `torch.equal` and fails on any mismatch. It records scores, sample indices, output hashes and the environment. For a shorter check, use `--n 128` to test 384 evenly spaced boards. A completed report contains all three difficulties.

Both implementations solve 1,988/2,000 easy (99.4%), 1,928/2,000 medium (96.4%) and 1,726/2,000 hard (86.3%) boards. Exact agreement preserves the original model’s errors as well as its correct answers.

Bitwise equality applies to the tested CPU environment and settings. Changes to hardware, library versions or batch size require another check. Metal GPU experiments exceeded the numerical error threshold and are not included.

## Credit

This independent port is based on upstream commit `c70274273f489abc73832b2b98584a8483b6b203`, under the [MIT license](LICENSE). Weights and datasets are downloaded from their original hosts, not redistributed; the code license does not automatically cover those assets.

For the model, cite [Kumbong et al., *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*](https://arxiv.org/abs/2610.14162v1), DOI: [10.48550/arXiv.2610.14162](https://doi.org/10.48550/arXiv.2610.14162). The full author list is in [CITATION.cff](CITATION.cff). For this port, include NCM Mac 0.2.0 and the source snapshot from `PUBLICATION_MANIFEST.json`.

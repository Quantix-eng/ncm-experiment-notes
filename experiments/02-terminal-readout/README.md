# Reading and editing terminal codes

[Overview](../../README.md#all-20-experiments-at-a-glance) · E10 · NCM · Sudoku

Build digit/code lookup tables on discovery data. Request single-cell writes on 128 source puzzles, then test lookup plus external certification on 1,024 fresh puzzles.

**Result.** Target-associated substitutions succeed in 784/1,024 requests on 128 puzzles. A lookup plus a Sudoku checker certifies 552/1,024 fresh puzzles before falling back to the native decoder.

**Interpretation.** Partial terminal readout; no full-pipeline speedup established.

**Limit.** Eight requests share each source puzzle. Gaussian controls do not match valid codebook geometry. The certificate depends on clue checks, uniqueness and exact fallback. No whole-model speedup is established.

## Run from this folder

```sh
python3 run.py check
python3 run.py stages
python3 run.py run sample --plan
```

`check` executes [the local recalculation](src/check_recorded.py), without a model. Remove `--plan` to execute an inference stage after [setup](../../common/README.md). New outputs go to `runs/<name>/`.

**Inputs:** Hard Sudoku assets + the included table for `sample`; extraction arrays and other checkpoints for full historical stages. **Tested here:** recorded-data check; `sample`. Other full workflows remain unvalidated after relocation.

[Code](src/) · [Inputs](inputs/) · [Saved results](recorded/) · [Stage details](experiment.json)

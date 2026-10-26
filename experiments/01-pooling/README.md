# A pooling symmetry in solved Sudoku

[Overview](../../README.md#all-20-experiments-at-a-glance) · E06 · NCM · Sudoku

Compare token histograms in the nine adaptive pooling windows, inspect native level-5 usage, then perturb the encoding pool while keeping the generator frozen.

**Result.** Weighted pooling activates 15 of the 16 encoding identifiers while preserving the recorded reconstruction counts. Agreement with the frozen generator target falls from 100% to 1.65%.

**Interpretation.** Encoding diagnosis; no generation improvement.

**Limit.** The algebra applies before GroupNorm and earlier residual subtraction. It does not prove that native quantization must collapse. The modified encoding path is not used by native generation, and no solving improvement follows.

## Run from this folder

```sh
python3 run.py check
python3 run.py stages
python3 run.py run checkpoint --plan
```

`check` executes [the local recalculation](src/check_recorded.py), without a model. Remove `--plan` to execute an inference stage after [setup](../../common/README.md). New outputs go to `runs/<name>/`.

**Inputs:** Sudoku assets; extraction arrays for weighted pooling. **Tested here:** recorded-data check; `checkpoint`, `algebra`. Other full workflows remain unvalidated after relocation.

[Code](src/) · [Inputs](inputs/) · [Saved results](recorded/) · [Stage details](experiment.json)

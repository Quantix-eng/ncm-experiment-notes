# A correct intermediate digit can change again

[Overview](../../README.md#all-20-experiments-at-a-glance) · E03 · NCM · Sudoku

Decode native Sudoku trajectories at each level. Count corrected and degraded cells, later reversals, and the quality of earlier full-grid readouts.

**Result.** On 600 hard puzzles, 25.04% of the cells corrected between levels 8 and 9 become wrong again later. Earlier complete readouts solve no full grid in the reported cutoff test.

**Interpretation.** A correct intermediate digit is not a stable commitment.

**Limit.** Decoded grids are intermediate observations, not the generator input. An oracle codec given the correct solution also shows reversals. This rules out strict decoded-value locking, not every possible latent commitment.

## Run from this folder

```sh
python3 run.py check
python3 run.py stages
python3 run.py run describe --plan
```

`check` executes [the local recalculation](src/check_recorded.py), without a model. Remove `--plan` to execute an inference stage after [setup](../../common/README.md). New outputs go to `runs/<name>/`.

**Inputs:** Per-level Sudoku extraction arrays. **Tested here:** recorded-data check. Other full workflows remain unvalidated after relocation.

[Code](src/) · [Inputs](inputs/) · [Saved results](recorded/) · [Stage details](experiment.json)

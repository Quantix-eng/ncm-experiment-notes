# Two terminal edits can interfere

[Overview](../../README.md#all-20-experiments-at-a-glance) · E11 · NCM · Sudoku

Apply two requested terminal-code edits jointly. Count both-target success, clean edits, collateral changes and retention by cell separation.

**Result.** Across 10,240 double edits, both requested targets are written in 55.12% of trials and the complete contract holds in 54.45%. Retention is lower for adjacent cells.

**Interpretation.** Edits interfere; single-edit success does not imply composition.

**Limit.** The terminal residual includes a local convolution, so spatial interference is not surprising by itself. Edits share sources and must not be counted as independent puzzles.

## Run from this folder

```sh
python3 run.py check
python3 run.py stages
python3 run.py run freeze --plan
```

`check` executes [the local recalculation](src/check_recorded.py), without a model. Remove `--plan` to execute an inference stage after [setup](../../common/README.md). New outputs go to `runs/<name>/`.

**Inputs:** Sudoku assets and terminal discovery tables. **Tested here:** recorded-data check. Other full workflows remain unvalidated after relocation.

[Code](src/) · [Inputs](inputs/) · [Saved results](recorded/) · [Stage details](experiment.json)

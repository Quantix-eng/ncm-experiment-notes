# The initial NCM audit

[Overview](../../README.md#all-20-experiments-at-a-glance) · E05 · NCM · Sudoku

Audit native reconstruction and code usage. Fit a naked-single probe with shuffled-label control, then test whether the feature-specific causal intervention is selective.

**Result.** The audit found the constant intermediate level and a naked-single probe with AUROC 0.877 versus 0.496 with shuffled labels. The causal ablation was not specific.

**Interpretation.** Decodability does not establish causal use of the feature.

**Limit.** Naked singles can be computed directly from the puzzle. Decodability alone does not show that the model uses the feature or implements a human Sudoku rule.

## Run from this folder

```sh
python3 run.py check
python3 run.py stages
python3 run.py run prepare --plan
```

`check` executes [the local recalculation](src/check_recorded.py), without a model. Remove `--plan` to execute an inference stage after [setup](../../common/README.md). New outputs go to `runs/<name>/`.

**Inputs:** Sudoku assets and original split inputs. **Tested here:** recorded-data check. Other full workflows remain unvalidated after relocation.

[Code](src/) · [Inputs](inputs/) · [Saved results](recorded/) · [Stage details](experiment.json)

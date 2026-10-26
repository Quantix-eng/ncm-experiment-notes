# Internal states added little to failure prediction

[Overview](../../README.md#all-20-experiments-at-a-glance) · E01 · NCM · Sudoku

Fit failure predictors on training-only features. Compare observable-only and hidden-state alternatives on the reserved 500-puzzle test with grouped uncertainty.

**Result.** The reserved 500-puzzle study reports an AUROC increment of about 0.0043 for one fitted contrast. A stronger observable-only summary outperforms that state-augmented comparison.

**Interpretation.** No convincing practical benefit from internal-state features.

**Limit.** A small conditional improvement is not an operational advantage. These are externally fitted predictors, not native confidence or metacognition. Earlier probe objectives are separate experiments.

## Run from this folder

```sh
python3 run.py check
python3 run.py stages
python3 run.py run pilot --plan
```

`check` executes [the local recalculation](src/check_recorded.py), without a model. Remove `--plan` to execute an inference stage after [setup](../../common/README.md). New outputs go to `runs/<name>/`.

**Inputs:** Sudoku assets and training/development feature arrays. **Tested here:** recorded-data check. Other full workflows remain unvalidated after relocation.

[Code](src/) · [Inputs](inputs/) · [Saved results](recorded/) · [Stage details](experiment.json)

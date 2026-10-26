# Predicting which cell errors persist

[Overview](../../README.md#all-20-experiments-at-a-glance) · E02 · NCM · Sudoku

Predict which currently wrong cells remain wrong. Use visible-feature controls, grouped evaluation and the restricted mixed-puzzle cohort; inspect persistence prevalence.

**Result.** On the restricted mixed-puzzle cohort, the reported increment is -0.00060 with an interval spanning zero. Most current errors are corrected later, creating a strong prevalence trap.

**Interpretation.** Local latent features do not help this fitted persistence predictor.

**Limit.** The cohort is selected and the target is a future event. The result concerns this fitted procedure, not the absence of information everywhere in the network.

## Run from this folder

```sh
python3 run.py check
python3 run.py stages
python3 run.py run development --plan
```

`check` executes [the local recalculation](src/check_recorded.py), without a model. Remove `--plan` to execute an inference stage after [setup](../../common/README.md). New outputs go to `runs/<name>/`.

**Inputs:** Sudoku assets, training features and development fits. **Tested here:** recorded-data check. Other full workflows remain unvalidated after relocation.

[Code](src/) · [Inputs](inputs/) · [Saved results](recorded/) · [Stage details](experiment.json)

# AR and parallel toy models on one task

[Overview](../../README.md#all-20-experiments-at-a-glance) · E15 · Toy AR / parallel · Symbol rotation

Train three paired seeds of AR and parallel toy models on symbol rotation. Calibrate a common damage dose before comparing recovery and exchange behavior.

**Result.** All six models solve the native examples, but no tested corruption dose meets the shared damage gate. Recovery contrasts at the fallback dose are negative. Cutoffs and seeds change exchange behavior.

**Interpretation.** No valid general comparison of AR and parallel repair.

**Limit.** The calibration objective fails. The parallel model is not NCM, and the training objectives differ. These runs cannot identify a general effect of autoregressive versus parallel generation.

## Run from this folder

```sh
python3 run.py check
python3 run.py stages
python3 run.py run prepare --plan
```

`check` executes [the local recalculation](src/check_recorded.py), without a model. Remove `--plan` to execute an inference stage after [setup](../../common/README.md). New outputs go to `runs/<name>/`.

**Inputs:** PyTorch/NumPy; fresh toy training and calibration. **Tested here:** recorded-data check. Other full workflows remain unvalidated after relocation.

[Code](src/) · [Inputs](inputs/) · [Saved results](recorded/) · [Stage details](experiment.json)

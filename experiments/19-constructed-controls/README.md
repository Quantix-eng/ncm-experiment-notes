# Checking the protocol on constructed programs

[Overview](../../README.md#all-20-experiments-at-a-glance) · E12 · Constructed programs

Generate known-behavior programs, apply observation/continuation/exchange access rules, and compare diagnostic sets with evaluator labels on 344 constructed instances.

**Result.** The complete diagnostic returns the intended answer in 344/344 constructed instances. The program library and diagnostic were designed together.

**Interpretation.** Implementation consistency, not external mechanism validation.

**Limit.** This is a consistency check, not external validation or a benchmark of mechanism discovery on learned networks. Observation-only failure on these programs is not a universal upper bound.

## Run from this folder

```sh
python3 run.py check
python3 run.py stages
python3 run.py run fixed-families --plan
```

`check` executes [the local recalculation](src/check_recorded.py), without a model. Remove `--plan` to execute an inference stage after [setup](../../common/README.md). New outputs go to `runs/<name>/`.

**Inputs:** NumPy/PyTorch for the fixed-family analytical rerun; no trained checkpoint required. **Tested here:** recorded-data check; `fixed-families`. Other full workflows remain unvalidated after relocation.

[Code](src/) · [Inputs](inputs/) · [Saved results](recorded/) · [Stage details](experiment.json)

# Repeating IOI interventions at two other heads

[Overview](../../README.md#all-20-experiments-at-a-glance) · E18 · GPT-2 · IOI

Repeat the IOI donor intervention at L9H6 and L10H0 using the same source families. Report continuous matching effects before applying the conventional 0.01-nat margin.

**Result.** Matching advantages are about 0.025 and 0.008 nat. Both retained intervals exclude zero; the L10H0 interval does not establish an effect above the conventional 0.01-nat margin.

**Interpretation.** Positive effects; L10H0 does not clear the conventional 0.01-nat margin.

**Limit.** These use the same GPT-2 checkpoint and source families. They are intervention replications, not independent model or research replications. The practical margin has no independent utility calibration.

## Run from this folder

```sh
python3 run.py check
python3 run.py stages
python3 run.py run heads --plan
```

`check` executes [the local recalculation](src/check_recorded.py), without a model. Remove `--plan` to execute an inference stage after [setup](../../common/README.md). New outputs go to `runs/<name>/`.

**Inputs:** Same IOI assets and development gate as experiment 03. **Tested here:** recorded-data check. Other full workflows remain unvalidated after relocation.

[Code](src/) · [Inputs](inputs/) · [Saved results](recorded/) · [Stage details](experiment.json)

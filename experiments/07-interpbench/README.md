# An extra mismatch check did not improve prediction

[Overview](../../README.md#all-20-experiments-at-a-glance) · E20 · Tracr · InterpBench

Use reserved InterpBench tasks 13, 18, 19 and 21. Compare the five-condition rule with internal ablations on future-donor queries, checking Craft/RASP agreement.

**Result.** On four reserved tasks and 5,632 queries, the five-condition rule has 50.52% coverage and 2.35% neural contradiction risk. Without mismatch, these are 50.90% and 2.34%.

**Interpretation.** No added benefit from the extra mismatch condition.

**Limit.** These are finite task averages, not a matched-coverage ranking. One task has substantial native model-program disagreement. The internal ablations are not independent competing methods.

## Run from this folder

```sh
python3 run.py check
python3 run.py stages
python3 run.py run reserve --plan
```

`check` executes [the local recalculation](src/check_recorded.py), without a model. Remove `--plan` to execute an inference stage after [setup](../../common/README.md). New outputs go to `runs/<name>/`.

**Inputs:** InterpBench weights, Tracr sources, dependencies and fresh seals. **Tested here:** recorded-data check. Other full workflows remain unvalidated after relocation.

[Code](src/) · [Inputs](inputs/) · [Saved results](recorded/) · [Stage details](experiment.json)

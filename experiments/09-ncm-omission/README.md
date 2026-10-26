# Omitting a residual did not establish specific repair

[Overview](../../README.md#all-20-experiments-at-a-glance) · E08 · NCM · Sudoku

Omit residual 9, cross native and regenerated later contributions, and compare recovery and interaction on fresh puzzles. Check that there is damage worth repairing.

**Result.** Mean recovery is -0.001641 nat and the interaction contrast is -0.003803 nat. The announced joint criterion fails.

**Interpretation.** Specific repair not established under this omission.

**Limit.** The omission does not create a consistent mean NLL deficit in the first place. This is a limited repair test and does not show that the generator is generally blind to errors.

## Run from this folder

```sh
python3 run.py check
python3 run.py stages
python3 run.py run crossed --plan
```

`check` executes [the local recalculation](src/check_recorded.py), without a model. Remove `--plan` to execute an inference stage after [setup](../../common/README.md). New outputs go to `runs/<name>/`.

**Inputs:** Sudoku assets and native extraction arrays. **Tested here:** recorded-data check. Other full workflows remain unvalidated after relocation.

[Code](src/) · [Inputs](inputs/) · [Saved results](recorded/) · [Stage details](experiment.json)

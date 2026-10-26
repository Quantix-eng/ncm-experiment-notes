# An abandonment policy saved too little

[Overview](../../README.md#all-20-experiments-at-a-glance) · E04 · NCM · Sudoku

Verify exact pause/resume parity, fit risk policies, then profile abandonment at fixed lost-solution budgets. Compare optimistic savings with the 5% gate.

**Result.** Optimistic savings are 2.29% at a budget of ten lost solutions per 1,000, or 1.47% at three per 1,000. The 5% feasibility gate is not met.

**Interpretation.** The tested early-abandonment policy is not worthwhile.

**Limit.** Abandonment can discard solvable puzzles. The error budget and measured CPU costs belong to this policy and environment.

## Run from this folder

```sh
python3 run.py check
python3 run.py stages
python3 run.py run technical --plan
```

`check` executes [the local recalculation](src/check_recorded.py), without a model. Remove `--plan` to execute an inference stage after [setup](../../common/README.md). New outputs go to `runs/<name>/`.

**Inputs:** Experiment 12 training features and fitted baseline. **Tested here:** recorded-data check. Other full workflows remain unvalidated after relocation.

[Code](src/) · [Inputs](inputs/) · [Saved results](recorded/) · [Stage details](experiment.json)

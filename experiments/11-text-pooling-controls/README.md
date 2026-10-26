# The constant level did not recur in text tokenizers

[Overview](../../README.md#all-20-experiments-at-a-glance) · E07 · NCM · LM1B / TinyStories

Inspect native tokenizer occupancy on random and long-text strata in LM1B and TinyStories. Test local permutations rather than extrapolating from Sudoku.

**Result.** No constant native level was found in the examined random and longer-text strata. Local permutations can change native codes.

**Interpretation.** The Sudoku observation does not generalize to these text checks.

**Limit.** Two tokenizers do not prove that every text level is useful. The raw averaging identity can hold while contextualization and residual quantization remain input-dependent.

## Run from this folder

```sh
python3 run.py check
python3 run.py stages
python3 run.py run prepare --plan
```

`check` executes [the local recalculation](src/check_recorded.py), without a model. Remove `--plan` to execute an inference stage after [setup](../../common/README.md). New outputs go to `runs/<name>/`.

**Inputs:** LM1B/TinyStories tokenizer assets and text data. **Tested here:** recorded-data check. Other full workflows remain unvalidated after relocation.

[Code](src/) · [Inputs](inputs/) · [Saved results](recorded/) · [Stage details](experiment.json)

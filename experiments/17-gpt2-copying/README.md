# The same access test on GPT-2 copying

[Overview](../../README.md#all-20-experiments-at-a-glance) · E14 · GPT-2 · Copying

Apply the exchange design to GPT-2 copying. Require native competence before assigning a response-preference diagnosis; retain unidentified families.

**Result.** Eight of sixteen families pass the native-anchor gate and receive input-dominant diagnoses. The other eight remain unidentified. Regeneration has little exact-answer advantage over the recipient-native frozen control.

**Interpretation.** Protocol transfer, with limited evidence beyond it.

**Limit.** Half the families fail native competence requirements. These are intermediate next-token computations, not edits to text already emitted.

## Run from this folder

```sh
python3 run.py check
python3 run.py stages
python3 run.py run prepare --plan
```

`check` executes [the local recalculation](src/check_recorded.py), without a model. Remove `--plan` to execute an inference stage after [setup](../../common/README.md). New outputs go to `runs/<name>/`.

**Inputs:** GPT-2 checkpoint and copying families. **Tested here:** recorded-data check. Other full workflows remain unvalidated after relocation.

[Code](src/) · [Inputs](inputs/) · [Saved results](recorded/) · [Stage details](experiment.json)

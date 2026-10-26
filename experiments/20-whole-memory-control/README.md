# Replacing the whole memory removes the recipient path

[Overview](../../README.md#all-20-experiments-at-a-glance) · E16 · Toy models · Whole-memory exchange

Exchange the entire retained memory in the toy models, leaving no recipient representation. Compare regenerated outputs with the donor-native output.

**Result.** Regenerated outputs reproduce the donor as required by the access contract. The resulting state-dominant classification follows from the construction.

**Interpretation.** Expected consequence of the access contract.

**Limit.** No recipient influence can be recovered through a path that has been removed. This is a control of the adapter, not a discovered property of intelligence.

## Run from this folder

```sh
python3 run.py check
python3 run.py stages
python3 run.py run structural --plan
```

`check` executes [the local recalculation](src/check_recorded.py), without a model. Remove `--plan` to execute an inference stage after [setup](../../common/README.md). New outputs go to `runs/<name>/`.

**Inputs:** NumPy/PyTorch for structural; trained toy checkpoints from experiment 18 for evaluate. **Tested here:** recorded-data check; `structural`. Other full workflows remain unvalidated after relocation.

[Code](src/) · [Inputs](inputs/) · [Saved results](recorded/) · [Stage details](experiment.json)

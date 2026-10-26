# Input and state exchanges in NCM

[Overview](../../README.md#all-20-experiments-at-a-glance) · E13 · NCM · Sudoku

Exchange input and retained state across Sudoku families under fixed and regenerated tails. Compare with the recipient-native frozen continuation.

**Result.** The operational rule labels 15/16 families input-dominant. Regeneration returns the recipient solution in 32/32 swaps, but a recipient-native frozen continuation does so too.

**Interpretation.** Access-dependent diagnosis; regeneration benefit not established.

**Limit.** That frozen continuation comes from a previous complete computation. The diagnostic depends on the boundary, access contract and chosen margins; it is not a mechanism label or shortcut.

## Run from this folder

```sh
python3 run.py check
python3 run.py stages
python3 run.py run freeze --plan
```

`check` executes [the local recalculation](src/check_recorded.py), without a model. Remove `--plan` to execute an inference stage after [setup](../../common/README.md). New outputs go to `runs/<name>/`.

**Inputs:** Sudoku assets and constructed Sudoku families from experiment 19. **Tested here:** recorded-data check. Other full workflows remain unvalidated after relocation.

[Code](src/) · [Inputs](inputs/) · [Saved results](recorded/) · [Stage details](experiment.json)

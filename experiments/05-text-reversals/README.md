# Token reversals in LM1B generation

[Overview](../../README.md#all-20-experiments-at-a-glance) · E09 · NCM · LM1B

Generate 128 independently seeded LM1B trajectories. Decode each cumulative code state and count returns away from the eventual final token. Compare with oracle encoding.

**Result.** In 128 native LM1B small trajectories, 416/2,340 selected positions leave their eventual final token after adopting it at level 9. An oracle text-codec control also has reconstruction reversals.

**Interpretation.** Readouts are non-monotonic; codec controls also reverse.

**Limit.** The final-token anchor is retrospective and is not a correctness label. Different levels have different resolutions; this does not establish semantic self-correction.

## Run from this folder

```sh
python3 run.py check
python3 run.py stages
python3 run.py run generate --plan
```

`check` executes [the local recalculation](src/check_recorded.py), without a model. Remove `--plan` to execute an inference stage after [setup](../../common/README.md). New outputs go to `runs/<name>/`.

**Inputs:** LM1B model and tokenizer; selected oracle inputs. **Tested here:** recorded-data check. Other full workflows remain unvalidated after relocation.

[Code](src/) · [Inputs](inputs/) · [Saved results](recorded/) · [Stage details](experiment.json)

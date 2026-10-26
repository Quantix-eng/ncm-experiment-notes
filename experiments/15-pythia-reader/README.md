# A reader-only Pythia control

[Overview](../../README.md#all-20-experiments-at-a-glance) · E19 · Pythia-160M · WikiText

Intervene at Pythia-160M’s final parallel layer. Separate final-reader changes from nonexistent downstream block adaptation and check the reader/self-repair identity.

**Result.** Across 512 WikiText sequences, mean reader recovery is -0.000601 nat; the descriptive interval includes zero. The reader/self-repair identity passes its numerical check.

**Interpretation.** Structural identity checked; no average recovery established.

**Limit.** This does not show that LayerNorm never contributes to self-repair. It uses a different distribution and sampling policy from prior work, and does not decompose the IOI result.

## Run from this folder

```sh
python3 run.py check
python3 run.py stages
python3 run.py run prepare --plan
```

`check` executes [the local recalculation](src/check_recorded.py), without a model. Remove `--plan` to execute an inference stage after [setup](../../common/README.md). New outputs go to `runs/<name>/`.

**Inputs:** Pythia-160M checkpoint and WikiText inputs. **Tested here:** recorded-data check. Other full workflows remain unvalidated after relocation.

[Code](src/) · [Inputs](inputs/) · [Saved results](recorded/) · [Stage details](experiment.json)

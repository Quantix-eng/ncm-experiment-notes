# Recovery and donor matching in GPT-2

[Overview](../../README.md#all-20-experiments-at-a-glance) · E17 · GPT-2 · IOI

Replace L9H9 output on paired IOI variants. Compare native, frozen, recomputed and wrong-donor continuations. Aggregate two donor assignments by source family.

**Result.** On 512 source families, L9H9 has 2.752 nats of frozen-tail damage, 1.910 nats of recovery and a 0.038-nat matching advantage. Replaying published backups increases loss more than the control group.

**Interpretation.** Recovery exceeds matching; compensation itself is already known.

**Limit.** IOI compensation and backup heads are established findings. Two fixed donor assignments do not control displacement norm or difficulty. Shared donors also limit bootstrap interpretation.

## Run from this folder

```sh
python3 run.py check
python3 run.py stages
python3 run.py run prepare --plan
```

`check` executes [the local recalculation](src/check_recorded.py), without a model. Remove `--plan` to execute an inference stage after [setup](../../common/README.md). New outputs go to `runs/<name>/`.

**Inputs:** GPT-2 checkpoint, selected IOI prompts and fresh local freezes. **Tested here:** recorded-data check. Other full workflows remain unvalidated after relocation.

[Code](src/) · [Inputs](inputs/) · [Saved results](recorded/) · [Stage details](experiment.json)

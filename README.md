# NCM experiment notes

Twenty exploratory studies on NCM and related model controls, with code, recorded results and bounded interpretations.

Start with the table below. Each experiment has its own source, inputs, saved results and entry point. Shared model adapters and utilities live in [`common/`](common/README.md).

## Try one experiment

```sh
git clone https://github.com/Quantix-eng/ncm-experiment-notes.git
cd ncm-experiment-notes/experiments/02-terminal-readout
python3 run.py check
python3 run.py stages
```

The first command recalculates this experiment’s saved counts. It does not rerun the model. The second lists its real inference stages. With the shared setup installed, `python3 run.py run sample` runs a small model trial in that same folder. See [setup and execution](docs/reproducing.md) for dependencies, assets and current validation status.

## All 20 experiments at a glance

Read across to compare the question area, recorded outcome and what we take from it. The display order starts with the most concrete observations, followed by their controls and less conclusive explorations. It is not a ranking of novelty or statistical strength. Several entries share models or data.

**Outcome labels:** observation = a measured pattern; mixed = a useful result with an important failed extension or control; negative = the tested benefit or criterion was not established; control = primarily a check on scope or implementation. These labels describe outcomes, not scientific quality.

GitHub Markdown tables cannot be sorted by clicking a column. Download the [CSV catalogue](experiments/catalog.csv) to filter or sort by domain, outcome or reading order in a spreadsheet. The [JSON catalogue](experiments/catalog.json) also includes full caveats and source links. Metrics across different tasks are not directly comparable.

| Order / original ID | Experiment | Domain | Outcome | Recorded result | Interpretation / limit |
| --- | --- | --- | --- | --- | --- |
| 01 / E06 | [A pooling symmetry in solved Sudoku](experiments/01-pooling/README.md) | NCM · Sudoku | Observation | 15/16 encoding codes active with weighted pooling; generator agreement falls to 1.65%. | Encoding diagnosis; no generation improvement. |
| 02 / E10 | [Reading and editing terminal codes](experiments/02-terminal-readout/README.md) | NCM · Sudoku | Mixed | 784/1,024 requested writes; 552/1,024 fresh puzzles certified by lookup + checker. | Partial terminal readout; no full-pipeline speedup established. |
| 03 / E17 | [Recovery and donor matching in GPT-2](experiments/03-ioi-recovery/README.md) | GPT-2 · IOI | Observation | Recovery 1.910 nats; donor matching adds 0.038 nat at L9H9. | Recovery exceeds matching; compensation itself is already known. |
| 04 / E11 | [Two terminal edits can interfere](experiments/04-terminal-composition/README.md) | NCM · Sudoku | Mixed | Both targets written in 55.12% of 10,240 double edits. | Edits interfere; single-edit success does not imply composition. |
| 05 / E09 | [Token reversals in LM1B generation](experiments/05-text-reversals/README.md) | NCM · LM1B | Observation | 416/2,340 selected positions leave the final token after reaching it at level 9. | Readouts are non-monotonic; codec controls also reverse. |
| 06 / E03 | [A correct intermediate digit can change again](experiments/06-sudoku-reversals/README.md) | NCM · Sudoku | Observation | 25.04% of cells corrected at levels 8–9 become wrong later. | A correct intermediate digit is not a stable commitment. |
| 07 / E20 | [An extra mismatch check did not improve prediction](experiments/07-interpbench/README.md) | Tracr · InterpBench | Negative | Coverage/risk: 50.52%/2.35%; without mismatch: 50.90%/2.34%. | No added benefit from the extra mismatch condition. |
| 08 / E18 | [Repeating IOI interventions at two other heads](experiments/08-ioi-secondary-heads/README.md) | GPT-2 · IOI | Mixed | Matching: 0.025 nat at L9H6; 0.008 nat at L10H0. | Positive effects; L10H0 does not clear the conventional 0.01-nat margin. |
| 09 / E08 | [Omitting a residual did not establish specific repair](experiments/09-ncm-omission/README.md) | NCM · Sudoku | Negative | Recovery −0.001641 nat; joint criterion fails. | Specific repair not established under this omission. |
| 10 / E05 | [The initial NCM audit](experiments/10-initial-audit/README.md) | NCM · Sudoku | Mixed | Naked-single probe AUROC 0.877; causal ablation not specific. | Decodability does not establish causal use of the feature. |
| 11 / E07 | [The constant level did not recur in text tokenizers](experiments/11-text-pooling-controls/README.md) | NCM · LM1B / TinyStories | Control | No constant native level in the examined text strata. | The Sudoku observation does not generalize to these text checks. |
| 12 / E01 | [Internal states added little to failure prediction](experiments/12-failure-probes/README.md) | NCM · Sudoku | Negative | Hidden-state contrast adds about 0.0043 AUROC; stronger observable baseline wins. | No convincing practical benefit from internal-state features. |
| 13 / E04 | [An abandonment policy saved too little](experiments/13-early-abandonment/README.md) | NCM · Sudoku | Negative | Optimistic savings 2.29% at 10 lost solutions per 1,000; gate was 5%. | The tested early-abandonment policy is not worthwhile. |
| 14 / E02 | [Predicting which cell errors persist](experiments/14-future-corrections/README.md) | NCM · Sudoku | Negative | Predictive increment −0.00060; interval spans zero. | Local latent features do not help this fitted persistence predictor. |
| 15 / E19 | [A reader-only Pythia control](experiments/15-pythia-reader/README.md) | Pythia-160M · WikiText | Control | Mean reader recovery −0.000601 nat; interval includes zero. | Structural identity checked; no average recovery established. |
| 16 / E13 | [Input and state exchanges in NCM](experiments/16-ncm-exchanges/README.md) | NCM · Sudoku | Mixed | 15/16 families input-dominant; frozen recipient tail also succeeds in 32/32 swaps. | Access-dependent diagnosis; regeneration benefit not established. |
| 17 / E14 | [The same access test on GPT-2 copying](experiments/17-gpt2-copying/README.md) | GPT-2 · Copying | Mixed | 8/16 families pass native competence; 8 remain unidentified. | Protocol transfer, with limited evidence beyond it. |
| 18 / E15 | [AR and parallel toy models on one task](experiments/18-matched-toy-models/README.md) | Toy AR / parallel · Symbol rotation | Negative | 6/6 models solve native examples; shared damage gate fails. | No valid general comparison of AR and parallel repair. |
| 19 / E12 | [Checking the protocol on constructed programs](experiments/19-constructed-controls/README.md) | Constructed programs | Control | 344/344 intended diagnoses on jointly designed instances. | Implementation consistency, not external mechanism validation. |
| 20 / E16 | [Replacing the whole memory removes the recipient path](experiments/20-whole-memory-control/README.md) | Toy models · Whole-memory exchange | Control | Outputs follow the donor when all recipient state is removed. | Expected consequence of the access contract. |

## Repository layout

```text
common/                       Shared adapters, dependencies and execution support
experiments/02-terminal-readout/
  run.py                      Local command line entry point
  experiment.json             Stage arguments and required inputs
  src/                        Scientific code and recorded-data recalculation
  inputs/                     Protocol settings and small input files
  recorded/                   Saved results
  runs/                       New executions, ignored by Git
```

The [connections guide](experiments/README.md) explains which studies to read together. [Scope](docs/scope.md) separates observations from claims, and [related work](docs/related-work.md) identifies prior findings. There is no claimed model-performance breakthrough or independent human replication. AI tools contributed substantially to planning, code, analysis and writing.

This work is independent of [Hazy Research NCM](https://github.com/HazyResearch/ncm). Upstream copyright notices are retained in [LICENSE](LICENSE) and [NOTICE](NOTICE.md). The [source inventory](provenance/source-inventory.json) records the relocation and omitted numerical arrays. The earlier archive remains in Git history, rather than a second working tree in this repository.

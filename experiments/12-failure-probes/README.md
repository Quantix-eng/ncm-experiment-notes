# Internal states added little to failure prediction

[All experiments](../../README.md#all-20-experiments-at-a-glance) · [Connections between studies](../README.md) · [Reproduction](../../docs/reproducing.md)

Original ID **E01** · NCM · Sudoku · Outcome: **negative**

Ask whether internal activations predict final Sudoku failure better than observable readout features.

## Recorded result

The reserved 500-puzzle study reports an AUROC increment of about 0.0043 for one fitted contrast. A stronger observable-only summary outperforms that state-augmented comparison.

## Interpretation

A useful stopping point for the original hypothesis: adding hidden-state features did not produce a convincing practical benefit.

## Limits

A small conditional improvement is not an operational advantage. These are externally fitted predictors, not native confidence or metacognition. Earlier probe objectives are separate experiments.

## Code and protocol

- [experiments/metacognition](../../historical/Experience_NCM/experiments/metacognition)

## Saved records

- [audit/metacognition/final-v0.3/analysis/report.json](../../historical/Experience_NCM/audit/metacognition/final-v0.3/analysis/report.json)
- [RAPPORT_METACOGNITION.md](../../historical/Experience_NCM/RAPPORT_METACOGNITION.md)

These are historical records. They were not regenerated for this page. The [reproduction guide](../../docs/reproducing.md) distinguishes quick recalculations, the checkpoint smoke test and full campaign reruns. The [export inventory](../../provenance/historical-manifest.json) records omitted traces.

## Read alongside

- [Predicting which cell errors persist](../14-future-corrections/README.md).
- [An abandonment policy saved too little](../13-early-abandonment/README.md).

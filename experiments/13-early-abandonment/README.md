# An abandonment policy saved too little

[All experiments](../../README.md#all-20-experiments-at-a-glance) · [Connections between studies](../README.md) · [Reproduction](../../docs/reproducing.md)

Original ID **E04** · NCM · Sudoku · Outcome: **negative**

Use a level-9 risk signal to stop trajectories likely to fail and measure the available computation savings.

## Recorded result

Optimistic savings are 2.29% at a budget of ten lost solutions per 1,000, or 1.47% at three per 1,000. The 5% feasibility gate is not met.

## Interpretation

A practical negative result. It explains why the confidence work did not become an inference optimization.

## Limits

Abandonment can discard solvable puzzles. The error budget and measured CPU costs belong to this policy and environment.

## Code and protocol

- [experiments/metacognition/early_control](../../historical/Experience_NCM/experiments/metacognition/early_control)

## Saved records

- [audit/metacognition/early-control-v1/campaign-summary.json](../../historical/Experience_NCM/audit/metacognition/early-control-v1/campaign-summary.json)
- [RAPPORT_CONTROLE_PRECOCE.md](../../historical/Experience_NCM/RAPPORT_CONTROLE_PRECOCE.md)

These are historical records. They were not regenerated for this page. The [reproduction guide](../../docs/reproducing.md) distinguishes quick recalculations, the checkpoint smoke test and full campaign reruns. The [export inventory](../../provenance/historical-manifest.json) records omitted traces.

## Read alongside

- [Internal states added little to failure prediction](../12-failure-probes/README.md).
- [Predicting which cell errors persist](../14-future-corrections/README.md).

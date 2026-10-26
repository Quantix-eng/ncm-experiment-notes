# Predicting which cell errors persist

[All experiments](../../README.md#all-20-experiments-at-a-glance) · [Connections between studies](../README.md) · [Reproduction](../../docs/reproducing.md)

Original ID **E02** · NCM · Sudoku · Outcome: **negative**

Test whether local latent features improve prediction of which currently wrong cells will remain wrong.

## Recorded result

On the restricted mixed-puzzle cohort, the reported increment is -0.00060 with an interval spanning zero. Most current errors are corrected later, creating a strong prevalence trap.

## Interpretation

Another hypothesis that did not pay off. Retaining the baseline and cohort definition matters more than the number of attempted probes.

## Limits

The cohort is selected and the target is a future event. The result concerns this fitted procedure, not the absence of information everywhere in the network.

## Code and protocol

- [experiments/metacognition/correction](../../historical/Experience_NCM/experiments/metacognition/correction)

## Saved records

- [audit/metacognition/correction-v0.1/external/analysis/report.json](../../historical/Experience_NCM/audit/metacognition/correction-v0.1/external/analysis/report.json)
- [RAPPORT_CORRECTION_FUTURE.md](../../historical/Experience_NCM/RAPPORT_CORRECTION_FUTURE.md)

These are historical records. They were not regenerated for this page. The [reproduction guide](../../docs/reproducing.md) distinguishes quick recalculations, the checkpoint smoke test and full campaign reruns. The [export inventory](../../provenance/historical-manifest.json) records omitted traces.

## Read alongside

- [Internal states added little to failure prediction](../12-failure-probes/README.md).
- [An abandonment policy saved too little](../13-early-abandonment/README.md).

# An extra mismatch check did not improve prediction

[All experiments](../../README.md#all-20-experiments-at-a-glance) · [Connections between studies](../README.md) · [Reproduction](../../docs/reproducing.md)

Original ID **E20** · Tracr · InterpBench · Outcome: **negative**

Use probe interventions to predict whether a future donor exchange preserves the native output.

## Recorded result

On four reserved tasks and 5,632 queries, the five-condition rule has 50.52% coverage and 2.35% neural contradiction risk. Without mismatch, these are 50.90% and 2.34%.

## Interpretation

An important negative result for the proposed use of matching. More conditions did not establish better predictions in this test.

## Limits

These are finite task averages, not a matched-coverage ranking. One task has substantial native model-program disagreement. The internal ablations are not independent competing methods.

## Code and protocol

- [analyse_fondamentale_NCM/validation_externe_directeur/02_interpbench](../../historical/Experience_NCM/analyse_fondamentale_NCM/validation_externe_directeur/02_interpbench)

## Saved records

- [analyse_fondamentale_NCM/validation_externe_directeur/02_interpbench/results/c2_summary.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/validation_externe_directeur/02_interpbench/results/c2_summary.json)
- [analyse_fondamentale_NCM/validation_externe_directeur/02_interpbench/RAPPORT_C2.md](../../historical/Experience_NCM/analyse_fondamentale_NCM/validation_externe_directeur/02_interpbench/RAPPORT_C2.md)

These are historical records. They were not regenerated for this page. The [reproduction guide](../../docs/reproducing.md) distinguishes quick recalculations, the checkpoint smoke test and full campaign reruns. The [export inventory](../../provenance/historical-manifest.json) records omitted traces.

## Read alongside

- [Checking the protocol on constructed programs](../19-constructed-controls/README.md).
- [Replacing the whole memory removes the recipient path](../20-whole-memory-control/README.md).

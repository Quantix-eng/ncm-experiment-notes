# Checking the protocol on constructed programs

[All experiments](../../README.md#all-20-experiments-at-a-glance) · [Connections between studies](../README.md) · [Reproduction](../../docs/reproducing.md)

Original ID **E12** · Constructed programs · Outcome: **control**

Check whether the observation, continuation and exchange rules distinguish programs with known behavior.

## Recorded result

The complete diagnostic returns the intended answer in 344/344 constructed instances. The program library and diagnostic were designed together.

## Interpretation

Keep it as an implementation and logic check rather than a headline finding.

## Limits

This is a consistency check, not external validation or a benchmark of mechanism discovery on learned networks. Observation-only failure on these programs is not a universal upper bound.

## Code and protocol

- [analyse_fondamentale_NCM/benchmark_causal_revision](../../historical/Experience_NCM/analyse_fondamentale_NCM/benchmark_causal_revision)

## Saved records

- [analyse_fondamentale_NCM/benchmark_causal_revision/results/summary.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/benchmark_causal_revision/results/summary.json)

These are historical records. They were not regenerated for this page. The [reproduction guide](../../docs/reproducing.md) distinguishes quick recalculations, the checkpoint smoke test and full campaign reruns. The [export inventory](../../provenance/historical-manifest.json) records omitted traces.

## Read alongside

- [An extra mismatch check did not improve prediction](../07-interpbench/README.md).
- [Replacing the whole memory removes the recipient path](../20-whole-memory-control/README.md).

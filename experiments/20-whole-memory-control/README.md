# Replacing the whole memory removes the recipient path

[All experiments](../../README.md#all-20-experiments-at-a-glance) · [Connections between studies](../README.md) · [Reproduction](../../docs/reproducing.md)

Original ID **E16** · Toy models · Whole-memory exchange · Outcome: **control**

Exchange every available state tensor, leaving no recipient representation for the remaining computation.

## Recorded result

Regenerated outputs reproduce the donor as required by the access contract. The resulting state-dominant classification follows from the construction.

## Interpretation

The least substantive scientific result, but a useful check that the implementation respects its declared inputs.

## Limits

No recipient influence can be recovered through a path that has been removed. This is a control of the adapter, not a discovered property of intelligence.

## Code and protocol

- [analyse_fondamentale_NCM/benchmark_tache_commune/evaluate.py](../../historical/Experience_NCM/analyse_fondamentale_NCM/benchmark_tache_commune/evaluate.py)

## Saved records

- [analyse_fondamentale_NCM/benchmark_tache_commune/results/summary.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/benchmark_tache_commune/results/summary.json)

These are historical records. They were not regenerated for this page. The [reproduction guide](../../docs/reproducing.md) distinguishes quick recalculations, the checkpoint smoke test and full campaign reruns. The [export inventory](../../provenance/historical-manifest.json) records omitted traces.

## Read alongside

- [An extra mismatch check did not improve prediction](../07-interpbench/README.md).
- [Checking the protocol on constructed programs](../19-constructed-controls/README.md).

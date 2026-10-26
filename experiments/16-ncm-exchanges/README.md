# Input and state exchanges in NCM

[All experiments](../../README.md#all-20-experiments-at-a-glance) · [Connections between studies](../README.md) · [Reproduction](../../docs/reproducing.md)

Original ID **E13** · NCM · Sudoku · Outcome: **mixed**

Exchange retained states between controlled Sudoku input pairs while varying which input context remains available.

## Recorded result

The operational rule labels 15/16 families input-dominant. Regeneration returns the recipient solution in 32/32 swaps, but a recipient-native frozen continuation does so too.

## Interpretation

A small functional demonstration that is useful mainly for understanding the intervention design.

## Limits

That frozen continuation comes from a previous complete computation. The diagnostic depends on the boundary, access contract and chosen margins; it is not a mechanism label or shortcut.

## Code and protocol

- [analyse_fondamentale_NCM/benchmark_ncm_reel](../../historical/Experience_NCM/analyse_fondamentale_NCM/benchmark_ncm_reel)

## Saved records

- [analyse_fondamentale_NCM/benchmark_ncm_reel/results/summary.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/benchmark_ncm_reel/results/summary.json)
- [analyse_fondamentale_NCM/benchmark_ncm_reel/results/output_quality.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/benchmark_ncm_reel/results/output_quality.json)

These are historical records. They were not regenerated for this page. The [reproduction guide](../../docs/reproducing.md) distinguishes quick recalculations, the checkpoint smoke test and full campaign reruns. The [export inventory](../../provenance/historical-manifest.json) records omitted traces.

## Read alongside

- [The same access test on GPT-2 copying](../17-gpt2-copying/README.md).
- [AR and parallel toy models on one task](../18-matched-toy-models/README.md).

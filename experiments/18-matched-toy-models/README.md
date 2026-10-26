# AR and parallel toy models on one task

[All experiments](../../README.md#all-20-experiments-at-a-glance) · [Connections between studies](../README.md) · [Reproduction](../../docs/reproducing.md)

Original ID **E15** · Toy AR / parallel · Symbol rotation · Outcome: **negative**

Train three paired seeds of small autoregressive and parallel Transformers on the same symbol-rotation task.

## Recorded result

All six models solve the native examples, but no tested corruption dose meets the shared damage gate. Recovery contrasts at the fallback dose are negative. Cutoffs and seeds change exchange behavior.

## Interpretation

A control experiment that sets a boundary on what the earlier cross-model comparisons can claim.

## Limits

The calibration objective fails. The parallel model is not NCM, and the training objectives differ. These runs cannot identify a general effect of autoregressive versus parallel generation.

## Code and protocol

- [analyse_fondamentale_NCM/benchmark_tache_commune](../../historical/Experience_NCM/analyse_fondamentale_NCM/benchmark_tache_commune)

## Saved records

- [analyse_fondamentale_NCM/benchmark_tache_commune/results/calibration.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/benchmark_tache_commune/results/calibration.json)
- [analyse_fondamentale_NCM/benchmark_tache_commune/results/summary.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/benchmark_tache_commune/results/summary.json)

These are historical records. They were not regenerated for this page. The [reproduction guide](../../docs/reproducing.md) distinguishes quick recalculations, the checkpoint smoke test and full campaign reruns. The [export inventory](../../provenance/historical-manifest.json) records omitted traces.

## Read alongside

- [Input and state exchanges in NCM](../16-ncm-exchanges/README.md).
- [The same access test on GPT-2 copying](../17-gpt2-copying/README.md).

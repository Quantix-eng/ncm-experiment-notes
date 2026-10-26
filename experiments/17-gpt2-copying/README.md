# The same access test on GPT-2 copying

[All experiments](../../README.md#all-20-experiments-at-a-glance) · [Connections between studies](../README.md) · [Reproduction](../../docs/reproducing.md)

Original ID **E14** · GPT-2 · Copying · Outcome: **mixed**

Port the input/state exchange protocol to a next-token copying task in GPT-2.

## Recorded result

Eight of sixteen families pass the native-anchor gate and receive input-dominant diagnoses. The other eight remain unidentified. Regeneration has little exact-answer advantage over the recipient-native frozen control.

## Interpretation

Evidence that the implementation transfers, with weak evidence for a broader scientific claim.

## Limits

Half the families fail native competence requirements. These are intermediate next-token computations, not edits to text already emitted.

## Code and protocol

- [analyse_fondamentale_NCM/benchmark_multimodel_ar](../../historical/Experience_NCM/analyse_fondamentale_NCM/benchmark_multimodel_ar)

## Saved records

- [analyse_fondamentale_NCM/benchmark_multimodel_ar/results/summary.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/benchmark_multimodel_ar/results/summary.json)

These are historical records. They were not regenerated for this page. The [reproduction guide](../../docs/reproducing.md) distinguishes quick recalculations, the checkpoint smoke test and full campaign reruns. The [export inventory](../../provenance/historical-manifest.json) records omitted traces.

## Read alongside

- [Input and state exchanges in NCM](../16-ncm-exchanges/README.md).
- [AR and parallel toy models on one task](../18-matched-toy-models/README.md).

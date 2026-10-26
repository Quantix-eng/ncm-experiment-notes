# A reader-only Pythia control

[All experiments](../../README.md#all-20-experiments-at-a-glance) · [Connections between studies](../README.md) · [Reproduction](../../docs/reproducing.md)

Original ID **E19** · Pythia-160M · WikiText · Outcome: **control**

Use the parallel branches in Pythia-160M final layer to separate changes in the reader from downstream block adaptation.

## Recorded result

Across 512 WikiText sequences, mean reader recovery is -0.000601 nat; the descriptive interval includes zero. The reader/self-repair identity passes its numerical check.

## Interpretation

A structural implementation control with a negative average outcome.

## Limits

This does not show that LayerNorm never contributes to self-repair. It uses a different distribution and sampling policy from prior work, and does not decompose the IOI result.

## Code and protocol

- [analyse_fondamentale_NCM/validation_externe_directeur/03_pythia/scripts](../../historical/Experience_NCM/analyse_fondamentale_NCM/validation_externe_directeur/03_pythia/scripts)

## Saved records

- [analyse_fondamentale_NCM/validation_externe_directeur/03_pythia/results/test_summary.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/validation_externe_directeur/03_pythia/results/test_summary.json)

These are historical records. They were not regenerated for this page. The [reproduction guide](../../docs/reproducing.md) distinguishes quick recalculations, the checkpoint smoke test and full campaign reruns. The [export inventory](../../provenance/historical-manifest.json) records omitted traces.

## Read alongside

- [Recovery and donor matching in GPT-2](../03-ioi-recovery/README.md).
- [Repeating IOI interventions at two other heads](../08-ioi-secondary-heads/README.md).
- [Omitting a residual did not establish specific repair](../09-ncm-omission/README.md).

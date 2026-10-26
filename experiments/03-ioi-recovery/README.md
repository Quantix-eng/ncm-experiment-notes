# Recovery and donor matching in GPT-2

[All experiments](../../README.md#all-20-experiments-at-a-glance) · [Connections between studies](../README.md) · [Reproduction](../../docs/reproducing.md)

Original ID **E17** · GPT-2 · IOI · Outcome: **observation**

After replacing a Name Mover head output, compare a frozen continuation, a recomputed continuation and one adapted to another donor.

## Recorded result

On 512 source families, L9H9 has 2.752 nats of frozen-tail damage, 1.910 nats of recovery and a 0.038-nat matching advantage. Replaying published backups increases loss more than the control group.

## Interpretation

The main question is what matching adds beyond recovery. These results support a bounded case study, not a new discovery of compensation.

## Limits

IOI compensation and backup heads are established findings. Two fixed donor assignments do not control displacement norm or difficulty. Shared donors also limit bootstrap interpretation.

## Code and protocol

- [analyse_fondamentale_NCM/validation_externe_directeur/01_ioi](../../historical/Experience_NCM/analyse_fondamentale_NCM/validation_externe_directeur/01_ioi)

## Saved records

- [analyse_fondamentale_NCM/validation_externe_directeur/01_ioi/results/test_observations.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/validation_externe_directeur/01_ioi/results/test_observations.json)
- [analyse_fondamentale_NCM/validation_externe_directeur/01_ioi/results/test_summary.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/validation_externe_directeur/01_ioi/results/test_summary.json)

These are historical records. They were not regenerated for this page. The [reproduction guide](../../docs/reproducing.md) distinguishes quick recalculations, the checkpoint smoke test and full campaign reruns. The [export inventory](../../provenance/historical-manifest.json) records omitted traces.

## Read alongside

- [Repeating IOI interventions at two other heads](../08-ioi-secondary-heads/README.md).
- [Omitting a residual did not establish specific repair](../09-ncm-omission/README.md).
- [A reader-only Pythia control](../15-pythia-reader/README.md).

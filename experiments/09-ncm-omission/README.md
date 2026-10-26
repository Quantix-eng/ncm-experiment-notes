# Omitting a residual did not establish specific repair

[All experiments](../../README.md#all-20-experiments-at-a-glance) · [Connections between studies](../README.md) · [Reproduction](../../docs/reproducing.md)

Original ID **E08** · NCM · Sudoku · Outcome: **negative**

Cross omission of residual 9 with frozen and regenerated later contributions on 256 fresh puzzles.

## Recorded result

Mean recovery is -0.001641 nat and the interaction contrast is -0.003803 nat. The announced joint criterion fails.

## Interpretation

A necessary negative result to keep next to any discussion of adaptation. Changed later codes alone do not establish useful compensation.

## Limits

The omission does not create a consistent mean NLL deficit in the first place. This is a limited repair test and does not show that the generator is generally blind to errors.

## Code and protocol

- [analyse_fondamentale_NCM/compensation_conditionnelle/scripts/crossed.py](../../historical/Experience_NCM/analyse_fondamentale_NCM/compensation_conditionnelle/scripts/crossed.py)

## Saved records

- [analyse_fondamentale_NCM/compensation_conditionnelle/results/fresh_crossed.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/compensation_conditionnelle/results/fresh_crossed.json)

These are historical records. They were not regenerated for this page. The [reproduction guide](../../docs/reproducing.md) distinguishes quick recalculations, the checkpoint smoke test and full campaign reruns. The [export inventory](../../provenance/historical-manifest.json) records omitted traces.

## Read alongside

- [Recovery and donor matching in GPT-2](../03-ioi-recovery/README.md).
- [Repeating IOI interventions at two other heads](../08-ioi-secondary-heads/README.md).
- [A reader-only Pythia control](../15-pythia-reader/README.md).

# Two terminal edits can interfere

[All experiments](../../README.md#all-20-experiments-at-a-glance) · [Connections between studies](../README.md) · [Reproduction](../../docs/reproducing.md)

Original ID **E11** · NCM · Sudoku · Outcome: **mixed**

Do two individually useful terminal substitutions still work when applied together?

## Recorded result

Across 10,240 double edits, both requested targets are written in 55.12% of trials and the complete contract holds in 54.45%. Retention is lower for adjacent cells.

## Interpretation

A useful limit on the terminal-readout result. It prevents the single-edit experiment from being read as a context-free editing interface.

## Limits

The terminal residual includes a local convolution, so spatial interference is not surprising by itself. Edits share sources and must not be counted as independent puzzles.

## Code and protocol

- [analyse_fondamentale_NCM/composition_terminale/experiment.py](../../historical/Experience_NCM/analyse_fondamentale_NCM/composition_terminale/experiment.py)

## Saved records

- [analyse_fondamentale_NCM/composition_terminale/results/summary.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/composition_terminale/results/summary.json)

These are historical records. They were not regenerated for this page. The [reproduction guide](../../docs/reproducing.md) distinguishes quick recalculations, the checkpoint smoke test and full campaign reruns. The [export inventory](../../provenance/historical-manifest.json) records omitted traces.

## Read alongside

- [Reading and editing terminal codes](../02-terminal-readout/README.md).

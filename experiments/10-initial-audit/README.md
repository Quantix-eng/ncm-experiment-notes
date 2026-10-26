# The initial NCM audit

[All experiments](../../README.md#all-20-experiments-at-a-glance) · [Connections between studies](../README.md) · [Reproduction](../../docs/reproducing.md)

Original ID **E05** · NCM · Sudoku · Outcome: **mixed**

Inspect reconstruction, code usage, deterministic repeats, symmetries, checkpoint transfer and simple probes.

## Recorded result

The audit found the constant intermediate level and a naked-single probe with AUROC 0.877 versus 0.496 with shuffled labels. The causal ablation was not specific.

## Interpretation

Background for the more focused pooling experiments. It is a collection of diagnostics rather than a complete causal anatomy.

## Limits

Naked singles can be computed directly from the puzzle. Decodability alone does not show that the model uses the feature or implements a human Sudoku rule.

## Code and protocol

- [analyse_fondamentale_NCM/scripts](../../historical/Experience_NCM/analyse_fondamentale_NCM/scripts)

## Saved records

- [analyse_fondamentale_NCM/results/tokenizer.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/results/tokenizer.json)
- [analyse_fondamentale_NCM/results/naked_single.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/results/naked_single.json)
- [analyse_fondamentale_NCM/results/causal.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/results/causal.json)

These are historical records. They were not regenerated for this page. The [reproduction guide](../../docs/reproducing.md) distinguishes quick recalculations, the checkpoint smoke test and full campaign reruns. The [export inventory](../../provenance/historical-manifest.json) records omitted traces.

## Read alongside

- [A pooling symmetry in solved Sudoku](../01-pooling/README.md).
- [The constant level did not recur in text tokenizers](../11-text-pooling-controls/README.md).

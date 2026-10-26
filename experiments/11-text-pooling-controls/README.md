# The constant level did not recur in text tokenizers

[All experiments](../../README.md#all-20-experiments-at-a-glance) · [Connections between studies](../README.md) · [Reproduction](../../docs/reproducing.md)

Original ID **E07** · NCM · LM1B / TinyStories · Outcome: **control**

Audit LM1B and TinyStories tokenizers for the same kind of constant native level.

## Recorded result

No constant native level was found in the examined random and longer-text strata. Local permutations can change native codes.

## Interpretation

This keeps the Sudoku observation from becoming a claim about all NCMs.

## Limits

Two tokenizers do not prove that every text level is useful. The raw averaging identity can hold while contextualization and residual quantization remain input-dependent.

## Code and protocol

- [analyse_fondamentale_NCM/generalisation_architecturale/scripts/text_audit.py](../../historical/Experience_NCM/analyse_fondamentale_NCM/generalisation_architecturale/scripts/text_audit.py)

## Saved records

- [analyse_fondamentale_NCM/generalisation_architecturale/README.md](../../historical/Experience_NCM/analyse_fondamentale_NCM/generalisation_architecturale/README.md)
- [analyse_fondamentale_NCM/generalisation_architecturale/results](../../historical/Experience_NCM/analyse_fondamentale_NCM/generalisation_architecturale/results)

These are historical records. They were not regenerated for this page. The [reproduction guide](../../docs/reproducing.md) distinguishes quick recalculations, the checkpoint smoke test and full campaign reruns. The [export inventory](../../provenance/historical-manifest.json) records omitted traces.

## Read alongside

- [A pooling symmetry in solved Sudoku](../01-pooling/README.md).
- [The initial NCM audit](../10-initial-audit/README.md).

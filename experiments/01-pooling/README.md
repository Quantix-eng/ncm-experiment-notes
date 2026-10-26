# A pooling symmetry in solved Sudoku

[All experiments](../../README.md#all-20-experiments-at-a-glance) · [Connections between studies](../README.md) · [Reproduction](../../docs/reproducing.md)

Original ID **E06** · NCM · Sudoku · Outcome: **observation**

Averaging raw token-plus-position embeddings loses the order of digits inside a valid Sudoku row. At one nine-position level, the hard checkpoint also uses a constant code on the examined inputs.

## Recorded result

Weighted pooling activates 15 of the 16 encoding identifiers while preserving the recorded reconstruction counts. Agreement with the frozen generator target falls from 100% to 1.65%.

## Interpretation

This is the clearest small observation to discuss with the NCM authors. The useful part is the checkpoint diagnosis and its limits; the averaging identity itself is elementary.

## Limits

The algebra applies before GroupNorm and earlier residual subtraction. It does not prove that native quantization must collapse. The modified encoding path is not used by native generation, and no solving improvement follows.

## Code and protocol

- [analyse_fondamentale_NCM/directions_mecanistes/scripts](../../historical/Experience_NCM/analyse_fondamentale_NCM/directions_mecanistes/scripts)
- [analyse_fondamentale_NCM/correction_pooling/scripts](../../historical/Experience_NCM/analyse_fondamentale_NCM/correction_pooling/scripts)

## Saved records

- [analyse_fondamentale_NCM/directions_mecanistes/results/pooling_algebra.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/directions_mecanistes/results/pooling_algebra.json)
- [analyse_fondamentale_NCM/directions_mecanistes/results/level5_diagnostic.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/directions_mecanistes/results/level5_diagnostic.json)
- [analyse_fondamentale_NCM/correction_pooling/results/heldout.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/correction_pooling/results/heldout.json)
- [analyse_fondamentale_NCM/correction_pooling/results/generator_alignment.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/correction_pooling/results/generator_alignment.json)

These are historical records. They were not regenerated for this page. The [reproduction guide](../../docs/reproducing.md) distinguishes quick recalculations, the checkpoint smoke test and full campaign reruns. The [export inventory](../../provenance/historical-manifest.json) records omitted traces.

## Read alongside

- [The initial NCM audit](../10-initial-audit/README.md).
- [The constant level did not recur in text tokenizers](../11-text-pooling-controls/README.md).

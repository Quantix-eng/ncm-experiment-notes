# A correct intermediate digit can change again

[All experiments](../../README.md#all-20-experiments-at-a-glance) · [Connections between studies](../README.md) · [Reproduction](../../docs/reproducing.md)

Original ID **E03** · NCM · Sudoku · Outcome: **observation**

Test whether decoded Sudoku values become permanently fixed at level 9.

## Recorded result

On 600 hard puzzles, 25.04% of the cells corrected between levels 8 and 9 become wrong again later. Earlier complete readouts solve no full grid in the reported cutoff test.

## Interpretation

A direct check on an initially tempting interpretation. Non-monotonic intermediate quality is already acknowledged in the NCM paper; the contribution here is its measurement and oracle control.

## Limits

Decoded grids are intermediate observations, not the generator input. An oracle codec given the correct solution also shows reversals. This rules out strict decoded-value locking, not every possible latent commitment.

## Code and protocol

- [analyse_fondamentale_NCM/dynamique_temporelle/scripts](../../historical/Experience_NCM/analyse_fondamentale_NCM/dynamique_temporelle/scripts)

## Saved records

- [analyse_fondamentale_NCM/dynamique_temporelle/RAPPORT.md](../../historical/Experience_NCM/analyse_fondamentale_NCM/dynamique_temporelle/RAPPORT.md)
- [analyse_fondamentale_NCM/mecanismes_revision/README.md](../../historical/Experience_NCM/analyse_fondamentale_NCM/mecanismes_revision/README.md)

These are historical records. They were not regenerated for this page. The [reproduction guide](../../docs/reproducing.md) distinguishes quick recalculations, the checkpoint smoke test and full campaign reruns. The [export inventory](../../provenance/historical-manifest.json) records omitted traces.

## Read alongside

- [Token reversals in LM1B generation](../05-text-reversals/README.md).

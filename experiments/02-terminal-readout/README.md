# Reading and editing terminal codes

[All experiments](../../README.md#all-20-experiments-at-a-glance) · [Connections between studies](../README.md) · [Reproduction](../../docs/reproducing.md)

Original ID **E10** · NCM · Sudoku · Outcome: **mixed**

Can a simple table read digits from the final Sudoku codes, and can a code replacement write a requested digit?

## Recorded result

Target-associated substitutions succeed in 784/1,024 requests on 128 puzzles. A lookup plus a Sudoku checker certifies 552/1,024 fresh puzzles before falling back to the native decoder.

## Interpretation

A concrete endpoint experiment, with an external verifier that makes the limits easy to inspect. It is not evidence of a human-like symbolic alphabet.

## Limits

Eight requests share each source puzzle. Gaussian controls do not match valid codebook geometry. The certificate depends on clue checks, uniqueness and exact fallback. No whole-model speedup is established.

## Code and protocol

- [analyse_fondamentale_NCM/recherche_positive/scripts](../../historical/Experience_NCM/analyse_fondamentale_NCM/recherche_positive/scripts)

## Saved records

- [analyse_fondamentale_NCM/recherche_positive/results/writing_interventions.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/recherche_positive/results/writing_interventions.json)
- [analyse_fondamentale_NCM/recherche_positive/results/hard_readout.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/recherche_positive/results/hard_readout.json)
- [analyse_fondamentale_NCM/recherche_positive/results/benchmark.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/recherche_positive/results/benchmark.json)

These are historical records. They were not regenerated for this page. The [reproduction guide](../../docs/reproducing.md) distinguishes quick recalculations, the checkpoint smoke test and full campaign reruns. The [export inventory](../../provenance/historical-manifest.json) records omitted traces.

## Read alongside

- [Two terminal edits can interfere](../04-terminal-composition/README.md).

# Token reversals in LM1B generation

[All experiments](../../README.md#all-20-experiments-at-a-glance) · [Connections between studies](../README.md) · [Reproduction](../../docs/reproducing.md)

Original ID **E09** · NCM · LM1B · Outcome: **observation**

Check whether intermediate readouts can leave a token after first adopting the token that appears at the end.

## Recorded result

In 128 native LM1B small trajectories, 416/2,340 selected positions leave their eventual final token after adopting it at level 9. An oracle text-codec control also has reconstruction reversals.

## Interpretation

It extends the readout observation beyond Sudoku, while keeping correctness and agreement with the eventual output separate.

## Limits

The final-token anchor is retrospective and is not a correctness label. Different levels have different resolutions; this does not establish semantic self-correction.

## Code and protocol

- [analyse_fondamentale_NCM/compensation_conditionnelle/scripts/text_generation.py](../../historical/Experience_NCM/analyse_fondamentale_NCM/compensation_conditionnelle/scripts/text_generation.py)

## Saved records

- [analyse_fondamentale_NCM/compensation_conditionnelle/results/lm1b_generation.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/compensation_conditionnelle/results/lm1b_generation.json)
- [analyse_fondamentale_NCM/compensation_conditionnelle/results/lm1b_oracle.json](../../historical/Experience_NCM/analyse_fondamentale_NCM/compensation_conditionnelle/results/lm1b_oracle.json)

These are historical records. They were not regenerated for this page. The [reproduction guide](../../docs/reproducing.md) distinguishes quick recalculations, the checkpoint smoke test and full campaign reruns. The [export inventory](../../provenance/historical-manifest.json) records omitted traces.

## Read alongside

- [A correct intermediate digit can change again](../06-sudoku-reversals/README.md).

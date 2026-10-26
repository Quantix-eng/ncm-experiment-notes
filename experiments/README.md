# How the experiments connect

[Compare all 20 experiments](../README.md#all-20-experiments-at-a-glance) · [Sort or filter the CSV](catalog.csv) · [Reproduction](../docs/reproducing.md)

The main README is the catalogue. This page explains which entries belong together and which comparisons need care. Folder numbers follow the README display order; `E01`–`E20` retain the original experiment identifiers. Neither numbering system measures scientific strength.

## Encoding and code usage

The initial audit led to the focused Sudoku pooling check. The text-tokenizer controls bound how far that observation travels.

- [A pooling symmetry in solved Sudoku](01-pooling/README.md) — E06.
- [The initial NCM audit](10-initial-audit/README.md) — E05.
- [The constant level did not recur in text tokenizers](11-text-pooling-controls/README.md) — E07.

## Reading and editing terminal codes

Read the single-edit result alongside the double-edit test: successful local writes do not guarantee that edits compose.

- [Reading and editing terminal codes](02-terminal-readout/README.md) — E10.
- [Two terminal edits can interfere](04-terminal-composition/README.md) — E11.

## Intermediate readouts

Both studies measure reversals, but the targets differ: a correct Sudoku digit versus agreement with an eventual text token.

- [Token reversals in LM1B generation](05-text-reversals/README.md) — E09.
- [A correct intermediate digit can change again](06-sudoku-reversals/README.md) — E03.

## Recovery and donor matching

The IOI study separates recovery from matching. Other heads, NCM omission and the Pythia reader control test different boundaries; they are not interchangeable replications.

- [Recovery and donor matching in GPT-2](03-ioi-recovery/README.md) — E17.
- [Repeating IOI interventions at two other heads](08-ioi-secondary-heads/README.md) — E18.
- [Omitting a residual did not establish specific repair](09-ncm-omission/README.md) — E08.
- [A reader-only Pythia control](15-pythia-reader/README.md) — E19.

## Prediction and early abandonment

These studies ask whether internal signals support a useful prediction or decision. The practical abandonment test follows the probe work and also fails its target.

- [Internal states added little to failure prediction](12-failure-probes/README.md) — E01.
- [Predicting which cell errors persist](14-future-corrections/README.md) — E02.
- [An abandonment policy saved too little](13-early-abandonment/README.md) — E04.

## Access and exchange protocols

These campaigns transfer the exchange protocol between tasks and models. Competence and damage gates limit the comparisons.

- [Input and state exchanges in NCM](16-ncm-exchanges/README.md) — E13.
- [The same access test on GPT-2 copying](17-gpt2-copying/README.md) — E14.
- [AR and parallel toy models on one task](18-matched-toy-models/README.md) — E15.

## Validation and implementation controls

Constructed programs and whole-memory replacement check the protocol. InterpBench asks whether an extra condition helps on reserved learned networks, with a negative result.

- [An extra mismatch check did not improve prediction](07-interpbench/README.md) — E20.
- [Checking the protocol on constructed programs](19-constructed-controls/README.md) — E12.
- [Replacing the whole memory removes the recipient path](20-whole-memory-control/README.md) — E16.

Several entries reuse checkpoints, source families or traces. Read their limits before treating related results as independent evidence. For definitions, statistical caveats and provenance, see [scope](../docs/scope.md); for prior findings, see [related work](../docs/related-work.md).

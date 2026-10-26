---
title: "Distributed Revision Without Level-9 Locking in Sudoku Next-Concept Models"
date: "24 October 2026"
geometry: margin=21mm
fontsize: 10pt
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
---

## Abstract

A peak of 41,620 incorrect-to-correct cell revisions at generation level 9 in a published Sudoku Next-Concept Model motivates an irreversible-decision hypothesis. We distinguish continuation necessity, residual contribution, local acceleration, decoded-value locking and perturbation sensitivity. Native trajectories refute strict locking: on 600 fixed hard test puzzles, 3088 of 12333 cells corrected at level 9 become incorrect again at levels 10-12 (25.04%; puzzle-bootstrap 95% CI [24.29, 25.83]). Stopping at levels 8, 9, 10 or 11 solves zero grids, versus 516/600 at level 12. Omitting only residual 9 while regenerating the tail solves 103/128 grids, versus 105/128 natively. We provide matched perturbations, fractional readout controls and explicit counterexamples. The evidence supports distributed non-monotone refinement with local acceleration, rather than irreversible level-9 locking or an established architectural phase transition.

## Scope and falsifiable hypotheses

We study published, frozen Sudoku NCM checkpoints using local CPU FP32 inference. This is a new temporal analysis of previously examined datasets and native traces, not an independent blind replication. The pooling-repair study is preserved separately. The methodological distinction is between a necessary continuation, a useful individual contribution, a locally faster improvement, a locked decoded value and an underlying phase transition. None implies all the others.

Our primary observable is the conditional rate of native reversions: a cell is incorrect at 8, correct at 9 and incorrect at least once at 10, 11 or 12. A single such native counterexample contradicts strict decoded-value locking. It need not exclude an unspecified internal commitment whose readout remains imperfect; that weaker claim requires its own observable definition. We do not attribute intention or metacognition to a deterministic change in a decoded digit.

Twelve levels with different lengths and parameters do not supply a system-size family and control parameter for finite-size scaling. A fitted bend at this resolution cannot establish a phase transition. Adding puzzles improves sampling of inputs, not the size of the computational system. We therefore report local acceleration and descriptive segmented fits without identifying them with a fundamental transition.

## Computational mechanism and measurements

The explicit runner implements the recurrence

$$c_k = \operatorname{argmax} G_k(P_k(z_{k-1}),p), \qquad z_k = z_{k-1} + R_k(c_k), \qquad y_k=\operatorname{argmax} D(z_k).$$

Here p is the puzzle prefix, P is level-specific pooling, and R includes embedding, upsampling and the residual transform. The first level uses its dedicated initial query. Earlier emitted VQ codes are never revised on the native path: this commitment holds at every level by construction. In contrast, each new residual can change the decoded digits anywhere in the grid. Append-only code commitment and decoded Sudoku-value locking are distinct observables.

Our primary ratio is the number of cells wrong at 8, correct at 9 and wrong at least once at 10-12, divided by the number wrong at 8 and correct at 9. Its bootstrap unit is the source puzzle. A last observed change within 12 levels describes a finite horizon, not irreversible convergence beyond that horizon.

### Data, runner and uncertainty

We reuse previously verified cumulative decodings for all 2,000 puzzles of each official difficulty and 256 independently generated external hard puzzles. The original hard split reserves 600 test puzzles. The causal subset is the first fixed 128 test puzzles; susceptibility uses its first 64. Easy, medium and hard share solution families, so they are correlated replications. All datasets had been inspected previously, and comparisons remain exploratory.

At stop k, the decoder receives the sum of contributions 1 through k at 89 positions, including the separators. We neither decode only block k nor fill absent fine blocks with the valid nonzero embedding indexed by code 0. We report exact 89-token success alongside the correctness of the 81 Sudoku cells. All 600 test puzzles have at least one wrong Sudoku cell at every stop before 12, excluding separator-only failure as an explanation.

An explicit cached runner reproduces native codes, final logits and accumulated states 8/9 exactly on 32 puzzles. All 128 intervention baselines additionally match stored native codes and outputs; their level-8 readouts match the original traces. Cache clearing and branch state preservation are checked. No weights, codebooks or decoder parameters are trained or modified.

Uncertainty uses 2,000 paired percentile bootstrap draws by puzzle, conditional on these frozen models. Reversion ratios resample numerator and denominator together. Noise seeds are averaged within each puzzle before bootstrap. No multiplicity correction or confirmatory significance claim is made. The protocol was frozen before this campaign but after earlier exploration.

## Native trajectories: improvement without locking

Correct-cell accuracy on hard test rises from 45.31% at 8 to 61.51% at 9, 81.68% at 11 and 98.10% at 12. The respective median numbers of wrong cells are 44, 31, 14 and 0. Exact-grid success remains zero through level 11 and reaches 86.00% at 12. Exactness is a conjunction across 81 cells: a late jump in exact-grid accuracy alone does not localize a discrete decision. Early readouts also use a decoder trained for complete representations, so failure does not establish the absence of information accessible to a different trained reader.

![Native stopping curves on fixed test and external puzzles. Exact-grid accuracy alone hides substantial late cell correction.](figures/01_native_stops.pdf)

Of 12333 test cells corrected at 9, 3088 become incorrect later (25.04%, 95% CI [24.29, 25.83]). Only 55 are incorrect at the final level (0.45%, CI [0.30, 0.62]); transient reversions dominate persistent failures. Across all hard puzzles, the corresponding any-later count is 10684/41620. External puzzles show 1362/5253 reversions (25.93%). Full easy/medium profiles and reversion counts are retained in the French report and JSON.

For source puzzle 938, row 2, column 4, the reference digit is 7; the native values at levels 8-12 are 6, 7, 7, 4, 7. A correction at 9 is lost at 11 and recovered at 12. In total, 50.50% of all test cell values read at 9 change at least once later. None of the 600 grids at 9 equals its final output. Zero solved grids lost after 9 is vacuous here: zero were solved at 9.

![Gross corrections, degradations and net changes, alongside one native counterexample to decoded-value locking.](figures/02_corrections_reversals.pdf)

The historical count 41,620 measures cell events, not puzzles or irreversible decisions. Degradations must be subtracted to obtain net improvement. The net accuracy gain from 8 to 9 exceeds the mean of the adjacent gains 7 to 8 and 9 to 10 by 2.96 percentage points (95% CI [2.30, 3.58]). This supports local acceleration under this observable, not irreversibility. Counts normalized by the added number of codes are also saved.

## Temporal residual interventions

We omit the selected residual contribution while executing its prediction head and regenerating later levels normally. The schedule and other parameters remain fixed. Thus these interventions measure a specific residual removal, not runtime savings from a compressed architecture. The native baseline solves 105/128 puzzles.

- Omit residual 8: 106/128 exact grids, paired difference 0.78 percentage points, 95% CI [-1.56, 3.91].
- Omit residual 9: 103/128 exact grids, paired difference -1.56 percentage points, 95% CI [-3.91, 0.00].
- Omit residual 10: 103/128 exact grids, paired difference -1.56 percentage points, 95% CI [-4.69, 1.56].
- Omit residual 11: 104/128 exact grids, paired difference -0.78 percentage points, 95% CI [-2.34, 0.00].
- Omit residual 12: 0/128 exact grids, paired difference -82.03 percentage points, 95% CI [-88.28, -75.00].

Omitting residual 9 changes 16 final grids and 1,962 codes, establishing causal influence on these outputs. Its success difference is -1.56 percentage points (95% CI [-3.91, 0.00]), without a demonstrated population-level performance advantage. Most performance is retained, showing substantial recovery by the remaining computation. Residual 12 dominates this omission test, without establishing a phase transition at 12 either. Residual norms differ across levels and the altered states can be out of distribution. Effect sizes therefore do not establish a perfectly amplitude-controlled ranking of head importance. A large loss from omission can show a functional contribution without identifying an irreversible decision. Continuing from z8 + gamma R9, with native R9 fixed, separately yields gamma 0.0: 103/128, gamma 0.5: 105/128, gamma 1.0: 105/128, gamma 1.5: 105/128. Gamma 1 exactly matches native codes and outputs; gamma 0 exactly matches omission 9. Gamma 1.5 is a diagnostic extrapolation.

![Final success after residual omission and fractional level-9 amplitude, with the tail regenerated.](figures/04_temporal_interventions.pdf)

## Fractional readout and matched susceptibility

With native level-9 codes fixed, we decode 21 states z8 + gamma R9 without generating further levels. Cell accuracy changes from 45.97% to 61.54% and gold-digit negative log likelihood from 1.3695 to 0.9982 nats. Saved logits and margins allow direct inspection of threshold crossings in the argmax reader. This path is not a native computation with recomputed discrete codes. Smooth readout interpolation does not imply that every generator-head path is smooth, nor that level 9 is functionless.

![Fixed-code interpolation of the decoder input: argmax quality and gold-digit negative log probability.](figures/03_reader_interpolation.pdf)

We also add the identical normalized Gaussian direction with identical absolute norm immediately before or after level 9, then regenerate the remaining tail. Perturbation norm is dose times norm(z8) in both conditions; relative norm against z9 is not matched. The before condition recomputes level 9 while the after condition preserves its native codes. We use two doses, three seeds and 64 puzzles.

- Norm dose 1%: changed final grids before/after 1.04%/0.52%; after-minus-before difference -0.52 percentage points, 95% CI [-1.56, 0.00].
- Norm dose 5%: changed final grids before/after 3.12%/2.60%; after-minus-before difference -0.52 percentage points, 95% CI [-1.56, 0.00].

![Susceptibility of the remaining computation to matched absolute perturbations. Intervals resample puzzles after averaging seeds.](figures/05_susceptibility.pdf)

Native success on these 64 puzzles is 54/64; none of the tested perturbations changes success status. Degenerate bootstrap intervals [0,0] are not evidence of population equivalence. This limited isotropic stress test cannot certify a global attractor basin. Reduced sensitivity would not prove irreversibility, while a changed final output would not alone establish reversal of an internal decision. Native reversions are the direct test of strict decoded-value locking.

## Descriptive bends and limits of the phase-transition claim

A continuous segmented least-squares model selects a knot among levels 3-10 using the same 12 points on which it is fitted. On test, the selected knot is level 6 when the abscissa is level number, and level 10 when it is cumulative generated-code count. Bootstrap preferences describe this conditional fit, not a significance test of a singularity. The flexible model necessarily has an advantage over a single line and is not independently validated.

![Selected descriptive bends depend on the horizontal parameterization of the same accuracy observations.](figures/06_descriptive_bends.pdf)

This campaign supports distributed, non-monotone correction with a local acceleration around level 9. It refutes strict locking of decoded Sudoku values at that level. It neither proves nor universally disproves an internal phase transition under a different operational definition. It does not generalize these temporal interventions to text. Stronger claims would require defined order parameters, controlled model families, repeated training and a reproducible scaling analysis; a weaker latent-commitment hypothesis would require candidate-specific patching and recovery tests with a prespecified criterion.

## Reproducibility

The accompanying French report, frozen protocol, journal and per-puzzle NPZ files document all measurements. Run `./run.sh` in this study directory to reproduce the observations, interventions, figures and verification. The four earlier campaigns are read-only inputs whose manifests are checked again. `sources_temporelles.zip` contains the executed scripts, local model source, configurations and earlier source archives. Large published weights and original traces remain in the parent analysis directory, linked by hashes. Verification results state their actual replay scope; hashes establish artifact identity, not correctness of interpretation.

Primary model source: [NCM article](https://arxiv.org/abs/2610.14162), [official repository](https://github.com/HazyResearch/ncm). Numerical results here are local measurements, not claims copied from these sources.

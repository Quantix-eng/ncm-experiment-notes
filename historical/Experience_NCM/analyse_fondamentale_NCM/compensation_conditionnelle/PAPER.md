---
title: "Apparent Answer Revision in NCM: Codec Dynamics and a Test of Conditional Residual Compensation"
date: "24 October 2026"
geometry: margin=22mm
fontsize: 10pt
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
---

# Abstract

We separate non-monotone progressive-codec readouts from conditional generator compensation in Next-Concept Models. Earlier Sudoku controls reproduce decoded reversions while encoding known correct solutions without generator execution. Here a crossed state/tail intervention removes residual 9, and a separate experiment executes the released LM1B small generator. On 256 newly generated unique Sudoku puzzles, recovery R is -0.001641 nats, with 97.5% paired-bootstrap interval [-0.02636, 0.02538]; specificity I is -0.003803, interval [-0.00725, -0.00095]. The prespecified joint criterion for useful, state-specific compensation is not met. In 128 native LM1B trajectories, 416/2340 positions adopting their own final raw token at level 9 later abandon it (17.78%). This retrospective stability metric does not measure semantic correctness. Results extend the existence of non-monotone readouts beyond Sudoku while distinguishing code adaptation from demonstrated compensation.

# Mechanistic question and oracle evidence

The Sudoku generator predicts codes from cumulative latent representations and puzzle clues. It does not consult its decoded intermediate grid. Earlier codes remain fixed, while each new residual can alter any decoded digit. Append-only codes therefore do not imply locked decoded answers.

On the earlier 600 hard test puzzles, a codec encoding known correct solutions produces 3,176 reversions among 12,717 cells corrected at level 9 (24.97%). Free generation produces 3,088 among 12,333 (25.04%). The oracle reconstructs all 600 final grids exactly, but none at levels 1-11. This establishes codec sufficiency to produce the phenomenon, not exclusive attribution of generator reversions: the selected correction sets differ. The codec decoder itself includes attention. Nested dropout in the tokenizer recipe also rules out simply assuming that only complete representations were supervised; configurations do not recover the full historical training exposure.

# Frozen crossed experiment

After native level 9, let z=z9 and delta=-r9. T0 is the sum of native contributions 10-12; Tdelta is the tail regenerated from z8 with the same puzzle. We evaluate Y00 (native state/native tail), Y10 (perturbed state/frozen native tail), Y11 (perturbed state/regenerated tail), and Y01 (native state/tail adapted to the perturbation).

We directly use z8 rather than subtracting and readding residuals in FP32. Contributions retain their native order; reassociation deviations are recorded. An explicit runner matches native codes, final logits and states 8/9 on 32 puzzles. All 600 official baselines match prior codes and outputs; the first 128 omitted branches match the prior temporal study. Cache cleanup and branch isolation are asserted. Parameters remain frozen.

For per-puzzle mean NLL L on initially empty cells, R=L10-L11 measures recovery from recomputing the tail. I=L10+L01-L11-L00 subtracts the benefit the same adapted tail has on the native state. A better Y11 than Y10 alone need not imply state-specific repair. These are explicit controlled contrasts, not a unique causal mediation decomposition or a measure of intent.

Both contrasts are prespecified co-primary. Four thousand paired puzzle-bootstrap draws provide 95% descriptive intervals and 97.5% intervals with a two-contrast Bonferroni adjustment. The declared compensation criterion requires both adjusted lower bounds positive on fresh puzzles. Coverage is approximate and conditional on frozen weights. A Gaussian readout control matches the norm of adapted-minus-native tail change, but not its geometry or distribution.

# Data and independence

The 600 official hard test and 256 previous external puzzles were examined previously and remain exploratory replications. After protocol freezing, we generate 256 new solutions by randomized backtracking, then remove clues greedily while retaining uniqueness until 30 clues remain. MRV and independent Algorithm X verify each puzzle. Solution keys under D4 and digit normalization are excluded against official, prior external and other new solutions. The audit does not cover every isomorphism or unknown training contamination.

The new corpus is a fixed precision budget, without guaranteed power, and its generator is not uniform. No level, dose, subset or decision threshold is changed based on its model results. Populations are kept separate rather than pooled after examining signs.

# Crossed results

- hard_test, n=600: Y00/Y10/Y11/Y01 solve 516/514/515/517 grids. R=-0.003959 nats, 97.5% CI [-0.02967, 0.02037]; I=-0.004520, CI [-0.00787, -0.00108].
- external_previous, n=256: Y00/Y10/Y11/Y01 solve 219/216/216/220 grids. R=0.041386 nats, 97.5% CI [0.00026, 0.09858]; I=-0.002284, CI [-0.00741, 0.00253].
- fresh, n=256: Y00/Y10/Y11/Y01 solve 222/221/224/224 grids. R=-0.001641 nats, 97.5% CI [-0.02636, 0.02538]; I=-0.003803, CI [-0.00725, -0.00095].

![Crossed negative log likelihood on initially empty cells, averaged by puzzle.](figures/01_crossed_nll.pdf)

![Recovery and specificity with 97.5% paired-bootstrap intervals. Fresh puzzles are the prespecified primary population.](figures/02_contrasts.pdf)

On fresh puzzles the four NLL values Y00/Y10/Y11/Y01 are 0.356166/0.349218/0.350859/0.354004. The prespecified joint criterion for useful, state-specific compensation is not met. A favorable recovery estimate on the previous external set does not demonstrate specificity when the adapted tail also benefits the native state. Exact-grid success is secondary and does not replace the announced loss observable. Overlap with zero does not establish equivalence.

The mean cosine of tail change with the removed residual is 0.0838; mean norm(delta+tail change)/norm(delta) is 1.1450. These assess Euclidean cancellation, not necessarily task-level compensation. Different downstream codes do not alone prove useful repair. We retain branch codes, logits and latent MSE to the encoded reference solution.

![Latent response to omission on fresh puzzles.](figures/03_geometry.pdf)

This tests one complete residual omission, not all possible compensation mechanisms. Crossed states may be out of distribution; a random norm-matched direction is not a fully distribution-matched control. Insensitivity of a frozen tail and adaptation without benefit remain distinct possibilities.

# Native LM1B generation

We execute the published LM1B small generator, with 97,065,600 parameters, strict loading and SHA-256 verification against the official LFS digest at the pinned revision. The already pinned tokenizer is reused. Native eager generation uses 128 independently seeded B=1 trajectories, CLS prefix 101, the checkpoint's saved level-0 unigram, temperature 1, top-k 0, top-p 1 and CFG 0. The level lengths are 1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108 and 128; level 9 is not the same spatial resolution as Sudoku level 9.

We explicitly decode each cumulative representation before mutation, avoiding aliasing of the original history list, and verify the final accumulation against the native method. The first two seeds and all their readouts are replayed exactly. Parameters are memory mapped and deterministic nonpersistent RoPE buffers are rebuilt as in the constructor. Shared-head views are bound to loaded biases using ordinary copying for that component; every registered tensor is checked against the checkpoint. An initial incorrect loader retained stale shared-bias references: its 46 trajectories are excluded and archived separately, and all reported sequences are recomputed. Disk and loading incidents are recorded in the journal, with no changed seeds or sampling parameters.

The main readout is raw decoder argmax at all 128 positions. The official final frequency/adjacency postprocessing is reproduced separately for inspection and disagrees with raw final tokens at 14.37% of positions. We do not mix this heuristic with intermediate curves or evaluate GPT2 perplexity or semantic quality.

Since unconditional generation has no unique target sentence, its own final raw token is a retrospective anchor. Adoption at k means disagreement at k-1 followed by agreement at k; abandonment means later intermediate disagreement. Final agreement is 100% by construction, and abandonment after arrival at levels 11/12 is structurally zero. All transitions are reported. A secondary mask retains final IDs 104-30521, excluding specials and unused vocabulary rows but conditioning on the future. Intervals resample trajectories rather than tokens assumed independent.

![Native text trajectories and abandonment of adopted final tokens.](figures/04_text_generation.pdf)

At arrival 9 the count is 416/2340 (17.78%; 95% trajectory-bootstrap interval in proportions [0.16111, 0.19603]). The final non-special mask gives 410/2274 (18.03%). Counterexamples retain seed, position and token identifiers. This measures non-monotone readout relative to an eventual output, not semantic error correction.

# LM1B oracle codec and scope

As a distinct control, we encode 128 previously selected validation rows and decode their partial accumulations. Non-PAD reference tokens give actual reconstruction targets. The number of corrections at 9 later degraded is 97/1675 (5.79%); final reconstruction accuracy is 99.84%. These rows are not paired references for unconditional generation and remain exploratory. Four-row oracle batches are checked against singleton batches on the first four rows.

![Text oracle reconstruction on reused validation rows.](figures/05_text_oracle.pdf)

The existence of these trajectories outside Sudoku is not a universal architectural law or defect. A final generated token can itself be poor and does not become correct by being final. Architecturally causal generalization would require controlled model families and training comparisons.

# Conclusion and reproducibility

The prespecified joint criterion for useful, state-specific compensation is not met. The defensible result separates codec sufficiency, effective code adaptation and useful compensation. No phase transition, exclusive codec origin, metacognitive monitoring or intent to repair is inferred. Attention patching and disambiguating-puzzle interventions are deliberately deferred.

Limitations include one Sudoku checkpoint for interventions, one text generator size, one omission, one text temperature, nonuniform puzzles and artificial crossed states. Frozen weights and uncertainty intervals do not remove these limits. Run `./run.sh` to reproduce the campaign. Saved NPZ files contain per-case logits, codes, seeds and all readouts; verification states its exact replay scope. The source archive contains scripts, model sources, configurations and protocol, excluding large weights. Six earlier campaign manifests remain unchanged.

Primary sources: [NCM article](https://arxiv.org/abs/2610.14162), [official repository](https://github.com/HazyResearch/ncm), [published checkpoints](https://huggingface.co/hazyresearch/ncm-checkpoints). Numerical results are local measurements.

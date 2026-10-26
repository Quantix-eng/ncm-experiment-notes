---
title: "A Partial Terminal Alphabet and Certified Decoder Bypass in Sudoku NCM"
date: "Experimental report - 24 October 2026"
fontsize: 10pt
geometry: margin=22mm
colorlinks: true
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
  - \setlength{\parskip}{3pt plus 1pt minus 1pt}
---

# Abstract

We report terminal-code interventions and a certified readout policy for frozen Sudoku Next-Concept Models. A favorable development pilot guided selection of the question and a local protocol; this was not an independent preregistration. On 128 reserved puzzles, replacing one terminal code writes a requested different digit in 784/1,024 trials (76.56%), versus 0/1,024 locally norm-matched Gaussian controls; the source-clustered 97.5% confidence interval for the paired advantage is [71.68, 80.96] percentage points. No other cell argmax changes in these substitutions. Gaussian directions do not match codebook geometry, limiting mechanistic interpretation. On 1,024 new unique-solution puzzles, train-only lookup yields 552 certified grids (53.91%; 97.5% CI [50.29, 57.32]). The 30% coverage criterion was chosen after 50.25% development coverage and lacks an independently justified utility threshold. Both announced criteria are met, but that is not a verdict on scientific importance. The fallback guarantee follows from external verification and uniqueness, independently of NCM internals. Native/hybrid success is 883/885 out of 1,024, without an established population accuracy gain. Mean readout latency falls from 1.091 to 0.493 ms in one CPU session; end-to-end acceleration remains uncertain. Novelty, architecture specificity and an explanation of intermediate revisions remain unestablished.

# Question and contribution

Earlier campaigns found non-monotone reconstructions even when the codec directly encoded correct solutions, refuting a strict decoded-value lock at level 9. A crossed omission experiment did not establish useful, perturbation-specific compensation by later levels. These results motivate a different mechanistic question: what information is present in the final discrete codes, and can that information be manipulated and used without changing model weights?

The final Sudoku level has 89 positions: 81 cells and eight separators. Unlike the 37-position ninth level, it is directly aligned with serialized cells. We distinguish three claims: a code predicts a digit observationally; substituting the code controls the digit causally; and a policy exploiting that association has useful coverage without loss of correctness. None follows automatically from another. We test them separately and retain failed terminal-only reconstruction and uncertain full-pipeline timing.

We report a local empirical characterization of a digit association, code substitution and certified readout for these published Sudoku models. A trained terminal codebook might be expected to carry digit information. No comparison establishes that the observed behavior is novel or specific to NCM. It does not explain intermediate revisions, locate a reasoning circuit or establish an architecture-wide law. This critical editorial revision preserves all original data, numerical outcomes and announced rules; prior versions and an integrity audit are retained.

# Development, freeze and reserved data

For each checkpoint, we use only the original 1,000 training-partition puzzles to count pairs of generated terminal code IDs and same-position native decoded digits. For each of 16 IDs, the readout table selects its most frequent digit among 1 through 9. For each digit, the writing alphabet selects the ID with the largest training count for that digit. References are not used to fit either table. Ties use the first ID/digit in numerical order. Each checkpoint has a separate table.

Old validation data guide direction selection. Hard validation cell agreement is 99.38%, with 201/400 certifiable lookup grids. A 32-puzzle writing pilot succeeds in 208/256 trials (81.25%). Decoding the terminal residual alone agrees with native digits on 98.19% of cells but solves 0/32 complete grids. The zero counts all 89 tokens, including separators. A post-hoc format audit finds 10/32 grids with all 81 digits correct but 0/32 with all eight separators correct; reference digit accuracy is 96.18%. This is not evidence that terminal-only input can never reconstruct grid digits. It fails as a reliable standalone readout, with format and digit errors distinguished. The frozen pilot metric and reserved criteria are unchanged.

The local freeze is dated 2026-10-24 21:53:58 UTC. Protocol, tables and pilot are hashed before recorded reserved data generation. The two announced co-primary criteria are mean source-level writing advantage above zero and hard-checkpoint certificate coverage above 30%. The 30% minimum was selected after observing 201/400 (50.25%) validation coverage. It is a favorable development-selected convention, without an independent justification of minimal practical utility. Its satisfaction should not be equated with an important scientific discovery. We use 4,000 paired/source-puzzle bootstrap draws, 97.5% percentile intervals for both criteria with Bonferroni adjustment, and descriptive 95% intervals elsewhere. This adjustment concerns these two criteria, not the earlier exploratory results. The approximate inference is conditional on the fixed models and tables; it does not represent training-seed uncertainty.

## Selection and limits of the local freeze

This direction was chosen during an explicit search for a positive result after earlier hypotheses were not supported. The development pilot already showed substantial writing and coverage. Selecting a promising direction on development data and then testing fixed rules on new data does not itself invalidate the new test. It does require an explicit account of selection, modest interpretation and preservation of failures.

This local record is not an independent preregistration or external review. The experimental scripts were not included in the initial freeze hashes, and there is no external, complete, append-only log of every exploratory command. Current code and recorded rules agree; local logs place the freeze before generation. These artifacts alone cannot establish the entire chronology or certify the absolute absence of p-hacking. Current hashes establish file integrity, not scientific validity. The historical protocol is preserved without retroactive rewriting.

We then generate 1,024 random-backtracking solution grids and greedily remove clues while maintaining uniqueness until 30 remain. MRV backtracking and an independent Algorithm X solver verify each puzzle. Digit-normalized D4 solution keys are excluded against the published corpora, both earlier external corpora and previously accepted new solutions. This is not a complete Sudoku-isomorphism audit, a uniform generator or proof against unknown training contamination. No puzzle is removed on the basis of NCM outputs. All new data are used for hard readout; the first 128 sources are used for writing and the first 256 for secondary easy/medium evaluation.

![Train-only terminal associations. Row labels include support, preventing rare IDs from appearing equally well established. The writing alphabet selects one ID per digit.](figures/01_terminal_alphabet.png){width=92%}

# Terminal editing: association, intervention and control limitations

## Intervention and matched control

Let the final cumulative state be `z12` and the terminal contribution be `R12(c12)`. Choose one initially blank cell uniformly in each reserved source, using the frozen seed and source index independently of prediction and effect. For each requested digit different from the pre-intervention native digit, replace only that cell's terminal ID with the frozen writing ID. Decode `z12 + R12(c12_modified) - R12(c12_native)` with every other code and level fixed. This gives eight eligible requests per source, or 1,024 trials. The sham reproduces native output. No later generator stage remains to be adapted.

The terminal residual transform contains a width-three convolution, so one ID replacement can change up to three full-resolution latent positions. At every affected position, the control uses an independent Gaussian direction scaled to exactly the substitution's local L2 norm, with the same support. We save and verify all norms. This controls local perturbation amplitude and support; it is not a control matched to the distribution of codebook differences. Both sets of mixed states can be off distribution. A numerical audit found FP32 convolution/subtraction tails outside the mathematical three-position support (maximum position norm 6.28e-06). Controls match the actual norms including these tails. A post-hoc direct FP64 affine contraction, cast back to FP32 before decoding, removes the tails and changes none of the complete readouts across all 1,152 requests (including native-digit requests) in either arm. Original frozen interventions and primary counts are retained.

## Reserved result

Substitutions write the requested digit in **784/1,024 trials (76.56%)**, versus **0/1,024 controls**. The paired source-level advantage is 76.56 percentage points, with adjusted 97.5% CI **[71.68, 80.96]**, satisfying the first frozen criterion. Per-digit outcomes are reported without additional confirmatory claims. All 240 unsuccessful writes remain in the archive.

Across eligible substitutions, **zero of the other 80 cells change**. Controls change an average of 0.0703 other cells per request. This is observed selectivity, not universal spatial independence: sources, one-cell selection, doses and IDs are limited. A zero-count bootstrap would be degenerate. Treating each puzzle as the independent source, zero of 128 sources with any collateral change has a descriptive 95% Wilson upper bound of approximately 2.91%. The 1,024 requests are not 1,024 independent sources.

The learned codebook differences and Gaussian directions have different geometry. Beating isotropic noise may reflect a trained decoder reading trained embeddings, without a novel or NCM-specific reasoning mechanism. Valid-code substitutions with permuted labels, other-position controls and a comparator codec would test stronger interpretations; these are not prospective criteria or completed replications in this campaign. Collateral-change counts concern the argmax of the other 80 cells, not invariance of logits or separators.

The lookup agrees with the native hard decoder on **99.47%** of reserved cells, above the 76.56% intervention success. These figures use different position populations and interventions; their difference is not a controlled estimate of context dependence. Unsuccessful writes do establish that the tested ID substitution does not invariably impose its associated digit. A descriptive post-hoc decomposition of the 240 failures finds **237 unchanged cells and three producing a different digit**. It does not localize the dependence to earlier residuals, neighboring codes or particular decoder modules. Editing a valid digit into a different one usually violates Sudoku constraints; this is not an accuracy-enhancing edit.

![Reserved causal writing by requested digit and aggregate contrast. Trial counts vary because requests equal to the native digit are excluded before intervention. Other-cell changes concern Sudoku cells, not a claim about every logit or separator.](figures/02_causal_writing.png){width=100%}

# Certified direct readout

## Policy and conditional guarantee

From generated terminal IDs, the policy constructs 81 lookup digits and fixed separators. It checks that each of the 27 row, column and block units contains digits 1 through 9, every given clue is preserved, and separators are valid. If all checks pass, it returns the candidate without calling the decoder. Otherwise it invokes the unchanged native decoder on the identical singleton latent state. The policy uses no reference, uniqueness solver or NCM retraining. A singleton baseline and fallback avoid floating-point changes caused by regrouping decoder batches; batched native agreement is also recorded.

**Proposition.** For a unique-solution puzzle with solution s, a sound certificate and exact native fallback, replacing the native output by a certified candidate cannot reduce exact-reference success on that puzzle.

**Proof.** Any accepted candidate satisfies the puzzle and hence equals s by uniqueness. When the candidate is rejected, the hybrid output is exactly the native output. In particular, whenever native output equals s, either branch of the hybrid policy also equals s. This argument does not require an accurate lookup table. For multiple-solution puzzles, it guarantees validity with clues, not agreement with an arbitrary selected reference. Uniqueness is a data assumption here, not established at policy runtime.

This is a standard verification-and-fallback argument, not a new principle of certification. It would apply to another candidate generator under the same conditions and is not evidence about NCM internals. Lookup-only accuracy could be poor and the argument would still hold, with lower coverage. On rejected inputs, native fallback can still produce incorrect answers.

## Reserved coverage and secondary quality

On hard, **552/1,024 candidates pass (53.91%)**, adjusted 97.5% CI **[50.29, 57.32]**, exceeding the frozen 30% criterion. Every accepted candidate equals its reference. The policy solves **885/1,024**, versus **883/1,024** natively, with zero correct-to-incorrect changes and two repairs. The descriptive exact-success difference is 0.195 percentage points, 95% CI [0.00, 0.49]; it does not establish a strictly positive population accuracy effect. Lookup-only reading solves 552/1,024, so the fallback is material.

Separate easy and medium tables certify **66/256 (25.78%)** and **99/256 (38.67%)**, respectively. Native/hybrid successes are 106/107 and 188/189, with zero regressions. These are secondary checkpoint replications on the same new 30-clue families, not independent data replications or evaluations on the official easy/medium distributions. No writing intervention is run on those checkpoints.

![Certified bypass and exact success. Hard uses 1,024 new puzzles; easy and medium share its first 256. Coverage error bars are descriptive 95% intervals; the hard co-primary interval is given in the text.](figures/03_certified_readout.png){width=100%}

# Timing: saved decoder calls versus full computation

All timings use CPU FP32, one thread, and the first 64 new puzzles, of which 59.38% are certified. Four warm-up inputs per arm are excluded. Full-pipeline timing has three paired repeats per puzzle, randomized puzzle order and randomized AB/BA arm order. Both arms run the same native cached generator through all twelve levels; only readout differs. Outputs are checked. Separate readout timing uses sixteen internal repetitions per arm and cached states. No other campaign inference runs during measurement.

Full native/hybrid arithmetic mean latencies are **254.367/249.609 ms**. The geometric native-to-hybrid ratio is **1.0147**, 95% CI **[0.9939, 1.0374]**. Because this interval includes one, **end-to-end acceleration is not established**.

Readout-only arithmetic means are **1.091/0.493 ms**, corresponding to a ratio of **2.21** and a descriptive reduction of **54.80%** for this step and sample. No arithmetic-mean difference interval was announced or estimated. This does not imply the same reduction for the full model.

The announced geometric readout ratio, retained as a complementary statistic, is 9.2117, 95% CI [5.8356, 14.4797]. It summarizes paired log ratios and is not the quotient of arithmetic means. Its magnitude reflects very fast accepted inputs alongside somewhat slower fallback inputs. It must not be read as a ninefold reduction in average readout cost or a ninefold acceleration of the complete model.

Intervals bootstrap source puzzles after averaging repeated log ratios. Raw durations and orders are retained. Uncertainty is conditional on this session, CPU and implementation, without between-session or between-machine replication. We make no GPU, throughput, energy or memory claim.

![Observed arithmetic mean latencies for the full pipeline and readout alone, with different y-axis scales. No arithmetic-mean confidence intervals are added; the announced paired analysis and its uncertainty are described in the text.](figures/04_timing.png){width=100%}

# Relation to prior work and interpretation

NCM [1] generates hierarchical discrete codes with a learned codec. The terminal digit association and controlled substitution characterize these checkpoints; their novelty should not be inflated into a claim that residual quantization or explicit output representations are new. Our earlier failures to establish strict decoded locking or omission-specific compensation remain unchanged. The present result concerns the final representation and its readout, not a replacement explanation for all intermediate oscillations.

Selective prediction with a reject option predates this work, including SelectiveNet [2]. Adaptive language-model computation, including CALM [3], also predates this work. Our policy retains every generator level and uses a task-specific exact external checker; it does not learn an abstention head or establish an earlier internal stopping rule. The empirical contribution is the causal alphabet characterization, its train-only implementation and the measured certified coverage for frozen Sudoku NCM.

Terminal IDs are associated with cell values and can often be used to edit the selected cell under this protocol. The novelty and scientific importance of that behavior remain open. A full-resolution codebook trained to reconstruct digit sequences could show this association without a new mechanism; a controlled codec comparison is needed. The terminal-only pilot includes both digit and separator failures and must not be reduced to an inability ever to reconstruct all 81 digits. A verified lookup has measured coverage because soundness is supplied by puzzle constraints, independently of any claim of perfect code semantics. None of this explains the earlier intermediate oscillations.

# Limitations and reproducibility

The causal study covers one hard checkpoint, 128 sources and one selected blank cell per source. Single-ID writes do not establish multi-cell compositional control, separator control, arbitrary perturbation locality or a human-concept basis. The Gaussian control matches local norm and support but not codebook geometry. There is no text replication of this mechanism. Three separately trained Sudoku checkpoints do not establish an architecture-wide law. New puzzles share one procedural generator and seed, with incomplete isomorphism and contamination audits. The conditional guarantee needs a sound checker, unique solution and unchanged fallback. The empirical absence of regressions alone would not prove that guarantee outside its assumptions.

The full experiment, frozen protocol and tables, all readouts and intervention predictions, norm arrays, source-level metrics and raw timing pairs are retained. A separate verification implementation recomputes counts and intervals, rechecks the 1,024 puzzles with two solvers, tests the certificate against invalid inputs and a set-based implementation, and replays sixteen fresh native trajectories and all nine writing targets with their random controls. Hooks verify zero decoder calls on accepted policy inputs and one on fallback. Seven prior manifest hashes are checked without modifying those campaigns.

Verification counts are software and data assertions: **4,096 check four properties for each of 1,024 puzzles**. These are not thousands of scientific replications. The verifier was written and run by the same research agent, so it is not an independent review and passing it does not establish absence of bias or p-hacking. An independent replication with code and protocol sealed in advance, stronger code controls and a comparator codec remains necessary for stronger claims; the current test data have now been examined and cannot serve as a new untouched test.

The prior publication is preserved in `sources/avant_revision_editoriale_20261024.zip`. An editorial baseline and audit pin the unchanged scientific files and record this change in interpretation. No original data, numerical outcomes, sampling seeds or decision criteria are changed. A clear effect under this protocol remains an empirical observation; novelty, general mechanism and end-to-end benefit remain unestablished.

Run `./run.sh` here to rebuild the documents and verify retained evidence without replacing historical measurements. For a full computational rerun in a new sibling folder, use the parent virtual environment to execute `scripts/reproduce.py --destination ../reproduction_alphabet`. It uses the preserved historical sources and same inputs; this is not independent confirmation, and its regenerated documents carry the earlier wording with an explicit notice directing readers to the current revised interpretation. Timing may vary. `sources_recherche_positive.zip` contains scripts, imported local model sources, configurations, split/protocol files and environment records; large weights and traces remain in the parent study or local result folders with hashes. `results/verification.json` records checks actually passed, and the final manifest pins deliverables. Scientific outcomes, including all failures and uncertain timing, are retained.

# References

1. *Next-Concept Models*, published article and implementation: [arXiv:2610.14162](https://arxiv.org/html/2610.14162v1), [official repository](https://github.com/HazyResearch/ncm). Local frozen copies and model-input hashes are inherited from the audited original study.
2. Geifman and El-Yaniv. *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*. ICML 2019. [Primary paper](https://arxiv.org/abs/1901.09192).
3. Schuster et al. *Confident Adaptive Language Modeling*. NeurIPS 2022. [Primary paper](https://arxiv.org/abs/2207.07061).

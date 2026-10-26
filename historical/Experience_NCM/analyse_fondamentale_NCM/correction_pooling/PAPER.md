---
title: "Architectural Invariance and Order-Sensitive Pooling in Next-Concept Models"
date: "24 October 2026"
geometry: margin=0.8in
fontsize: 10pt
---

# Abstract

An empirically constant quantization level in a structured task does not by itself establish an architectural failure across domains. We audit published Next-Concept Model (NCM) checkpoints, connecting a Sudoku representation constraint to controlled tests of two text tokenizers. In an additive encoder, fixed token histograms within pooling windows yield identical pooled embeddings before normalization and residual subtraction. The Sudoku serialization aligns the nine-position scale with complete rows and their fixed digit histograms. This explains a structural source of low variation, but does not mathematically force residual codebook collapse. On LM1B and TinyStories, we audit all 12 levels using 256 randomly selected validation contexts and a second stratum of 128 long contexts per tokenizer. Neither exhibits an empirically constant level in either stratum. Block permutations preserve local raw embedding averages while changing native tokenizer codes, separating an isolated pooling invariance from end-to-end representational collapse. A validation-only screen of Sudoku bypasses selects levels 5, 2, 11. Retaining their modal code contributions reduces predicted levels from 12 to 9 and solves 510/600 test puzzles versus 516/600 natively: -1.00 percentage points, paired 95% CI [-2.17, 0.17]. These results support a data/architecture interaction and checkpoint-specific bypass opportunities; they do not establish a universal NCM defect, exact equivalence of the nine-pass variant, or a measured speedup.
 We additionally test a parameter-free position-weighted pooling repair at fixed output size. The validation-selected weighted_1 variant reconstructs 600/600 held-out hard solutions exactly, versus 600/600 natively, while using 15 level-5 codes. Its encoder reconstruction and latent-noise results are reported separately from the frozen generator, which never calls the modified operation during tested inference.

# Introduction and scope

NCM uses a multiscale residual vector-quantized tokenizer and a generator that predicts one code level at a time [1]. Our starting observation is narrow: level 5 of the published hard-Sudoku tokenizer selects one code on the official solutions examined in our earlier campaign. We ask whether this is a Sudoku-specific interaction, whether comparable constant levels appear in text, and whether further generator predictions can be omitted without retraining.

Three distinct claims require different evidence. A pooling operator can lose order information by construction. A trained quantizer can select a constant code sequence on a particular corpus. A generator prediction can sometimes be replaced by a constant template with little or no change to final answers. None of these statements alone implies either of the others. We keep their measurements and interventions separate.

The contribution is a checkpoint audit, not a new training algorithm or a claim that all hierarchical representations fail. The average-pooling identity is elementary; the mechanistic contribution is its alignment with the exact Sudoku serialization, the residual and dropout controls, and the empirical limits on cross-domain generalization. Previous campaigns and their sources are preserved alongside this extension.

# A conditional invariance result

Consider an additive encoder before contextual processing and normalization,
$E(x)_i=e(x_i)+p_i$, where $p_i$ is fixed by position. Let $B_j$ be a pooling window, and let $n_{jv}(x)$ count token $v$ inside that window. Then

$$
P_jE(x)=\frac{1}{|B_j|}\sum_v n_{jv}(x)e(v)+\frac{1}{|B_j|}\sum_{i\in B_j}p_i.
$$

**Proposition.** If two sequences have the same token histogram in every pooling window, their pooled additive embeddings are identical in real arithmetic.

**Proof.** Every token-embedding sum depends only on the corresponding counts $n_{jv}$. Every positional-embedding sum is fixed. Both terms are therefore equal for each window. Finite-precision implementations can introduce rounding differences.

This result concerns the specified representation and operator. With positional embeddings, GroupNorm before pooling can reintroduce differences through its statistics. Learned, position-sensitive downsampling can break the averaging invariance. Most importantly, a residual quantizer receives $E_{norm}(x)-\hat z_{<k}(x)$; the previous contributions can vary even when a raw pooled embedding does not. The proposition therefore does not force a single residual code, imply a reconstruction failure, or identify the history of codebook optimization.

## Sudoku specialization and prior controls

The Sudoku sequence has 89 tokens: nine rows of nine digits and eight separators. The nine adaptive pooling windows use $\lfloor j89/9\rfloor:\lceil(j+1)89/9\rceil$. Each contains one complete row and one or two separators. Every valid row contains each digit once. Thus every window has an invariant token histogram across solved boards.

On 256 hard training solutions, raw pooled-embedding variance is 3.3e-16 at nine positions, compared with 0.0465 and 0.0593 at eight and ten positions. The residual entering level 5 remains variable: its inter-puzzle variance is 0.001828. Our earlier train-only 16-center k-means probe reduces held-out quantization MSE by 54.84%; this is not a demonstrated improvement to Sudoku solving.

Previous frozen-weight controls also activate additional level-5 codes with encoder dropout, without EMA updates. Suppressing the residual contribution can change final grids, while forcing its native constant code and retaining that contribution reproduces answers exactly on 856 tested puzzles. These observations rule out the shortcut explanation that a constant level must be a zero or globally useless component. Full results are in the preserved mechanistic report.

# Experimental protocol

## Published assets and text data

We use the published LM1B tokenizer, its small-generator configuration, and an independently trained TinyStories tokenizer. LM1B small, medium, and large share one tokenizer, so they are not three independent tokenizer replications. Generator text weights are not loaded and text generation quality is not evaluated. Exact Hugging Face commits, URLs, sizes and SHA-256 hashes are recorded. Existing local TinyStories weights are copied into the study and checked against the official LFS digest.

For each task, the official tokenized validation rows provide 128-token contexts by truncating longer rows and right-padding shorter ones. LM1B row boundaries come from the published offsets; TinyStories rows come from the published plain-data pickle. We draw 256 rows without replacement and separately draw up to 128 with at least 96 non-PAD tokens, using seed 84117. The second stratum can overlap the first and is descriptive, not an independent replication. Reconstruction includes all non-PAD tokens, including other special tokens. Controls use the first 32 selected long contexts. Permuted sequences are diagnostic inputs, not evidence of natural-language quality.

**LM1B.** Validation rows: 71,825; random sample padding 0.00%; long-stratum padding 0.00%; overlap 0 rows. The encoder has 0 contextual blocks and 0 convolutional blocks; positional embeddings: False; learned first-level downsampling: True, cascaded: False.

**TINYSTORIES.** Validation rows: 27,544; random sample padding 0.14%; long-stratum padding 0.28%; overlap 1 rows. The encoder has 0 contextual blocks and 0 convolutional blocks; positional embeddings: True; learned first-level downsampling: True, cascaded: True.

Both corpora mostly provide full contexts, so the long stratum offers little length contrast; it is not a strong padding stress test. No selected random context is duplicated within either task's sample.

## Measurements and fidelity

For each level we measure empirical code counts, marginal entropy, entropy conditioned on position, perplexity, unique code sequences, and fixed-position inter-context variance before and after native downsampling. A level is empirically constant if its complete code sequence is identical across all sampled contexts; unused entries in a large codebook do not meet this definition. We also retain residual MSE before/after each contribution and nearest-code MSE.

Explicit residual accumulation is compared with the untouched official encode path on eight examples. On a small first-batch sample, direct FP64 differences to codebook vectors check nearest assignments independently of the expanded-distance formula. We hash model tensors before and after the audit. All inference is CPU FP32 with one compute thread; the LM1B run uses batch 1 after a documented memory amendment, while TinyStories uses batch 4. These controls establish the tested computations, not universal numerical stability.

## Sudoku screen and statistical units

For all twelve hard-Sudoku levels, templates are positional modes of generated codes from the 1,000 training puzzles. On the first 96 validation puzzles, we separately fix each level to its template while retaining its contribution and regenerate subsequent levels. We also suppress each residual contribution while retaining its prediction. Fixed-template candidates are ranked by changed success status, then changed final grids, then level index. The best three are frozen before the new test interventions.

The selected levels are then fixed separately, in a pair, and jointly on 600 test and 256 external puzzles. Every level and position retains its original identity; fewer Transformer predictions are executed. Paired percentile intervals use 2,000 puzzle-level resamples, seed 84117. Wilson intervals describe discordance rates. Bootstrap intervals are conditional on the templates and validation-selected intervention; zero observed discordances yield a degenerate bootstrap and do not prove population equivalence. No equivalence margin or speed benchmark is specified. Earlier studies already inspected the Sudoku test and external data, so this extension remains exploratory. Cross-difficulty families are shared and are not independent dataset replications.

# Cross-domain tokenizer results

![Empirical code perplexity across hierarchy levels. Text has 256 random and 128 long contexts per tokenizer. Codebook sizes and sampling ceilings differ between tasks; these curves are not comparable utilization percentages.](figures/01_cross_domain_diversity.pdf)

**LM1B, random (n=256).** Constant levels: none. Across levels, marginal perplexity ranges from 226.02 to 2717.00; observed distinct codes from 233 to 3925; minimum position-conditioned entropy 5.1828 nats. Non-PAD reconstruction accuracy is 99.86%, over 32,768 tokens. These are frozen-tokenizer reconstruction results, not text-generation scores.

**LM1B, long (n=128).** Constant levels: none. Across levels, marginal perplexity ranges from 123.91 to 1892.67; observed distinct codes from 125 to 2855; minimum position-conditioned entropy 4.6479 nats. Non-PAD reconstruction accuracy is 99.81%, over 16,384 tokens. These are frozen-tokenizer reconstruction results, not text-generation scores.

**TINYSTORIES, random (n=256).** Constant levels: none. Across levels, marginal perplexity ranges from 217.98 to 1844.50; observed distinct codes from 230 to 3233; minimum position-conditioned entropy 5.0711 nats. Non-PAD reconstruction accuracy is 99.98%, over 32,723 tokens. These are frozen-tokenizer reconstruction results, not text-generation scores.

**TINYSTORIES, long (n=128).** Constant levels: none. Across levels, marginal perplexity ranges from 114.39 to 1654.78; observed distinct codes from 118 to 2813; minimum position-conditioned entropy 4.5349 nats. Non-PAD reconstruction accuracy is 99.98%, over 16,338 tokens. These are frozen-tokenizer reconstruction results, not text-generation scores.

The Sudoku constant-level observation is therefore not reproduced in these two text tokenizers. This is a counterexample to inevitability, not evidence that all text checkpoints are free of underutilization, redundancy, dimensional collapse or generation defects. A small sample cannot establish that an unobserved code is permanently dead, and varied codes need not carry unique causally useful information.

## Isolated pooling invariance versus native codes

Within each four-token block, we randomly permute tokens. At 32 positions, raw additive embedding averages should remain the same under the proposition. We compare this with the actual first learned downsampler and with every native code level. A global token permutation is a control that also changes local histograms.

**LM1B.** Relative L2 difference of raw 32-position averages: 5.01e-08; after pre-quant normalization: 6.47e-08; actual first-level quantizer input: 1.23. First-level native code disagreement: 100.00%. Across all levels, disagreement ranges from 8.01% to 100.00%.

**TINYSTORIES.** Relative L2 difference of raw 32-position averages: 5.91e-08; after pre-quant normalization: 0.000838; actual first-level quantizer input: 1.18. First-level native code disagreement: 100.00%. Across all levels, disagreement ranges from 12.70% to 100.00%.

![Code disagreement after synthetic permutations on 32 paired long contexts per text tokenizer. Raw pooling can be invariant while the learned tokenizer remains order-sensitive.](figures/02_text_permutations.pdf)

The observed invariance of raw local averages does not survive as a constant full code representation. Learned first-level sampling and the resulting residual history supply a concrete route for order information to reach later levels. We do not identify their separate causal shares by retraining or ablating all alternatives.

# Auditing and bypassing other Sudoku levels

Only level 5 is empirically constant on hard training codes. Level 2 is highly concentrated but variable. Other levels can have diverse codes while their individual fixed-template interventions are partly compensated downstream. Code diversity and practical bypassability must therefore be measured separately.

![All-level validation screen. Fixed preserves each template's residual contribution; zero removes that contribution. The two interventions answer different questions.](figures/03_all_levels_screen.pdf)

**Level 2.** Train modal-position agreement 92.00%, conditional entropy 0.2787 nats; on validation, fixing changes 1/96 final grids and 0 success statuses; suppression changes 3/96 grids and 0 statuses.

**Level 4.** Train modal-position agreement 9.45%, conditional entropy 2.6269 nats; on validation, fixing changes 11/96 final grids and 3 success statuses; suppression changes 10/96 grids and 3 statuses.

**Level 5.** Train modal-position agreement 100.00%, conditional entropy 0.0000 nats; on validation, fixing changes 0/96 final grids and 0 success statuses; suppression changes 0/96 grids and 0 statuses.

**Level 6.** Train modal-position agreement 14.85%, conditional entropy 2.4162 nats; on validation, fixing changes 14/96 final grids and 3 success statuses; suppression changes 10/96 grids and 4 statuses.

**Level 11.** Train modal-position agreement 14.01%, conditional entropy 2.4485 nats; on validation, fixing changes 11/96 final grids and 0 success statuses; suppression changes 7/96 grids and 0 statuses.

The frozen ranking selects levels 5, 2, 11. Selection is based on validation behavior rather than an assumption that the smallest entropy levels are always the most removable.

## Test evaluation (native 516/600)

**single_5**, fixed levels [5], 11 predictions: 516/600 solved, 0 final grids changed, 0 errors introduced and 0 rescued. Paired solved-rate difference 0.00 percentage points, 95% CI [0.00, 0.00]; upper Wilson bound on status discordance 0.64%.

**single_2**, fixed levels [2], 11 predictions: 515/600 solved, 5 final grids changed, 1 errors introduced and 0 rescued. Paired solved-rate difference -0.17 percentage points, 95% CI [-0.50, 0.00]; upper Wilson bound on status discordance 0.94%.

**single_11**, fixed levels [11], 11 predictions: 511/600 solved, 71 final grids changed, 8 errors introduced and 3 rescued. Paired solved-rate difference -0.83 percentage points, 95% CI [-2.00, 0.17]; upper Wilson bound on status discordance 3.25%.

**joint_two**, fixed levels [5, 2], 10 predictions: 515/600 solved, 5 final grids changed, 1 errors introduced and 0 rescued. Paired solved-rate difference -0.17 percentage points, 95% CI [-0.50, 0.00]; upper Wilson bound on status discordance 0.94%.

**joint_three**, fixed levels [5, 2, 11], 9 predictions: 510/600 solved, 72 final grids changed, 9 errors introduced and 3 rescued. Paired solved-rate difference -1.00 percentage points, 95% CI [-2.17, 0.17]; upper Wilson bound on status discordance 3.46%.

## External evaluation (native 219/256)

**single_5**, fixed levels [5], 11 predictions: 219/256 solved, 0 final grids changed, 0 errors introduced and 0 rescued. Paired solved-rate difference 0.00 percentage points, 95% CI [0.00, 0.00]; upper Wilson bound on status discordance 1.48%.

**single_2**, fixed levels [2], 11 predictions: 219/256 solved, 1 final grids changed, 0 errors introduced and 0 rescued. Paired solved-rate difference 0.00 percentage points, 95% CI [0.00, 0.00]; upper Wilson bound on status discordance 1.48%.

**single_11**, fixed levels [11], 11 predictions: 219/256 solved, 26 final grids changed, 3 errors introduced and 3 rescued. Paired solved-rate difference 0.00 percentage points, 95% CI [-1.95, 1.56]; upper Wilson bound on status discordance 5.02%.

**joint_two**, fixed levels [5, 2], 10 predictions: 219/256 solved, 1 final grids changed, 0 errors introduced and 0 rescued. Paired solved-rate difference 0.00 percentage points, 95% CI [0.00, 0.00]; upper Wilson bound on status discordance 1.48%.

**joint_three**, fixed levels [5, 2, 11], 9 predictions: 219/256 solved, 26 final grids changed, 3 errors introduced and 3 rescued. Paired solved-rate difference 0.00 percentage points, 95% CI [-1.95, 1.56]; upper Wilson bound on status discordance 5.02%.

![Paired performance changes for validation-selected bypasses. Confidence intervals crossing zero do not establish equivalence.](figures/04_bypass_test.pdf)

Replacing level 5 alone is exactly faithful on the evaluated puzzles. The ten- and nine-prediction variants are practical tradeoffs rather than exact simplifications of the native function. The joint result need not equal the sum of individual effects. We make no runtime claim: levels have different lengths, and prefix encoding, cache management and decoding contribute to cost.

# A fixed-size, order-sensitive pooling repair without retraining

## Construction and what it corrects

We replace only the level-5 downsampling operator, preserving nine output positions, the original codebooks, learned upsampling, decoder and all other levels. For each original adaptive window $B_j$, let $t_{ji}$ vary linearly from -1 to +1 and define

$$
u_j^{(\alpha)}=\frac{1}{|B_j|}\sum_{i\in B_j}(1+\alpha t_{ji})r_i
=u_j^{(0)}+\frac{\alpha}{|B_j|}\sum_{i\in B_j}t_{ji}r_i.
$$

The weights sum to one. They are nonnegative for $0\leq\alpha\leq1$; the diagnostic variant $\alpha=2$ permits negative weights. There are no learned parameters or new dimensions. For the additive raw embedding representation, swapping tokens $a,b$ at positions $i,h$ changes the weighted output by

$$
\frac{\alpha(t_{ji}-t_{jh})}{|B_j|}\big[e(b)-e(a)\big].
$$

If the positional weights and token embeddings differ, this breaks the particular within-window permutation invariance, even when histograms are identical. It does not make the map injective, establish semantic sufficiency, or prove a globally better tokenizer. The operation also differs from appending an identical separator: separators already exist in the Sudoku serialization, and an additional fixed token does not help if every window retains identical token counts.

A deterministic input-independent bias is a negative control: $\mathrm{Var}_x[u(x)+b]=\mathrm{Var}_x[u(x)]$ at fixed position. Such a bias may move nearest-code boundaries and alter observed assignments without adding order information. Pure token-embedding max-pooling is itself permutation invariant; positional embeddings and residuals can change its behavior. Neither higher variance nor higher occupancy alone is a valid quality criterion.

## Finite screen and frozen selection

Before experiments, we specify eleven variants: native; a zero-strength weighted sham; weighted strengths 0.05, 0.25, 1 and 2; right-shift mixtures 0.25 and 1, with right-edge replication; mean/max mixtures 0.1 and 1; and a fixed pseudo-random bias with RMS 0.25. All changes are applied at level 5 only. There is no training or codebook refitting.

The first 128 hard validation solutions screen the variants. Candidates must introduce no exact reconstruction failures relative to native and increase position-conditioned level-5 code entropy. We maximize that entropy, break ties by the smaller RMS input change, then lexical name. The sham and bias are excluded from selection. The selected variant is frozen before the new test evaluations. This is a further exploratory extension of an already inspected Sudoku study, not a new untouched confirmatory experiment.

**native:** 128/128 exact reconstructions; raw pooled variance 3.3335e-16; residual-input variance 0.0018686; 1/16 codes, positional entropy 0.0000 nats; 0 level-5 codes and 0 total codes changed.

**weighted_0:** 128/128 exact reconstructions; raw pooled variance 3.3335e-16; residual-input variance 0.0018686; 1/16 codes, positional entropy 0.0000 nats; 0 level-5 codes and 0 total codes changed.

**weighted_0.05:** 128/128 exact reconstructions; raw pooled variance 0.00010167; residual-input variance 0.0019452; 1/16 codes, positional entropy 0.0000 nats; 0 level-5 codes and 0 total codes changed.

**weighted_0.25:** 128/128 exact reconstructions; raw pooled variance 0.0025417; residual-input variance 0.0035106; 6/16 codes, positional entropy 0.0452 nats; 10 level-5 codes and 92 total codes changed.

**weighted_1:** 128/128 exact reconstructions; raw pooled variance 0.040667; residual-input variance 0.027325; 16/16 codes, positional entropy 2.5498 nats; 1132 level-5 codes and 6994 total codes changed.

**weighted_2:** 128/128 exact reconstructions; raw pooled variance 0.16267; residual-input variance 0.10315; 15/16 codes, positional entropy 2.5468 nats; 1152 level-5 codes and 7151 total codes changed.

**shift_0.25:** 128/128 exact reconstructions; raw pooled variance 0.0010301; residual-input variance 0.0024044; 2/16 codes, positional entropy 0.0051 nats; 1 level-5 codes and 14 total codes changed.

**shift_1:** 128/128 exact reconstructions; raw pooled variance 0.016482; residual-input variance 0.011745; 15/16 codes, positional entropy 2.4333 nats; 867 level-5 codes and 6247 total codes changed.

**max_0.1:** 128/128 exact reconstructions; raw pooled variance 3.7155e-05; residual-input variance 0.0019415; 1/16 codes, positional entropy 0.0000 nats; 0 level-5 codes and 0 total codes changed.

**max_1:** 128/128 exact reconstructions; raw pooled variance 0.0037155; residual-input variance 0.0094784; 5/16 codes, positional entropy 0.2232 nats; 1152 level-5 codes and 8491 total codes changed.

**fixed_bias_0.25:** 128/128 exact reconstructions; raw pooled variance 3.7716e-16; residual-input variance 0.0018686; 3/16 codes, positional entropy 0.0140 nats; 3 level-5 codes and 17 total codes changed.

The frozen choice is **weighted_1**. It passes the validation preservation criterion. The zero-strength sham reproduces all native codes and reconstructions. In the paired raw-embedding swap control, relative L2 change is 2.21e-08 for native averaging and 0.0515 for the selected operator; this control uses synthetic within-row swaps, not valid new Sudoku tasks.

![Validation screen separates variance recovery from discrete code entropy. The fixed bias is an input-independent negative control.](figures/05_repair_screen.pdf)

## Held-out reconstruction and checkpoint replication

Known solutions, rather than puzzle clues, are encoded. Reconstruction is exact over all 89 tokens including separators. We evaluate native, the selected operator, weighted strength 1 and max strength 1 on 600 hard test and 256 external solutions, deduplicating identical variants. Native and the selected operator are also tested on 600 easy and medium test solutions. The checkpoints have different weights but shared solution families, so these are correlated replications. Bootstrap intervals are paired by solution, conditional on the fixed operator; zero observed discordances give a degenerate interval and do not establish population equivalence.

**hard_test, native:** 600/600 exact; 0 token errors; 1/16 level-5 codes; positional entropy 0.0000 nats. Versus native: 0 failures introduced, 0 repaired; exact-rate difference 0.00 percentage points, paired 95% CI [0.00, 0.00]; 0 level-5 codes changed.

**hard_test, weighted_1:** 600/600 exact; 0 token errors; 15/16 level-5 codes; positional entropy 2.5887 nats. Versus native: 0 failures introduced, 0 repaired; exact-rate difference 0.00 percentage points, paired 95% CI [0.00, 0.00]; 5311 level-5 codes changed.

**hard_test, max_1:** 600/600 exact; 0 token errors; 6/16 level-5 codes; positional entropy 0.1755 nats. Versus native: 0 failures introduced, 0 repaired; exact-rate difference 0.00 percentage points, paired 95% CI [0.00, 0.00]; 5398 level-5 codes changed.

**external, native:** 256/256 exact; 0 token errors; 1/16 level-5 codes; positional entropy 0.0000 nats. Versus native: 0 failures introduced, 0 repaired; exact-rate difference 0.00 percentage points, paired 95% CI [0.00, 0.00]; 0 level-5 codes changed.

**external, weighted_1:** 256/256 exact; 0 token errors; 15/16 level-5 codes; positional entropy 2.5773 nats. Versus native: 0 failures introduced, 0 repaired; exact-rate difference 0.00 percentage points, paired 95% CI [0.00, 0.00]; 2262 level-5 codes changed.

**external, max_1:** 256/256 exact; 0 token errors; 5/16 level-5 codes; positional entropy 0.1946 nats. Versus native: 0 failures introduced, 0 repaired; exact-rate difference 0.00 percentage points, paired 95% CI [0.00, 0.00]; 2304 level-5 codes changed.

**easy_test, native:** 600/600 exact; 0 token errors; 2/16 level-5 codes; positional entropy 0.5372 nats. Versus native: 0 failures introduced, 0 repaired; exact-rate difference 0.00 percentage points, paired 95% CI [0.00, 0.00]; 0 level-5 codes changed.

**easy_test, weighted_1:** 600/600 exact; 0 token errors; 15/16 level-5 codes; positional entropy 2.5555 nats. Versus native: 0 failures introduced, 0 repaired; exact-rate difference 0.00 percentage points, paired 95% CI [0.00, 0.00]; 5284 level-5 codes changed.

**medium_test, native:** 599/600 exact; 1 token errors; 3/16 level-5 codes; positional entropy 0.0073 nats. Versus native: 0 failures introduced, 0 repaired; exact-rate difference 0.00 percentage points, paired 95% CI [0.00, 0.00]; 0 level-5 codes changed.

**medium_test, weighted_1:** 599/600 exact; 1 token errors; 15/16 level-5 codes; positional entropy 2.5369 nats. Versus native: 0 failures introduced, 0 repaired; exact-rate difference 0.00 percentage points, paired 95% CI [0.00, 0.00]; 5331 level-5 codes changed.

![Level-5 code distributions on hard test solutions. Changed encoder targets do not imply that the frozen generator learned to predict them.](figures/06_repair_codes.pdf)

## Robustness: valid transformations and latent corruption

On the first 128 hard test solutions, a fixed digit permutation and transposition provide valid transformed grids. Native and selected encoders reconstruct each transformed reference independently. These tests do not evaluate solving under transformed clues.

**digits, native:** 128/128 exact, 0 failures introduced and 0 repaired versus native.

**digits, weighted_1:** 128/128 exact, 0 failures introduced and 0 repaired versus native.

**transpose, native:** 128/128 exact, 0 failures introduced and 0 repaired versus native.

**transpose, weighted_1:** 128/128 exact, 0 failures introduced and 0 repaired versus native.

We also add Gaussian noise to the full pre-quant encoder state before residual quantization, with SD 0.05, 0.2 or 0.5 times the RMS measured only on 256 train solutions. Identical noise draws are used between operators for each source and seed. Five fixed seeds are averaged within each puzzle; 2,000 bootstrap draws resample puzzles. This is a codec latent-noise stress test, not Sudoku difficulty, input corruption robustness, or an end-user generation benchmark. The three noise contrasts are exploratory and not corrected for multiplicity.

**Noise 0.05:** native exact reconstruction 100.00%, weighted_1 100.00%; paired difference 0.00 percentage points, 95% CI [0.00, 0.00]. Secondary token accuracy: native 100.00%, selected 100.00%; these means do not treat individual tokens as independent statistical units.

**Noise 0.2:** native exact reconstruction 100.00%, weighted_1 100.00%; paired difference 0.00 percentage points, 95% CI [0.00, 0.00]. Secondary token accuracy: native 100.00%, selected 100.00%; these means do not treat individual tokens as independent statistical units.

**Noise 0.5:** native exact reconstruction 100.00%, weighted_1 100.00%; paired difference 0.00 percentage points, 95% CI [0.00, 0.00]. Secondary token accuracy: native 100.00%, selected 100.00%; these means do not treat individual tokens as independent statistical units.

Both codecs remain at the reconstruction ceiling at every specified noise amplitude. The test therefore cannot establish a robustness advantage or equivalence beyond these perturbations. Noise nevertheless changes discrete assignments: at SD 0.5, 35,431 native codes and 38,521 selected-operator codes differ from their clean counterparts across the five seeds. Code variation does not imply a decoded reconstruction failure.

![Latent-noise stress test with uncertainty clustered by source solution. Clean reconstruction preservation and noise robustness are distinct claims.](figures/07_repair_noise.pdf)

## Post-hoc contribution and fine-code compensation control

After observing preserved clean reconstruction, a documented post-hoc amendment tests the first 128 hard test solutions with fixed fine codes. An artificial hybrid replaces only the nine level-5 codes by their modified-encoder values, retaining all other native codes. This combination can be outside the native training distribution and is not a deployable repair.

Native, fully reencoded and hybrid streams reconstruct 128/128, 128/128 and 128/128 solutions exactly. The hybrid introduces 0 token errors and changes 0 final grids relative to native. Full reencoding changes 6039 fine codes after level 5. Thus final reconstruction should not be interpreted from occupancy alone; this control measures whether replacing the code while holding its downstream context fixed behaves like full reencoding.

Decoded after level 5, cell accuracy is 21.19% natively and 18.31% after the repair. The paired difference is -2.88 percentage points, CI [-3.65, -2.11]. The level-4 decoding is a common pre-intervention control. These are partial reconstructions of known solutions, not predictions from puzzle clues.

On these 128 contexts, changing the fine codes is not required to preserve final argmax reconstruction. The extra occupancy is therefore not evidence of a necessary new contribution to the final decoded answer. The poorer partial level-5 accuracy further limits an interpretation as better intermediate task representation. Probability changes, redundancy outside these contexts and other decoders remain untested.

## Frozen-generator compatibility and interpretation

Patching the actual quantizer method agrees with the explicit modified codec on eight test solutions: True. The modified method is called 1 time(s) during the encoded batch, but **0 times during native generation** on the corresponding puzzles. Generator codes, logits and answers remain bit-identical: True, True, True. The tested tokenizer state dictionaries before and after are identical.

This equality has a structural explanation: the released Sudoku inference path embeds the puzzle and generates codes without encoding its unknown solution through the modified pooling operation. Therefore changing only this encoder operation cannot improve that inference path. Unchanged generated answers are not evidence that the new codes were learned or that the generator became more robust.

On the 600 test solutions, agreement between native generated level-5 codes and native solution encodings is 100.00%, compared with 1.65% for the modified encoder targets. This is a changed-target diagnostic, not a measure of semantic correctness or a new solving score.

The proposed operator is thus an explicit remedy for the raw averaging invariance. Whether it is also a useful encoder retrofit depends on the held-out reconstruction and stress-test results reported above. Even perfect clean reconstruction would not establish a free improvement to the complete NCM: later quantization levels can compensate, the generator still uses its old target distribution, and the added computation has not been benchmarked. A future jointly trained system could initialize a positive position-weighted filter or learn such weights, while controlling codebook assignments, reconstruction and downstream quality. That recommendation remains an architectural proposal, not a training result from this experiment.

## Repair evidence and reproduction

The repair protocol and selection snapshot precede held-out repair evaluations. All modified encoder inputs, codes and reconstructions are retained by example. `repair.py` supplies the fixed-size operator and a reversible patch context manager; `experiments.py` executes the finite screen, held-out evaluations and generator controls. `verify.py` recomputes statistics and paired intervals, checks previous campaign manifests, and replays the modified native path. `./run.sh` reproduces this repair extension; the earlier cross-domain campaign has its separate runner at `../generalisation_architecturale/run.sh`.

# Related work and interpretation

Vector-quantized latent representations originate in VQ-VAE [2]. Next-scale generation has also been studied in visual autoregressive models [3]. Representation collapse in vector quantization is an established research topic [4]; this note does not introduce codebook collapse as a new phenomenon. The NCM paper itself analyzes tokenizer reconstruction and utilization, perturbs coarse text codes, and discusses fixed hierarchy costs [1, Appendices G-I]. Our distinct focus is the exact structured-data/pooling alignment and the separation of occupancy, invariance and bypass effects on released checkpoints.

The mathematical risk is broader than Sudoku: additive representations followed by averaging cannot distinguish arbitrary within-window reorderings with equal histograms. Its harmful realization is conditional on data regularities, the encoder, normalization, learned samplers, residual paths and training. Sudoku guarantees equal row histograms for every valid solution; general text does not impose the same constraint across contexts. The learned text architectures examined here also differ at the first level. The domain comparison therefore changes both data and architecture and does not isolate a single causal factor between them.

# Limitations and falsifiable next steps

The evidence covers three Sudoku tokenizers and two text tokenizers, one released training run per tokenizer and one local backend. Our interventions are concentrated on the hard-Sudoku generator. Neither the absence of constant text levels nor accurate reconstruction evaluates text generator quality. We have not established a universal defect, nor demonstrated that the remaining text codes are semantically disentangled or necessary. Occupancy and entropy estimates depend on sample size, position and codebook size. The two text strata can share rows and mostly contain full contexts; more padding-rich data would be needed to study length dependence.

We lack training trajectories and matched architecture-by-data retraining. We cannot establish why optimization selected the Sudoku codebook configuration or isolate the causal effects of learned first-level sampling. Earlier dropout and k-means controls are post-hoc. The nine-prediction intervention was selected on validation, but this whole research program was motivated by previously inspected test behavior. Confidence intervals do not integrate variability over training seeds, template estimation or repeated intervention searches. Multiple exploratory comparisons are not multiplicity-corrected.

A stronger causal follow-up would retrain matched additive and contextual encoders with average and learned pooling on both structured token histograms and text, using multiple seeds and predeclared quality margins. A practical follow-up would evaluate adaptive bypasses on genuinely untouched puzzles and measure wall-clock savings with randomized, interleaved timing. Testing further published text tokenizers would broaden empirical scope without turning a finite audit into a universal architectural theorem.

# Reproducibility and evidence

The release includes the frozen protocol, amendments, asset commits and checksums, exact selected text tokens, per-level code arrays and histograms, validation screening predictions, frozen Sudoku selection, held-out intervention predictions, PNG/SVG/PDF figures, CSV summaries, scripts and copied model sources. Previous campaigns remain unchanged and supply the original Sudoku proof controls. The source archive excludes large weights and corpora, whose pinned URLs and local locations are recorded.

The previous cross-domain extension is reproduced with `../generalisation_architecturale/run.sh`; its verification evidence remains preserved. Run `./run.sh` here to reproduce the added repair experiments and integrated article. The local verifier recomputes repair metrics and paired intervals, validates repair selection, checks the earlier campaign manifests, replays the modified native Sudoku encoder and checks the PDF. It does not rerun the earlier text models. Numerical identity is not guaranteed across BLAS, PyTorch or hardware versions. The memory amendment and interrupted attempt are retained in the journal.

# References

1. Kumbong et al. (2026). *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*. [arXiv:2610.14162v1](https://arxiv.org/html/2610.14162v1). [Official model assets](https://huggingface.co/hazyresearch/ncm-checkpoints), [official tokenized data](https://huggingface.co/datasets/hazyresearch/ncm-datasets), [source repository](https://github.com/HazyResearch/ncm).
2. van den Oord, Vinyals and Kavukcuoglu (2017). *Neural Discrete Representation Learning*. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937).
3. Tian et al. (2024). *Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction*. [arXiv:2404.02905](https://arxiv.org/abs/2404.02905).
4. *Representation Collapsing Problems in Vector Quantization* (2024). [arXiv:2411.16550](https://arxiv.org/abs/2411.16550).

# Appendix: numerical and interpretive checks

The complete per-level statistics are in `results/text_levels.csv` and `results/sudoku_levels.csv`; exact histograms and code sequences are retained in JSON/NPZ. The first learned downsampler changes scale as well as resolution. In particular, TinyStories exhibits large first-input magnitudes; its input/residual variance ratio must not be interpreted as attenuation by an ordinary averaging operator. Direct nearest-assignment checks and native-encode parity pass on the examined samples, but these checks are not a broad numerical stability study. Raw first-level scales are retained in the JSON evidence rather than silently normalized away.


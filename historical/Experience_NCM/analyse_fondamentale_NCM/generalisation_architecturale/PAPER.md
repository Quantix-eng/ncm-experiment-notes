---
title: "Architectural Invariance and Bypassable Levels in Next-Concept Models: A Cross-Domain Audit"
date: "24 October 2026"
geometry: margin=0.85in
fontsize: 10pt
---

# Abstract

An empirically constant quantization level in a structured task does not by itself establish an architectural failure across domains. We audit published Next-Concept Model (NCM) checkpoints, connecting a Sudoku representation constraint to controlled tests of two text tokenizers. In an additive encoder, fixed token histograms within pooling windows yield identical pooled embeddings before normalization and residual subtraction. The Sudoku serialization aligns the nine-position scale with complete rows and their fixed digit histograms. This explains a structural source of low variation, but does not mathematically force residual codebook collapse. On LM1B and TinyStories, we audit all 12 levels using 256 randomly selected validation contexts and a second stratum of 128 long contexts per tokenizer. Neither exhibits an empirically constant level in either stratum. Block permutations preserve local raw embedding averages while changing native tokenizer codes, separating an isolated pooling invariance from end-to-end representational collapse. A validation-only screen of Sudoku bypasses selects levels 5, 2, 11. Retaining their modal code contributions reduces predicted levels from 12 to 9 and solves 510/600 test puzzles versus 516/600 natively: -1.00 percentage points, paired 95% CI [-2.17, 0.17]. These results support a data/architecture interaction and checkpoint-specific bypass opportunities; they do not establish a universal NCM defect, exact equivalence of the nine-pass variant, or a measured speedup.

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

# Related work and interpretation

Vector-quantized latent representations originate in VQ-VAE [2]. Next-scale generation has also been studied in visual autoregressive models [3]. Representation collapse in vector quantization is an established research topic [4]; this note does not introduce codebook collapse as a new phenomenon. The NCM paper itself analyzes tokenizer reconstruction and utilization, perturbs coarse text codes, and discusses fixed hierarchy costs [1, Appendices G-I]. Our distinct focus is the exact structured-data/pooling alignment and the separation of occupancy, invariance and bypass effects on released checkpoints.

The mathematical risk is broader than Sudoku: additive representations followed by averaging cannot distinguish arbitrary within-window reorderings with equal histograms. Its harmful realization is conditional on data regularities, the encoder, normalization, learned samplers, residual paths and training. Sudoku guarantees equal row histograms for every valid solution; general text does not impose the same constraint across contexts. The learned text architectures examined here also differ at the first level. The domain comparison therefore changes both data and architecture and does not isolate a single causal factor between them.

# Limitations and falsifiable next steps

The evidence covers three Sudoku tokenizers and two text tokenizers, one released training run per tokenizer and one local backend. Our interventions are concentrated on the hard-Sudoku generator. Neither the absence of constant text levels nor accurate reconstruction evaluates text generator quality. We have not established a universal defect, nor demonstrated that the remaining text codes are semantically disentangled or necessary. Occupancy and entropy estimates depend on sample size, position and codebook size. The two text strata can share rows and mostly contain full contexts; more padding-rich data would be needed to study length dependence.

We lack training trajectories and matched architecture-by-data retraining. We cannot establish why optimization selected the Sudoku codebook configuration or isolate the causal effects of learned first-level sampling. Earlier dropout and k-means controls are post-hoc. The nine-prediction intervention was selected on validation, but this whole research program was motivated by previously inspected test behavior. Confidence intervals do not integrate variability over training seeds, template estimation or repeated intervention searches. Multiple exploratory comparisons are not multiplicity-corrected.

A stronger causal follow-up would retrain matched additive and contextual encoders with average and learned pooling on both structured token histograms and text, using multiple seeds and predeclared quality margins. A practical follow-up would evaluate adaptive bypasses on genuinely untouched puzzles and measure wall-clock savings with randomized, interleaved timing. Testing further published text tokenizers would broaden empirical scope without turning a finite audit into a universal architectural theorem.

# Reproducibility and evidence

The release includes the frozen protocol, amendments, asset commits and checksums, exact selected text tokens, per-level code arrays and histograms, validation screening predictions, frozen Sudoku selection, held-out intervention predictions, PNG/SVG/PDF figures, CSV summaries, scripts and copied model sources. Previous campaigns remain unchanged and supply the original Sudoku proof controls. The source archive excludes large weights and corpora, whose pinned URLs and local locations are recorded.

Run `./run.sh` from the study folder to reproduce the extension sequentially, or `../.venv/bin/python scripts/verify.py` with the cache variables in that runner to verify existing evidence. The verifier checks manifests, recomputes metrics and bootstrap intervals from saved predictions, validates selection, replays native text encoding and verifies the PDF. Numerical identity is not guaranteed across BLAS, PyTorch or hardware versions. The memory amendment and interrupted attempt are retained in the journal.

# References

1. Kumbong et al. (2026). *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*. [arXiv:2610.14162v1](https://arxiv.org/html/2610.14162v1). [Official model assets](https://huggingface.co/hazyresearch/ncm-checkpoints), [official tokenized data](https://huggingface.co/datasets/hazyresearch/ncm-datasets), [source repository](https://github.com/HazyResearch/ncm).
2. van den Oord, Vinyals and Kavukcuoglu (2017). *Neural Discrete Representation Learning*. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937).
3. Tian et al. (2024). *Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction*. [arXiv:2404.02905](https://arxiv.org/abs/2404.02905).
4. *Representation Collapsing Problems in Vector Quantization* (2024). [arXiv:2411.16550](https://arxiv.org/abs/2411.16550).

# Appendix: numerical and interpretive checks

The complete per-level statistics are in `results/text_levels.csv` and `results/sudoku_levels.csv`; exact histograms and code sequences are retained in JSON/NPZ. The first learned downsampler changes scale as well as resolution. In particular, TinyStories exhibits large first-input magnitudes; its input/residual variance ratio must not be interpreted as attenuation by an ordinary averaging operator. Direct nearest-assignment checks and native-encode parity pass on the examined samples, but these checks are not a broad numerical stability study. Raw first-level scales are retained in the JSON evidence rather than silently normalized away.

The earlier level-9 revision study found that a strong observable baseline predicts replacement values and that adding the examined hidden summaries gives no established improvement. That result concerns predictive monitoring and interventions on a different question. It does not establish the architectural invariance result or strengthen the universality of the present tokenizer comparison.

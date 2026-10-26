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

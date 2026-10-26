# Attributing Apparent Revision in Next-Concept Models

Coarse-to-fine generation invites an interpretation of changing intermediate outputs as decisions being revised. We examine what such observations identify in published Next-Concept Model (NCM) checkpoints, using a retrospective synthesis of eight frozen-model campaigns. On Sudoku, a codec given the correct solution still produces non-monotone reconstructions: 3176/12717 values corrected at level 9 become wrong again. This establishes codec sufficiency, not exclusive attribution of generated revisions. Separately, we prove a raw additive-pooling symmetry, characterize a constant intermediate code, and test a position-weighted encoding repair that preserves recorded reconstruction but does not alter the frozen generation path. A four-condition intervention on 256 new puzzles fails its announced recovery-and-specificity criterion for omitting residual 9. At the terminal level, target-code substitutions write requested digits in 784/1024 trials, versus 0 for locally norm-matched Gaussian controls; a fixed lookup and external checker certify 552/1024 grids without calling the decoder. Non-degradation is conditional on uniqueness, checker soundness and exact fallback; an end-to-end speedup is not established. Text-codec and LM1B-generator controls extend non-monotonicity beyond Sudoku without identifying a universal architectural defect. The contribution is a bounded attribution of several observable effects, with claim-to-code-to-record links, rather than an account of metacognition or an exhaustive causal anatomy.

# Introduction

Next-Concept Modeling (NCM) generates a sequence through progressively finer discrete representations rather than emitting one text token at a time [@kumbong2026ncm]. This organization makes a tempting behavioral interpretation possible: when a decoded Sudoku value or text token changes between levels, the model appears to reconsider an answer. Yet the object being read is a cumulative latent representation, and the intermediate decoded sequence need not be the object on which the next generation step operates. **What do apparent revisions actually identify about the mechanism?**

We address this question by bringing together eight completed campaigns on published, frozen checkpoints. Their original questions included confidence prediction, intermediate-code redundancy, temporal locking, compensation and direct readout. The present manuscript selects and connects their evidence retrospectively; it introduces no new experiments and does not convert their history into a single preregistered study. Its organizing principle is to separate an observation, a subsystem sufficient to produce it, and the stronger causal interpretation that would require additional evidence.

The resulting account has four parts (Figure 1). First, intermediate-grid non-monotonicity survives when the codec encodes the correct answer without running the generator. Second, an elementary symmetry identifies information that raw additive averaging discards; a weighted operator breaks this symmetry, although the frozen generator never invokes that repaired encoding path. Third, code changes following an omission do not meet the announced test of useful, state-specific compensation. Fourth, terminal codes support a simple predictive table and local interventions, which can be exploited by an external certificate with exact fallback. These are connected questions about attribution and accessibility, **not a demonstrated causal chain from pooling to oscillation to a stable terminal decision**.

\input{figures/overview.tex}

Our contribution is therefore a set of checkable distinctions: codec sufficiency versus exclusive origin; an operator identity versus learned code collapse; adaptation versus useful compensation; and empirical readout coverage versus a conditional verification guarantee. Residual quantization itself is established machinery [@lee2022rq], and the checker argument is not specific to NCM. We do not claim a new theory of consciousness, a universal defect, or a complete explanation of the studied models. Each central claim is indexed E01--E14 in the accompanying evidence ledger; those identifiers link the printed numbers to saved records, code and compact traces.

# Experimental objects and evidential status {#sec-validity}

For a puzzle $p$, the executed Sudoku runner predicts a block of codes $c_k$, adds its residual contribution $R_k(c_k)$, and eventually decodes the sum:
$$
c_k=\arg\max G_k(\operatorname{pool}_k(z_{k-1}),p),\qquad
z_k=z_{k-1}+R_k(c_k),\qquad \widehat{s}_k=\arg\max D(z_k).
$$
The first level uses a dedicated initial query. Previously emitted codes remain fixed at every level by construction. The decoded values $\widehat{s}_k$ can change because every new contribution changes the input to $D$. The intermediate grids used here are observational readouts: the generator does not receive their argmax values as feedback. Encoder nearest-codebook quantization and generator code-logit argmax are different operations. The decoder also contains attention; removing the generator does not remove all attention.

The three Sudoku checkpoints have distinct weights. Their twelve generation lengths are $1,2,4,6,9,13,18,26,37,50,67,89$, distinct from their eight Transformer layers. A complete sequence contains 81 digits and eight separators. The official easy, medium and hard corpora each contain 2,000 puzzles but share solution families across difficulties. The initial hard partition is 1,000 training, 400 validation and 600 test puzzles. Those old test sets and an earlier 256-puzzle external set were repeatedly inspected in subsequent campaigns. They are not fresh confirmations each time they reappear. The crossed intervention and terminal-readout campaigns added separate sets of 256 and 1,024 new unique-solution, 30-clue puzzles after their respective local freezes. All conclusions remain conditional on these non-uniformly generated families and the published weights. Detectable overlap exclusions do not rule out unknown training contamination.

Unless specified otherwise, the reported intervals are historical paired percentile bootstraps by puzzle, not by cell; weights remain fixed. The crossed and terminal campaigns used 4,000 resamples and 97.5% intervals for two locally co-primary criteria (Bonferroni). Descriptive intervals are 95%. We retain the original units, criteria and sampling scopes; this synthesis performs no new estimation. All computations were CPU FP32. Parity and software checks are described in the supplement with their actual scopes, rather than treated as independent scientific replications. E13 records provenance and study dependencies.

# Does a changing grid imply a generator decision? {#sec-dynamics}

## Strict decoded-value locking fails

On the 600 hard test puzzles, 12333 cells are incorrect at level 8 and correct at level 9. Of these, 3088 become incorrect at least once at levels 10--12: **25.04%**, 95% CI [24.28, 25.81]%. Thus a strict claim that level 9 irreversibly locks corrected decoded values has direct counterexamples. This does not rule out every possible latent commitment. Stopping at levels 8, 9, 10 or 11 gives zero exact grids; level 12 gives 516/600. The early-cutoff contrast establishes prefix insufficiency under this decoder, not a special decision at level 9. Full-grid exactness is a conjunction of 81 cell events and format, so a late jump can coexist with gradual cell improvement. E04 links the full trajectories and interventions.

## The correct-answer codec already produces reversals

The oracle control encodes each correct solution through the encoder and quantizer, then decodes the cumulative contributions. It never calls the generator. On the same 600 families, 3176/12717 corrections at level 9 subsequently degrade: **24.97%**, 95% CI [24.26, 25.73]%. Its final reconstruction is exact on all 600 solutions, while no earlier complete reconstruction is exact. Reversals are therefore possible without the generator solving the puzzle from its clues (E05).

This is a positive localization result at subsystem resolution: the progressive codec is sufficient. It does not identify a specific decoder module, the origin of every generated reversal, or a causal fraction attributable to the codec. The generated and oracle rates concern different sets of corrected cells; their numerical proximity is not an equivalence test. The raw-pooling symmetry studied next was not intervened on to measure these reversals, so it cannot be inserted as their established cause.

## Text broadens the observation, not its interpretation

The LM1B small **generator**, not only its tokenizer, was run for 128 unconditional trajectories. Among positions that adopt at level 9 the raw token they will have at the final level, 416/2340 abandon it at level 10 or 11: **17.78%**, 95% CI [16.11, 19.60]%. This final-token anchor is retrospective and is not a correct answer. A separate oracle-codec control on 128 reused validation phrases has true reconstruction references: 97/1675 corrections at 9 degrade later (**5.79%**, 95% CI [4.70, 6.90]%). The masks, sampling, raw/heuristic distinction and excluded loader incident are documented in the supplement (E07). These observations establish non-Sudoku examples, not semantic self-correction or universality.

# What information can aggregation preserve? {#sec-pooling}

Having separated readout dynamics from an inferred decision, we turn to a distinct mechanistic question: what can the encoding operator distinguish before any such interpretation? This question concerns information loss; its answer does not identify the source of every oscillation.

## An exact raw symmetry and an empirical constant level

Let $h_i(x)=e(x_i)+p_i$ be additive token and position embeddings, and let $W_j$ be a fixed pooling window. If two inputs have the same token histogram inside $W_j$, then
$$
\frac{1}{|W_j|}\sum_{i\in W_j}h_i(x)
=\frac{1}{|W_j|}\sum_a n_{j,a}e(a)+\frac{1}{|W_j|}\sum_{i\in W_j}p_i
$$
is identical for the two inputs. The identity follows by grouping equal tokens; additive position embeddings do not restore order after averaging. For the Sudoku serialization, adaptive area pooling from 89 to nine positions covers the nine row digits, plus a fixed number of separators, in every window. Valid solutions thus have fixed within-window histograms. This is **level 5, with nine positions**, not generation level 9, with 37 positions (E01).

The exact identity applies to raw additive embeddings. The executed encoder applies GroupNorm before quantization, and the residual hierarchy subtracts earlier contributions. These operations can preserve input dependence. In the hard audit, raw inter-puzzle pooled variance is 3.30e-16 at length nine, versus 4.65e-02 and 5.93e-02 at neighboring lengths eight and ten; after normalization, the length-nine value is 1.58e-08. Nevertheless, the native hard level-5 code is empirically constant at identifier 11 on the examined clean inputs. The theorem does not force this learned outcome or identify its training history.

Replacing the level-5 Transformer prediction by code 11 **while retaining its residual** preserves recorded codes and outputs on 600 hard test puzzles and 256 earlier external puzzles; exact-grid counts remain 516 and 219. Removing the residual is a different intervention and changes some answers. A constant code need not be a zero or useless contribution. No reliable speed benefit for bypassing this prediction is established (E02). Native LM1B and TinyStories tokenizer audits do not show a constant level; their raw symmetry controls do not establish native collapse (E06).

## A symmetry-breaking repair has a bounded scope {#sec-repair}

Replace a window mean by $\sum_i w_{ji}h_i$, where
$$
w_{ji}=\frac{1+\alpha t_{ji}}{|W_j|},\qquad t_{ji}\in[-1,1],\quad \sum_i t_{ji}=0.
$$
For $\alpha=1$, weights are nonnegative and sum to one. Swapping distinct token embeddings at positions $i,h$ changes the output by $(w_{ji}-w_{jh})(e(b)-e(a))$. Distinct weights therefore break the corresponding swap symmetry, without making this compression injective or proving better solving.

A validation-selected fixed weighting preserves solution reconstruction on the held-out hard set (600/600), earlier external set (256/256), easy (600/600) and medium (599/600), matching native exact counts. It activates 15/16 level-5 codes on hard test, with positional entropy 2.5887 nat. These are encodings of **known solutions**, not generations from clues (E03).

The repaired encoding pool is never called by the controlled native generation path. Generated answers remain identical; agreement between its fixed level-5 prediction and the new encoding targets drops from 100.00% to 1.65%. Thus the repair is an operator proof of concept, not a free improvement to the frozen NCM. Codec noise tests remain at ceiling in both arms and establish no robustness benefit. Changing an encoding target and teaching a generator to predict it are separate problems.

# Does adaptation specifically repair an omitted contribution? {#sec-crossed}

Codec sufficiency does not preclude a generator that adapts usefully to a changed latent. Conversely, altered subsequent codes do not prove repair. We therefore retain the crossed omission experiment as a discriminating test, with its unfavorable result visible.

After level 9, let $z=z_9$ and $\delta=-r_9$. Let $T_0$ contain native contributions 10--12, and $T_\delta$ the contributions obtained by regenerating those levels from $z_8$. Four decoded conditions separate perturbing the state from changing the continuation:
$$
\begin{array}{ll}
Y_{00}=D(z+T_0),&Y_{10}=D(z+\delta+T_0),\\
Y_{11}=D(z+\delta+T_\delta),&Y_{01}=D(z+T_\delta).
\end{array}
$$
These expressions abbreviate ordered residual additions; the perturbed branch starts directly from saved $z_8$ to avoid a subtract/add round trip. The crossed states are artificial interventions and may be off distribution.

Let $L_{ab}$ be reference-digit NLL, averaged over initially empty cells within each puzzle. The announced contrasts are recovery $R=L_{10}-L_{11}$ and specificity $I=(L_{10}-L_{11})-(L_{00}-L_{01})$. The latter asks whether the adapted continuation helps the perturbed state more than the native state. Neither is an intent measure or a unique mediation decomposition.

\begin{table}[t]
\centering
\caption{All four conditions on the new 256-puzzle corpus. NLL is the announced primary loss; exact-grid counts are secondary. E08.}
\label{tab:crossed}
\begin{tabular}{llrr}
\toprule
Condition & State / continuation & Mean NLL (nat) & Exact grids \\
\midrule
$Y_{00}$ & native / native & 0.356166 & 222 \\
$Y_{10}$ & omitted / native & 0.349218 & 221 \\
$Y_{11}$ & omitted / regenerated & 0.350859 & 224 \\
$Y_{01}$ & native / regenerated & 0.354004 & 224 \\
\bottomrule
\end{tabular}
\end{table}

The locally frozen co-primary criterion required both 97.5% lower bounds to be positive on the 256 new puzzles. Instead, **$R=-0.001641$ nat**, CI [-0.02636, 0.02538], and **$I=-0.003803$ nat**, CI [-0.00725, -0.00095]. The criterion fails. The negative interaction is inconsistent with the proposed positive specificity for this omission and loss. The frozen continuation under omission already has lower mean NLL than the native condition, so the intervention does not create a consistent primary-loss deficit to repair. A secondary gain of three exact grids from $Y_{10}$ to $Y_{11}$ cannot replace the announced outcome.

Later codes change, but the mean cosine of $\Delta T=T_\delta-T_0$ with $-\delta$ is 0.0838, and $\|\delta+\Delta T\|/\|\delta\|$ averages 1.1450. This is not Euclidean cancellation. An earlier external set has positive $R$ but no positive specificity conclusion; its separate results remain in the supplement. The evidence supports adaptation and tolerance to this omission, **not established useful, state-specific compensation**. It does not establish that adaptation is globally blind to correctness or that all other perturbations would fail.

# What is accessible at the terminal level? {#sec-terminal}

The preceding tests constrain interpretations of intermediate change. They do not prevent a more local positive result at the endpoint. The final 89 code positions align with the 89 output positions. A table fitted on old training codes and their native decoded digits maps each identifier to its most frequent digit. A second fixed choice selects a code for each requested digit. This is a partial readout and intervention rule; it is not evidence that the system has dynamically stabilized.

## Association and intervention are different claims

On 1,024 new hard puzzles, lookup agrees with native decoded digits on 99.47% of cells. For the causal test, one initially empty cell is sampled uniformly in each of the first 128 new puzzles; all eight digits different from its native pre-intervention readout are requested. Other codes remain fixed. The intervention replaces only the selected terminal code through $z_{12}+R_{12}(c'_{12})-R_{12}(c_{12})$.

Target substitutions write the requested digit in **784/1024 trials (76.56%)**, versus **0/1024** for Gaussian directions matched in norm at every affected latent position. The paired advantage is 76.56 percentage points, 97.5% CI [71.68, 80.96]. No argmax of the other 80 cells changes in these target substitutions. This does not assert invariant logits or separators, population-wide absence of collateral effects, or independence across the eight requests from a source (E09).

The terminal residual includes a width-three convolution. Controls match local support and norm, including recorded FP32 tails; a post-hoc direct-affine FP64 audit changes no decoded output. They do not match the geometry of valid codebook differences. The contrast therefore establishes efficacy of the tested target-associated substitutions against these Gaussian controls, not new context-free semantics, a Sudoku rule circuit or an NCM-specific phenomenon. Predictive agreement and writing success use different positions and conditions: their numerical gap is not a controlled estimate of context dependence.

\begin{figure}[t]
\centering
\includegraphics[width=.94\linewidth]{figures/writing.pdf}
\caption{Recorded terminal interventions. The annotated interval retains its historical definition; source puzzles, rather than requests or cells, are bootstrap units. Gaussian controls match local support and norm, not codebook geometry. The original writing and collateral metrics are retained. E09.}
\label{fig:terminal}
\end{figure}

## A certificate converts partial readout into safe fallback

The policy constructs a digit grid from the table, supplies fixed separators, and verifies the 27 Sudoku units, clues and format. It returns an accepted grid, otherwise calls the native decoder on exactly the same singleton latent. Neither the reference nor a uniqueness solver is available to this policy.

**Conditional guarantee (E11).** If a puzzle has unique solution $s$ and the checker accepts only valid clue-respecting grids, an accepted candidate equals $s$. When the native decoder is correct, the policy therefore either returns $s$ by certification or returns that same native output by fallback. Exact success cannot decrease pointwise under these assumptions. This elementary verification argument applies to any candidate provider; it is not a learned property of NCM. On multiple-solution puzzles it does not preserve agreement with an arbitrary chosen reference, and rejected candidates can still lead to an incorrect native fallback.

The policy certifies **552/1024 = 53.91%**, 97.5% CI [50.29, 57.32]%, avoiding those decoder calls (E10). Native and hybrid exact counts are 883 and 885; the two rescued grids do not establish a population improvement. The local 30% coverage criterion is met, but was selected after a development coverage of 201/400. It has no independent justification as a minimum useful threshold. The favorable development selection, local freeze and absence of a complete immutable exploration log remain material qualifications, not claims of independent preregistration.

## Decoder avoidance is not established end-to-end acceleration

Historical paired timings on the first 64 new puzzles, one CPU thread and one session, retain all twelve generator predictions. Arithmetic mean pipeline latency is 254.367 ms native versus 249.609 ms hybrid. The announced geometric native/hybrid ratio is 1.0147, 95% CI [0.9939, 1.0374]; it does **not** establish a global speedup. For readout alone the arithmetic means are 1.091 and 0.493 ms. Its much larger geometric ratio (9.2117) is not a division of mean readout cost, still less total model cost. The recorded session does not establish GPU, batching, energy or memory savings (E12).

# Discussion and limits of attribution

The common thread is a change in the level of explanation. A visible revision is first an observation of a reader applied to a changing latent. The oracle control places a sufficient source inside the codec. The pooling analysis identifies a separate, exact information-loss symmetry and a repair with an explicit deployment boundary. The crossed intervention tests, and does not confirm, one stronger functional interpretation of code adaptation. The endpoint then provides a limited operational result: some code substitutions work locally, and some table outputs can be checked before fallback. This progression relates the findings without replacing missing causal links by a story.

**No exclusive source attribution.** Neither the oracle control nor the pooling proof apportions all generated revisions between quantization, normalization, decoder attention, MLPs and generator computations. No intervention in these campaigns establishes pooling as the cause of oscillation. The final alphabet does not prove stability or irreversibility; generation ends by protocol. The results are compatible with multiple mechanisms at once.

**No exhaustive test of metacognition or consciousness.** External confidence probes are predictors fitted by us. Readout reversals, useful compensation and internal monitoring are distinct constructs. Failing one omission test does not prove that the model lacks all monitoring, confidence representations or control of its computation. Consciousness was not operationalized. Claims that NCM is intelligent, unintelligent, conscious or non-conscious do not follow from these experiments.

**Limited generality and novelty.** Three Sudoku checkpoints, two text tokenizers and one text generator are not a controlled comparison of architectures or training histories. Shared Sudoku families are not independent replications. The terminal association may be an expected consequence of a position-aligned reconstruction codebook; no comparator codec establishes otherwise. Selective prediction and adaptive computation have substantial antecedents [@geifman2019selectivenet; @schuster2022calm]. Our certificate uses an exact external verifier and retains the full generator, rather than implementing their learned selection or early-exit mechanisms. We make no priority claim.

**Retrospective synthesis and selection.** Local new-data evaluations matter, but do not erase the exploratory search across campaigns. The terminal direction was sought after unsuccessful hypotheses and selected on favorable development results. Protocol documents and recorded timestamps are locally conserved; scripts were not all included in their initial freeze. Hashes establish consistency of present artifacts, not an unalterable history or the absence of p-hacking. Bootstraps are conditional on fixed weights and do not incorporate retraining or all research-selection uncertainty. The evidence ledger makes scrutiny easier; it does not make the claims immune to criticism. Attention patching and disambiguating-puzzle experiments remain unexecuted proposals, not hidden support.

# Conclusion

The accumulated evidence supports bounded statements about pooling symmetry, codec-generated non-monotonicity, an unconfirmed conditional-compensation hypothesis and terminal readout with external certification. It does not support a single exclusive mechanism for apparent revision. The scientific value lies in separating these levels of attribution and supplying the code, recorded outcomes and counterexamples needed to inspect each claim. New experiments are outside this synthesis; stronger mechanism or generality claims remain open.

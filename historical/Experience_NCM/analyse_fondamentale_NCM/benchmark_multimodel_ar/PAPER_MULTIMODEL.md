# Attributing Apparent Revision in Neural Generators: A Calibrated Protocol with NCM and Autoregressive Case Studies

Working manuscript, version 2, 25 October 2026. This version adds a locally frozen GPT-2 experiment to the retained NCM study and analytical calibration. It preserves their original outcomes and distinguishes the new prespecified experiment from post-hoc descriptive exports. No public release, external validation or submission is implied.

## Abstract

A changing intermediate answer does not identify the computation responsible for that change. We develop a revision-attribution protocol with analytical controls, crossed continuations and explicit access contracts. Constructed systems share identical readout trajectories despite different continuation programs; a complete diagnostic recognizes all 344 announced library instances as expected by construction. We then instantiate the protocol on two frozen learned architectures. A Sudoku NCM solves all 48 counterfactual variants; its preference diagnostic is input-dominant in 15/16 source families and abstains in one. Local perturbations create no substantial repair opportunity, and regenerated state-swap continuations are matched in exactness by recipient-native frozen controls. For GPT-2 small on 48 controlled word-copy prompts, native accuracy is 33/48. Swapping the last-position state after block six while retaining recipient context yields input-dominant preference in 8/16 families; the other eight fail the native-anchor gate and remain non-identifiable. Regenerated and recipient-native frozen tails give 19/32 and 18/32 recipient-correct swapped outcomes. Mean NLL recovery under local perturbation coexists with two correctness losses. These studies establish executable interventions on real FP32 states while illustrating limits on attribution. They do not identify internal-mechanism ground truth, separate architectures independently of task and access, or measure a Wyner common-information rate. The resource is a calibrated protocol with two learned-model case studies, requiring independent methods and external evaluation before a reference-benchmark claim.

## 1. Introduction

Parallel generators expose sequences of intermediate representations whose decoded outputs may appear to revise an answer. An initially incorrect output can become correct, become incorrect again and eventually recover. Such a trajectory is compatible with useful adaptation, progressive reconstruction of an answer already encoded, or a continuation that recomputes from an unchanged input. Correct final performance does not distinguish these accounts.

The practical problem is attribution: which observations justify saying that a continuation corrected a perturbed computation? In particular, a successful regenerated continuation is difficult to interpret when a previously generated, frozen continuation already yields the same answer. Similarly, tolerance to an intervention creates little opportunity to observe repair. Both cases require controls that a trajectory-only evaluation omits.

We develop an evaluation protocol around three access regimes: native observations; crossed state–continuation interventions at fixed input; and reciprocal input–state interventions with frozen and regenerated continuations. The protocol permits set-valued answers and explicitly separates functional diagnostics from claims about an internal algorithm. Analytical systems establish known equivalence classes under each observation regime. Closely related Sudoku inputs provide a learned-model application with verifiable counterfactual answers.

Our contribution has three parts. First, we formalize controls in which readout revision is matched while continuation mechanisms differ, including a response that detects perturbation but does not respond to its identity. Second, we implement the protocol on real FP32 NCM states and retain controls that prevent a successful state swap from being mistaken for necessary adaptation. Third, we delimit a connection to common-information theory and test a small, explicitly finite notion of input distinguishability in saved codes. A second learned-model study applies the protocol to GPT-2, preserving a recipient-context channel while exchanging its last-position state between layers. This broadens the executed architectures and tasks, and tests whether functional signatures can be treated as architecture labels. The contribution is a protocol and two case studies; we do not claim a reference benchmark, a new common-information theorem, or independently validated mechanism classification.

## 2. Related work and intended scope

[MIB](https://proceedings.mlr.press/v267/mueller25a.html) evaluates localization of causal circuits and variables across models and tasks, including a model with known circuitry. Our target is narrower: attribution of temporal answer change when the readout itself can create apparent revision. Known controls and counterfactual interventions are therefore antecedents, not priority claims. We have not compared our diagnostics against independent MIB methods or established that this resource improves their evaluation.

[Tyen et al.](https://aclanthology.org/2024.findings-acl.826/) separate finding a reasoning error from correcting one when its location is supplied. Our distinction concerns where an apparent correction originates and whether a changed continuation is necessary and specific. These questions are related but our experiments do not replicate their language-model tasks.

[Next-Concept Modeling](https://arxiv.org/html/2610.14162v1) generates discrete levels sequentially while predicting positions within a level in parallel, then decodes the accumulated reconstruction. We use its frozen Sudoku hard checkpoint and locally retained inference implementation. Architectural statements about the executed path are supported by the pinned [code](../vendor/ncm_mac/cached.py) and [study adapter](../benchmark_ncm_reel/runtime.py); the observations below concern this checkpoint and data, not the entire NCM family.

Common-information theory concerns distributions and specified synthesis architectures. [Wyner](https://doi.org/10.1109/TIT.1975.1055346), its [multivariate extension](https://arxiv.org/abs/1010.3613) and [conditional common information](https://www.isiweb.ee.ethz.ch/papers/docu/rgra-alap-mwig-2022-2.pdf) supply a relevant formal vocabulary. They do not turn a count of input-dominant interventions into an information estimate. Section 7 states the assumptions needed for that connection.

## 3. Evaluation object and intervention protocol

### 3.1 Functional claims and observational access

Let x be an input, z its intermediate state, T a continuation and D the remaining readout. We distinguish an observed improvement in D from a useful change in T. For a perturbation delta, save the native continuation T0 and the continuation Tdelta generated from the perturbed state. Never substitute a regenerated continuation silently for a frozen baseline.

Regime O provides native trajectories. Regime C provides outcomes from crossed continuations at fixed input. Regime E adds reciprocal input/state swaps, omission of the continuation, and answer-preserving input controls. These are explicit observation contracts, not unrestricted access to every saved tensor. For the analytical calibration, [the actual projections](../benchmark_causal_revision/methods.py) expose discrete trajectories in O, crossed losses in C and selected outcome distances in E. Full states and residuals are retained for audit. A different white-box access contract would require different equivalence classes.

The labels of the analytical library describe its programs, not mutually exclusive categories of cognition. Recomputing from input can also compensate a residual perturbation. On a learned model, effects are reported as conditional functional dependencies and ambiguous cases remain ambiguous.

### 3.2 Crossed continuations

With ordered residual addition suppressed in the notation, define

\[
Y_{00}=D(z+T_0),\quad Y_{10}=D(z+\delta+T_0),\quad
Y_{11}=D(z+\delta+T_\delta),\quad Y_{01}=D(z+T_\delta).
\]

Add \(Y_{1w}=D(z+\delta+T_{\delta'})\), where delta-prime perturbs a different support with the same norm and sign. Given reference loss L, measure

\[
d=L_{10}-L_{00},\quad r=L_{10}-L_{11},\quad
i=L_{10}+L_{01}-L_{11}-L_{00},\quad s=L_{1w}-L_{11}.
\]

Lower loss is better. Damage d measures the opportunity to repair; r measures recovery; i compares the continuation change across native and perturbed states; s compares matched and mismatched adaptations. These are contrasts, not mediation fractions or measurements of an intention to repair. The mismatched control matches norm and sign, not difficulty or full latent geometry.

**A specificity trap.** A constructed program detects any nonzero perturbation and always applies the same correction -delta0. For the perturbation delta0, squared-loss values can be L00=0, L10=k, L11=0, L01=k, with k>0. Then r=k and i=2k are positive. Yet another perturbation elicits the same continuation, so s=0. Useful recovery for one intervention therefore does not establish a response specific to perturbation identity.

### 3.3 Reciprocal input/state/continuation design

For inputs A and B with different known answers, save their native states z_A and z_B at the same cutoff. Execute all combinations of two recipient inputs, two donor states, and three continuation policies: frozen native A, frozen native B, and regeneration from the input/state combination. This gives 12 conditions. Transfer only the donor state; build the prefix cache from the recipient input. With state and tail fixed, changing recipient input cannot affect a decoder that does not directly read that input, providing an implementation check.

Use an additional answer-preserving pair as a control for input changes unrelated to changing the correct solution. A valid Sudoku output must be distinguished from a solution consistent with the recipient's clues. A donor solution can satisfy every Sudoku unit while failing the recipient puzzle.

### 3.4 Calibration systems and scoring

The [analytical construction](../benchmark_causal_revision/controls.py) uses answer vector a, mutable payload b and a fixed nonzero residual v. Its programs implement: oracle replay; tolerance of an ignored mutable channel with a useful fixed continuation; correction from a stored state anchor; oracle recomputation from input; a fixed response to detected perturbation; and mixtures of state and input targets. The programs receive oracle answers during construction. They are not learned solvers or deployable answer-free systems.

For each source, a common time-dependent reader offset creates identical error/correction/reversal/correction trajectories and identical final accuracy. The observations thus cannot uniquely identify the program. At fixed input with anchor a equal to the correct answer, state compensation, input recomputation and their mixtures also produce identical crossed continuations. Regime C must preserve this ambiguity. Oracle replay and tolerance are indistinguishable only under the executed output/loss projection; their actual residual vectors differ and would distinguish them under a richer contract.

The four fixed diagnostics are a forced trajectory attribution, a cautious observation-only set, a crossed-outcome set, and a complete finite-library diagnostic. Report true-class coverage, set size, number of singleton decisions and correctness of those decisions. Coverage alone rewards vacuous predictions; singleton accuracy alone ignores abstention. The complete diagnostic recognizes the announced finite library. Outside it, its mixture result must mean “mixture or unmodeled mechanism.”

## 4. Data, execution and inference rules

### 4.1 Constructed sources and natural counterfactuals

Calibration uses 52 source families: 12 equality, 12 even-parity, 12 permutation and 16 Sudoku families. Equality and parity form development; permutation and Sudoku are reserved structural evaluation families. Five pure programs run on every source; three mixture coefficients, absent from development, run on each of 28 evaluation sources. This yields 344 instances, including 224 evaluation instances. These are constructed partitions, not a trained method's external generalization test.

Each Sudoku family starts from a base with exactly two solutions. Counting is capped at three, so a count of two is exhaustive rather than censored at two. Add one of two alternative digits at a cell where the solutions differ to obtain unique puzzles A and B with equal clue counts and exactly one differing input cell. Add a true clue to A to form Aplus with the same solution. MRV backtracking and Algorithm X independently verify the base and all three unique variants. Branches have 29–62 clues; difficulty is not otherwise calibrated.

The 16 families were generated after the analytical protocol freeze. They were already known when the learned-model protocol was written, although their new NCM outcomes were not used to set it. D4/digit-normalized solution keys were excluded against 3,536 historical local keys and prior retained families. This is an incomplete Sudoku-isomorphism audit and cannot exclude unknown training contamination. All variants stay within their source-family partition.

### 4.2 Learned-model instantiation

The hard checkpoint is frozen and run deterministically with argmax, CPU FP32, one thread and batch size one. Scoring uses FP64 log probabilities from retained logits. The adapter must match the native implementation bitwise on codes, all 12 accumulated states and final logits for all 48 inputs. Resuming from the cutoff without perturbation must match the native suffix; caches must be cleared after use.

The cutoff is after level 9, leaving levels 10–12. The primary mask contains cells blank in A, B and Aplus, excluding both added clues. Preferences between solutions use only differing-answer cells in that mask. References are never used to produce codes or choose a direction according to its observed effect.

For C, two distinct eligible full-state positions are chosen by a fixed seed. Independent Gaussian directions affect the 256 coordinates at one position each. Both signs are tested, giving four perturbations for each A/B anchor and 128 trials in total. Each direction has norm \(0.5\|z_9\|_F/\sqrt{89}\), approximately 5.3% of the global state norm, concentrated locally. The perturbations need not correspond to valid codes. Their dose is fixed without outcome-based escalation. Frozen and regenerated residuals are added sequentially in native FP32 order, not as a reassociated sum.

Reference-digit NLL is averaged over the common blank mask, then over four perturbations and both anchors within each source. Twenty thousand percentile-bootstrap draws resample the 16 whole families. Four co-primary contrasts d/r/i/s use 98.75% Bonferroni-adjusted intervals. The exploratory useful-effect margin is 0.01 nat per blank cell. Useful and specific recovery requires every adjusted lower bound to exceed that margin. Equivalence of mean frozen damage requires its entire adjusted interval inside ±0.01; an interval crossing zero alone does not establish equivalence.

### 4.3 Input-dominance diagnostic

For regenerated continuations, let h_ij be the mean log-probability preference for solution B over A, with input i and donor state j. Define

\[
u=\tfrac12[(h_{BA}-h_{AA})+(h_{BB}-h_{AB})],\quad
v=\tfrac12[(h_{AB}-h_{AA})+(h_{BB}-h_{BA})],
\]
\[
w=h_{BB}-h_{BA}-h_{AB}+h_{AA},\qquad g=h_{BB}-h_{AA}.
\]

Effects u and v concern input and state, respectively; w is an interaction contrast. The identity u+v=g makes their normalized ratios complementary, not independent resources or causal percentages.

The prespecified per-family diagnostic requires both native grids exact, g>0.1 nat, |w/g|≤0.2 and u/g in [0,1]. Within this domain, u/g≥0.8 is input-dominant, u/g≤0.2 is state-dominant, and an intermediate value is two sensitivities/mixture-or-unmodeled. A failed gate or nonmonotone ratio does not get forced into a class. All families remain in aggregate effect estimates. These margins instantiate functional tests on real logits; they are not unmodified exact-equality classification rules from the analytical library.

### 4.4 Autoregressive extension and its access contract

We additionally use the pretrained [GPT-2 small checkpoint](https://huggingface.co/openai-community/gpt2), revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, with 124,439,808 parameters. Inference uses CPU FP32, one thread, evaluation mode and eager causal attention in Transformers 4.57.1. No fine-tuning is performed. Sixteen manually chosen word pairs define prompts `The code is WORD. The code is`; A/B differ in one assigned-word token and Aplus changes the initial `The` to `My` without changing the target. All 48 prompts have eight tokens and a single-token reference with initial space. Exactness is the argmax over the entire vocabulary. Tokenization is checked before model outputs; no source is removed or replaced after observing performance.

We cut after block six of twelve. The exchanged state is the final prompt position's 768-dimensional residual vector. Earlier-token representations at every remaining block are retained from the recipient's native computation. Each block receives these representations and the current intervened last-position vector. Causality keeps earlier positions invariant. A tail consists of the last position's separate attention and MLP contributions from blocks 7–12, added in native FP32 order; the final normalization/head supplies the reader. Obtaining the context representations costs a prior computation; no speedup is claimed.

The intervention concerns depth within one next-token computation, not already-emitted-token time. It is a partial-state exchange with recipient memory retained, not a swap of all prompt information. This contract is explicit because it determines what an input-effect diagnostic can mean. The thirteen O readouts use the final reader on embeddings and every block output, without assuming intermediate calibration.

C tests two Gaussian directions on disjoint halves of the residual vector, both signs, at norm 0.1 times the state norm, giving 128 trials. E tests the same two 12-condition factorials for A/B and A/Aplus, giving 384 outcomes. Family aggregation, 20,000 bootstrap draws, four adjusted C intervals, the 0.01-nat margin and the E gates/ratios are unchanged. The margin is inherited rather than independently validated on text. Native failures cause abstention and remain in aggregate effects. All scripts, thresholds, data, weights and imported sources were locally sealed before inference at 2026-10-25T15:09:14Z, covering 1,291 files. The [full AR protocol](PROTOCOLE.md) records the contract and scope.

## 5. Results

### 5.1 Calibration checks the intended distinctions

The forced trajectory rule makes 52/344 correct singleton attributions, or 15.12%; its class-macro accuracy is 1/6. The cautious observation rule covers every true class but returns all six classes. The crossed diagnostic also covers every true class, with mean set size 2.395 and 52 correct singleton decisions. The complete diagnostic makes 344/344 correct singleton decisions, including 224/224 on reserved task families and mixtures.

The last number is expected by construction: program equations and diagnostics were developed together, and the library is public. It is a calibration check, not independent evidence of a generally accurate mechanism classifier. The 15.12% result is not a universal upper bound for trajectory-based methods. The exact matched observations establish non-identifiability for these constructed systems under the defined access regime.

### 5.2 Native performance is at ceiling

All 48 A/B/Aplus inputs are solved exactly at level 12. No earlier complete 89-token readout is exact, including separators. This does not imply absence of useful information in earlier codes. Corrective and degrading transitions occur across the intermediate trajectories and are retained for all transitions without selecting a favorable peak.

### 5.3 Local corruption creates no substantial repair opportunity

All five C conditions produce 128/128 exact final grids, preserving clues and separators. The primary mean frozen-damage contrast is d=1.74718×10^-6 nat, adjusted interval [-7.44527×10^-7, 6.31669×10^-6]. The corresponding results are r=-7.45874×10^-6 [-2.97070×10^-5, 1.23516×10^-7], i=9.47467×10^-7 [-2.83004×10^-7, 4.07239×10^-6], and s=9.67951×10^-7 [-2.92560×10^-7, 4.17891×10^-6].

The frozen-damage interval lies within the prespecified equivalence margin. This supports bounded mean robustness to the tested dose, not successful repair after material damage. Approximately 0.448% of suffix codes change, illustrating that code adaptation need not improve answer quality. No alternate dose or cutoff was selected after this result.

### 5.4 Input-dominant preference does not establish necessary adaptation

Across all families, the input effect is u=29.4676 nat, descriptive 95% interval [28.1586, 30.7572]; the state effect is v=3.1798 [2.1810, 4.1940]; the interaction is w=-0.7311 [-1.4021, 0.0092]. The effect of state is therefore not erased by the final answer counts.

The fixed rule diagnoses input-dominant response preference in 15/16 families. The remaining family has u/g=1.0069949 and remains nonmonotone/non-identifiable without rounding it into the majority. No internal-mechanism ground truth is assigned to any family.

After reciprocal A↔B state swaps, regenerated continuations produce the recipient solution in 32/32 cases and never the donor solution. Crucially, frozen native recipient continuations also produce the recipient solution in 32/32 cases, whereas frozen native donor continuations produce the donor solution in 32/32. Thus regeneration has no exact-answer advantage over the recipient-native frozen control in these swaps. That control was produced by a prior full native recipient run; it is neither free answer access nor an established inference shortcut.

All outputs remain complete valid Sudoku grids, but donor solutions violate the discriminating recipient clue. In A↔Aplus swaps, all 32 off-diagonal outputs under each of recipient-native, donor-native and regenerated continuation policies match the common reference. Small probability changes persist without changing the solution.

The exactness/clue/format measures were planned, but their separately aligned descriptive exporter was added after primary results. It does not change frozen contrasts, thresholds or labels. This reporting distinction is recorded in the [study journal](../benchmark_ncm_reel/JOURNAL.md).

### 5.5 Auxiliary analysis separates distinguishability from dominance

We performed a locally sealed but explicitly post-hoc analysis of saved codes, with no new model calls. For each family F, define a balanced selector V between its two unique inputs. On this two-atom design, the exact finite-design quantity I(V;R_k|F=f) is one bit when the two complete code vectors differ, and zero otherwise. This is not an estimated population information measure or a test of independently decodable semantics.

Level 5 is constant at code 11 and distinguishes no family. Nevertheless, levels 8 and 9 each distinguish both inputs in 16/16 families. The concatenated levels 1–9 and 10–12 each distinguish the branch in all families, so their finite-design conditional mutual information given family is one bit. Conditioning instead on the complete input yields zero analytically under deterministic argmax; the conditioning variables define different questions.

Distinct code vectors can reflect the changed clue without providing a reusable semantic decoder. Code distinguishability alone also does not guarantee distinguishability of the accumulated reconstruction: the code tables and transformations need not be injective. This result rules out an account based simply on absence of all pair identity in the tested coarse codes, not all forms of information loss. It also shows why dominance of a retained input cannot by itself establish an informational deficit in the earlier state.

All planned post-hoc sensitivity cutoffs are retained: 15/16 families meet input ratio 0.70 or 0.80; 8/16 meet 0.90 and 4/16 meet 0.95, with the same eligibility gates. The historical threshold remains 0.80. These counts expose the descriptive threshold dependence rather than redefine the principal result.

### 5.6 GPT-2: context dominance is compatible with autoregression

Official/native reconstruction and sham continuation agree exactly for all 48 prompts. Native accuracy is 12/16 on A, 8/16 on B and 13/16 on Aplus: 33/48 overall. Both A/B anchors are correct in eight families. Intermediate readings contain corrections and degradations: B is correct in twelve sources after block ten and eight at the final block. These are internal readouts, not revisions of emitted text.

For C, D=0.00181115 nat/token, adjusted interval [0.00141522, 0.00213673]; R=0.00781555 [0.00088758, 0.01519674]; I=0.00181959 [0.00113743, 0.00242062]; S=0.00168670 [0.00070116, 0.00268459]. Mean frozen damage falls within the fixed equivalence margin and the joint useful-and-specific recovery criterion is not met. Y00 and Y10 are correct in 80/128 trials; Y11, Y01 and Y1wrong in 78/128. A disclosed post-hoc paired exporter finds no argmax change under frozen corruption, but two correctness losses and no gains in each adapted-tail condition. Thus improved mean NLL under regeneration coexists with reduced exactness. The exporter uses saved outputs and changes no frozen diagnostic.

Mean E context effect u is 3.76857 nat, descriptive 95% interval [1.23271, 6.39620]; state effect v is 0.30567 [0.19423, 0.42402]; interaction w is 0.01608 [-0.21121, 0.22946]. The unchanged rule diagnoses input-dominant preference in **8/16 families**, while **8/16 remain non-identifiable due to native-anchor failure**. Every family passing the native gate meets the other input-dominance conditions; reporting only this eligible subset would hide half of the corpus. All sixteen effects are retained.

Across 32 off-diagonal A/B swaps, regenerated tails give 19 recipient-correct outputs, 10 donor-correct and three others. Recipient-native frozen tails give 18, 11 and three; donor-native frozen tails give nine, 20 and three. A paired descriptive export finds one gain and no loss for regeneration against the recipient-native control. This local difference does not establish a universal repair mechanism. In A/Aplus swaps, regenerated, recipient-native and donor-native tails give 25/32, 26/32 and 25/32 correct outputs. Identical references are not normalized into a response-preference ratio.

The AR [audit](results/verification.json) recomposes all 1,024 C/E decoder outcomes, recalculates 640 scalar NLLs and reproduces scores, grouped C intervals, labels, norms, diagonals and invariances. Explicit sublayer calls replay 16 C and 16 E continuations on the first two families, plus six official native forwards. States and logits match exactly. Historical manifests and frozen inputs remain unchanged. This is an internal software audit, not an independent empirical replication.

## 6. What the case studies validate

The learned-model application validates executable interventions, native-path parity and a protocol capable of returning bounded functional conclusions on FP32 states. It also catches two attribution traps: apparent success without a material corruption deficit, and regenerated success already matched by a cached native continuation. That is stronger evidence of practical applicability than analytical calibration alone.

It is not evidence that “input-dominant” is the true internal class of the NCM, that its generator recomputes from scratch, or that the protocol identifies every relevant mechanism. We do not have learned-model labels against which to score such accuracy. The frozen-recipient control does not establish that all arbitrary tails would work; it establishes sufficiency of particular tails on particular donor states. The dependence of confidence on state remains measurable.

The GPT-2 experiment adds a second learned architecture and a distinct task, with successful parity and crossed-outcome audits. It demonstrates that input-dominant preference can occur in an AR next-token computation when recipient context remains accessible. The usual append-only output interface does not determine this internal dependency. Nor is `frozen_recipient_native` an architecture class: it is a continuation policy applied in both studies.

The 8/16 AR and 15/16 NCM diagnostic counts cannot rank architectures. Task, native correctness, intervention geometry, retained memory and the meaning of the continuation axis differ. A future diffusion study likewise cannot assume a mixed signature before defining conditioning, state and remaining noise. No diffusion result is claimed here.

## 7. Coordination and common information: a conditional link

For discrete output variables Y_1,...,Y_m and shared side information P, define the conditional multivariate extension

\[
C_W(Y_1;\ldots;Y_m\mid P)
=\inf_{U:\,P(Y\mid U,P)=\prod_iP(Y_i\mid U,P)} I(Y;U\mid P).
\]

This combines established multivariate and conditional definitions; it is not a new quantity. For local producers with mutually independent private randomness conditioned on (U,P), their U is an admissible candidate, hence C_W≤I(Y;U|P)≤H(U|P). This bounds a discrete shared-information budget only when all shared resources and the local-readout restriction are accounted for. Exact finite synthesis and asymptotic approximate rate interpretations must also be distinguished; see [Exact Common Information](https://arxiv.org/abs/1402.0062). Input-dependent side information is not automatically the independent common randomness considered in [distributed channel synthesis](https://arxiv.org/abs/1208.4415).

Two elementary targets clarify the issue. The uniform even-parity distribution on n bits has C_W=n−1 bits under mutually conditionally independent production, although total correlation is one bit and pairwise information vanishes for n≥3. A uniform mixture of two solutions of one ambiguous Sudoku base has C_W=1 bit when at least two cells differ. A complete distinguishing clue reveals that branch; conditional on the resulting unique puzzle, C_W is zero. Proofs and finite enumeration checks are provided in the [theory note](../article_benchmark_coordination/THEORIE_WYNER.md).

In the executed deterministic NCM, both coarse and fine generated codes are functions of the full input, so I(coarse;fine|input)=0. In a future stochastic campaign, if per-position sampling is audited to use independent private draws, codes would factorize conditioned on input and previous state. That property would concern generated codes, not arbitrary codes produced by the frozen solution encoder, and not necessarily the final digits. A global decoder can create dependent output coordinates from independent code sources: a parity-computing decoder gives an immediate counterexample to applying the local-producer bound directly to NCM outputs.

The 15/16 functional diagnoses therefore measure neither C_W nor a common-information rate. They are consistent with the input acting as a shared conditioning signal, but do not establish minimality, coordination cost or compensation of pooling loss. The historical pooling result concerns an exact raw averaging symmetry and a specific empirical constant level; the 3a experiment did not manipulate pooling. The raw length-nine invariant concerns level 5, whereas the present level-9 cutoff has 37 positions. The solution encoder is not called by native generation. No causal chain from pooling to a need for shared information has been identified. The finite paired-code analysis shows that a constant level can coexist with branch-distinguishing coarse codes.

A nontrivial future test must specify a stochastic target distribution at fixed ambiguous input, audit the sampler, preserve output diversity, and separate local from global readout. Pooling would require its own matched intervention. These requirements convert the common-information connection into a testable program rather than an explanation asserted after observing input sensitivity.

## 8. Limitations, reproducibility and research status

Two learned checkpoints were used, one per architecture on different tasks, with sixteen source families per task. Both corpora are nonuniform; Sudoku clue counts and training relationships are incompletely controlled, and the word-copy sources are manually selected. The tasks are not a matched comparison, and half of the GPT-2 families fail the native-anchor gate. Swapped native states can be out of distribution conditional on the recipient input; Gaussian perturbations need not correspond to code substitutions. An envelope diagnostic measures amplitude, not density. Small-family bootstrap intervals are approximate and conditional on fixed weights. The experiment neither retrains the model nor measures a stochastic output distribution at fixed puzzle.

All protocols were frozen locally before their new computations, without an external immutable register. Analytical diagnostics and controls share an author; the neural and statistical audits were also implemented by the same research agent. A second agent checked the theoretical assumptions, not an independent empirical replication. No secret external-method evaluation, independently developed diagnostic or matched-task architecture comparison was performed. A second learned architecture is executed in this version, with an explicitly different access contract.

For a broader benchmark claim, the resource still requires independent methods, a matched-task learned comparison, explicit white-box access tracks, corruption calibration on separate development data and reserved evaluation families. Our data do not justify choosing a stronger successful corruption retrospectively. Novelty relative to established causal evaluation resources remains to be tested by use, not asserted by naming a protocol.

Reproduction instructions and tensor-axis conventions are retained with the [calibration](../benchmark_causal_revision/README.md) and [NCM study](../benchmark_ncm_reel/README.md). All raw outcomes and local freezes are preserved. The [original evidence ledger](../article_benchmark_coordination/PREUVES.md) maps calibration/NCM claims to retained files; the new [AR report](RAPPORT.md), [protocol](PROTOCOLE.md), [summary](results/summary.json) and [reproduction instructions](README.md) document this extension. The [journal](JOURNAL.md) distinguishes the frozen analysis from the later paired descriptive export. No visual presentation or public release was created for this manuscript.

## 9. Conclusion

Apparent revision, continuation change, recovery and necessary adaptation are separate observations. A useful evaluation protocol must represent their ambiguities and include frozen continuations rather than reward final accuracy alone. Our constructed calibration, frozen-NCM application and autoregressive extension show how this can be done reproducibly, including cases where the experiment remains uninformative about repair or abstains because native anchors fail. Input dominance is a scoped functional diagnosis, while common-information theory supplies assumptions for a future coordination experiment. Keeping those statements separate is necessary for a scientifically valid account of the evidence.

## References

- Mueller, A., et al. (2025). [MIB: A Mechanistic Interpretability Benchmark](https://proceedings.mlr.press/v267/mueller25a.html). ICML, PMLR 267, 45069–45108.
- Tyen, G., Mansoor, H., Carbune, V., Chen, P., and Mak, T. (2024). [LLMs cannot find reasoning errors, but can correct them given the error location](https://aclanthology.org/2024.findings-acl.826/). Findings of ACL, 13894–13908.
- Kumbong, H., et al. (2026). [Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation](https://arxiv.org/abs/2610.14162v1). arXiv:2610.14162v1.
- Wyner, A. D. (1975). [The Common Information of Two Dependent Random Variables](https://doi.org/10.1109/TIT.1975.1055346). IEEE Transactions on Information Theory 21(2), 163–179. Original publisher full text was not directly available during this audit; the definition was cross-checked against the primary extensions below.
- Liu, W., Xu, G., and Chen, B. (2010). [The Common Information of N Dependent Random Variables](https://arxiv.org/abs/1010.3613). Allerton Conference.
- Graczyk, R., Lapidoth, A., and Wigger, M. (2022). [Conditional and Relevant Common Information](https://www.isiweb.ee.ethz.ch/papers/docu/rgra-alap-mwig-2022-2.pdf). Information and Inference. DOI: 10.1093/imaiai/iaab021.
- Cuff, P. (2013). [Distributed Channel Synthesis](https://arxiv.org/abs/1208.4415). IEEE Transactions on Information Theory 59(11), 7071–7096.
- Kumar, G. R., Li, C. T., and El Gamal, A. (2014). [Exact Common Information](https://arxiv.org/abs/1402.0062). IEEE ISIT, 161–165.

- GPT-2 checkpoint and model card. [openai-community/gpt2](https://huggingface.co/openai-community/gpt2), fixed revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.

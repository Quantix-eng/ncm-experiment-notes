# Matched Tasks, Repeated Training and Explicit Access Contracts

Empirical extension, 25 October 2026. This addendum accompanies the [consolidated manuscript](PAPER_MATCHED_TASK.md) and the [French report](RAPPORT.md). The earlier NCM and GPT-2 experiments remain unchanged.

## Abstract

We extend revision-attribution evaluation to two small learned Transformers on the same sequence-rotation task, with paired initializations, training examples and optimization budgets across three seeds. Data are split into 140 training, 30 development and 40 evaluation orbits, keeping rotations and reversals together. All six models solve all 120 held-out variants, including full free generation. A development-only corruption grid fails to meet its prespecified common-damage criterion; the fixed fallback is retained and disclosed. At the primary cutoff after two blocks, regenerated continuations on exchanged query states always produce the donor's first symbol for both systems. At a prespecified earlier cutoff, recipient-correct outcomes vary by seed and access contract. Transferring the whole memory removes the recipient path under regeneration exactly as specified. Many strict functional labels abstain despite perfect native accuracy, because small negative input ratios fall outside the announced domain. The study controls task and training replication more closely than the preceding cross-task comparison, while revealing calibration and labeling limitations. It does not establish an architecture taxonomy or internal-mechanism ground truth.

## 1. Motivation and experimental object

The preceding NCM/Sudoku and GPT-2/text comparison changed task, native performance, state geometry and retained conditioning simultaneously. We therefore add a common-task study and vary information access on the same trained weights. This is a controlled comparison of two specified training systems; it is not a pure causal effect of architecture, because autoregressive teacher forcing and fixed parallel queries differ during learning.

The parallel system is a conditional Transformer, not a miniature NCM: there is no learned tokenizer, VQ-VAE or concept hierarchy. The primary intervention concerns computation of the first output symbol, so neither model receives a correct output history at test time. Full-sequence free-generation accuracy is reported separately.

[Causal abstraction work](https://papers.nips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html) motivates specifying the variables aligned by an interchange intervention. [Activation-patching methodology](https://openreview.net/pdf?id=Hf17y6u9BC) motivates explicit corruption and metric choices. We claim neither priority for interchange nor a new causal identification theorem. The present contribution is an executable evaluation design with declared limitations.

## 2. Models, data and prospective boundaries

Inputs contain a sequence of four distinct symbols from an alphabet of eight, a rotation bit b and a nuisance bit n. The target is a left rotation by b∈{0,1}; n leaves the target unchanged. A/B change b and Aplus changes n. The first target always differs between A and B. Accuracy uses all eight output symbols.

All 1,680 input permutations are grouped into 210 rotation/reversal orbits. Before training, a fixed seed assigns 140 to training, 30 to development and 40 to evaluation. Every orientation and variant of an orbit stays in its split. Training expands its orbits into 4,480 examples. Development and evaluation use one orientation, selected before outputs, with three variants. Symbols and the task rule are shared; this is held-out configuration generalization within one simple task, not unseen-alphabet or unseen-task generalization.

Both systems have 77,576 parameters: four pre-normalized Transformer blocks, width 48, four heads, MLP width 96 and no dropout. Context positions are causal and cannot attend to output queries in both. AR queries are causal and receive shifted targets during training; parallel queries use fixed tokens and can attend to each other. Seeds 17,29,43 share initial learned parameters and batch indices across each AR/parallel pair. Every model receives 1,200 AdamW updates, batch 64, fixed warmup/cosine schedule and the final checkpoint without validation selection.

The entire experimental implementation and analysis were locally sealed before training at 2026-10-25T15:38:19Z, covering 785 files. Development tests relative corruption doses 0.1,0.3,1,3. A common dose must, separately for each system averaged over its three seeds, cause a 10–40 percentage-point frozen-accuracy drop and at least 0.05 nat mean loss increase. If none qualifies, the declared fallback is 3 and the calibration is reported as unsuccessful. Final weights and this decision were sealed at 2026-10-25T15:40:53Z before evaluation. These are local boundaries, not independent external registration.

## 3. States, contracts and outcomes

The primary cutoff follows block two; block one is a declared descriptive secondary analysis. The state has seven context positions P and four query positions Q. Regime O reads embeddings and four block states using the final reader. C perturbs query positions on two disjoint feature-half supports, both signs, and stores the five crossed outcomes and separate attention/MLP residual contributions.

For E, the **retained-context** contract regenerates from (P_recipient,Q_donor). The **whole-memory** contract regenerates from (P_donor,Q_donor), with no other recipient tensor. Both execute two inputs × two donor states × three tail policies for A/B and A/Aplus. Frozen tails are native-left or native-right contributions, not imposed answer symbols.

Whole-memory regeneration has a limited construction truth: Y_ij=Y_j, hence u=0, w=0 and v=g. This checks a blocked information path on learned networks; it is not accuracy against a known internal algorithm. A frozen recipient-native tail is a different intervention that introduces recipient information from its earlier native run. Its success is therefore compatible with recipient invariance under regeneration. The [access note](CONTRATS_ET_ESTIMANDS.md) makes this distinction explicit.

C loss contrasts, the 0.01-nat margin and the four adjusted intervals are unchanged from earlier studies. Each model/cut aggregates two anchors and four perturbations within each of 40 source families, then bootstraps whole families 20,000 times. Seeds are reported separately rather than pooled into an artificial sample of 120 independent sources. The same E gate and dominance ratios are retained, including abstention for nonmonotone ratios and interactions. Block-one findings are descriptive and do not replace primary results.

## 4. Results

### 4.1 Native success does not ensure an informative corruption

Every model solves all 40 A, 40 B and 40 Aplus examples at the first symbol and full freely generated sequence: 120/120 per training. Thus native-anchor failure no longer explains abstention in this experiment. The task remains easy and synthetic.

No candidate dose passes the common-damage criterion on development. At dose 3, AR has mean NLL damage 0.00164032 nat and zero accuracy loss; parallel has 0.05683645 nat and a 0.278-percentage-point loss, far below the required ten points. We retain the fixed fallback without widening the grid. Consequently this campaign does **not** establish a successfully calibrated common informative corruption.

On evaluation at the primary cutoff, frozen/adapted exactness over 320 trials is 320/237, 320/267 and 320/210 for the three AR seeds; 316/237, 316/244 and 317/255 for the parallel seeds. Mean recovery contrasts R are respectively −0.801597, −0.556550, −1.035251 and −0.670127, −0.596030, −0.532856 nat. Each model's adjusted R interval is wholly negative. I and S means are also negative at all six primary cells. The three AR frozen-damage means are equivalent within the 0.01 margin; parallel damage means range from 0.058768 to 0.072958 nat. No model meets the useful-and-specific recovery criterion. Regeneration can amplify perturbations better tolerated by a previously computed frozen tail, conditional on this large dose and its unsuccessful calibration.

### 4.2 Primary exchanged states follow the donor

After block two, retained-context regeneration produces the donor's first symbol in all 80 off-diagonal A/B swaps for every model, with zero recipient-correct outputs. Donor-native frozen tails also give 80/80 donor outputs. Recipient-native frozen tails give 80/80 recipient outputs for every AR seed, and 56/80,44/80,36/80 for the parallel seeds. Regeneration therefore provides no recipient-exactness gain against these controls. Frozen tails required a prior full native computation; no efficiency claim follows.

The strict retained-context diagnostic labels 20,25,27 of 40 families state-dominant for AR seeds 17,29,43 and abstains as nonmonotone in the remaining 20,15,13. Parallel counts are 18,17,28 state-dominant and 22,23,12 nonmonotone. Native anchors are all exact. Negative input ratios are preserved even when close to zero; continuous effects and uniformly donor-following argmax outcomes remain visible alongside the strict labels. Abstention here is neither evidence of task incompetence nor a complexity theorem.

### 4.3 Earlier cutoffs and access contracts change the result

At the declared secondary cutoff after one block, retained-context regeneration gives 0/80,80/80,0/80 recipient outputs across AR seeds and 0/80,40/80,80/80 across parallel seeds. These seed effects are reported in full. AR 29 has 20 input-dominant and 20 nonmonotone families; parallel 43 has 37 input-dominant and three nonmonotone families. Parallel 29 fails the interaction gate in all 40 families; its intermediate output counts are not classified as a known mixture mechanism. The remaining models have mostly state-dominant or abstaining labels, detailed in the full report.

Recipient-native frozen tails already give 80/80 recipient answers for all six models at this earlier cutoff. Early regenerated success does not establish its necessity for exactness. In AR 29 and parallel 43, identical weights switch from recipient-following early regeneration to donor-following regeneration after block two. This is a boundary-dependent functional result, not a universal architecture signature.

Under whole-memory regeneration, every model/cut exactly reproduces donor-native logits for every recipient/donor combination. The receiver effect and interaction are exactly zero; each cell gives 40/40 state-dominant labels under the correct native anchors, as expected by construction. Early AR 29 and parallel 43 therefore also change from recipient-following to donor-following behavior when the access contract changes on the same weights. Other seeds were already donor-following, so that access effect is not asserted to be universal in the retained track.

## 5. Audit and implications

The sealed audit passes 42,240 exact residual/decoder recompositions, 576 explicit sublayer regenerations, 19,200 scalar NLL checks, 72 native replays and 3,840 blocked-path conditions. It also checks orbit partitions, paired initializations and batches, final checkpoints, dose selection, C contrasts and intervals, labels, exactness and paired gains/losses. All frozen and historical files remain unchanged. These counts include repeated conditions and are not independent statistical units. The same agent implements and audits the study; no external replication is claimed.

The resource now includes common-task learned comparisons, repeated training, a separate corruption-development boundary and explicit access controls. It also exposes three weaknesses: failed informative-corruption calibration, dependence on cutoff and initialization, and strict-label abstention despite perfect native task accuracy. Keeping these failures makes the resource more auditable; it does not automatically establish novelty, publication suitability or broad classifier validity.

A next study should calibrate natural-state and readout-damaging corruptions on separate development data before new evaluation, declare the information and optimization used to construct them, and test alternative abstention rules prospectively against continuous effects. Independently developed attribution methods and outside evaluation remain needed. Larger tasks and a faithful NCM trained at common task are not implicit in these small Transformers. No new Wyner quantity or pooling-compensation claim is supported by this deterministic experiment.

Data, code, manifests, per-seed effects and full outcomes are linked from the [README](README.md), [protocol](PROTOCOLE.md), [summary](results/summary.json), [journal](JOURNAL.md) and [audit](results/verification.json).

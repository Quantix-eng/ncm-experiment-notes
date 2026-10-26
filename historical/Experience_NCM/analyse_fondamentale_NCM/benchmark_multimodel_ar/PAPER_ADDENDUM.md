# Extending Revision Attribution to an Autoregressive Transformer

Empirical addendum, 25 October 2026, to [Attributing Apparent Revision in Parallel Generators](../article_benchmark_coordination/PAPER.md). The original manuscript and experiments are retained. This addendum reports a new, locally frozen experiment on GPT-2; it does not convert a functional label into internal-mechanism ground truth.

## Abstract

We extend a revision-attribution protocol from constructed controls and a frozen Sudoku NCM to a pretrained autoregressive Transformer. GPT-2 small is evaluated on 16 paired word-copy sources, each with an answer-preserving prompt variant. Interventions exchange or perturb the final prompt token's residual state after block six while retaining the recipient's earlier-token context in the remaining layers. Native reconstruction and sham continuation match the official model exactly on all 48 inputs. The model answers 33/48 prompts correctly. With the original functional thresholds unchanged, eight source families exhibit input-dominant response preference and eight remain non-identifiable because a native anchor fails. Across 32 reciprocal state swaps, regenerated and recipient-native frozen continuations give 19 and 18 recipient-correct outputs, respectively. Local perturbations produce a mean frozen loss increase within the prespecified equivalence margin; regeneration improves mean NLL but loses two correct answers. The experiment establishes an executable second-architecture application and illustrates why architecture names, frozen-continuation policies and functional labels must be separated. Different tasks and access contracts preclude attributing the NCM/GPT-2 comparison to architecture alone.

## 1. Scope of the extension

An autoregressive interface normally appends tokens without revising previously emitted text. This constraint does not specify how later layers compute a future token from a residual state and retained context. It also does not imply that an AR model belongs to a class named after a frozen continuation. `frozen_recipient_native` is a policy that our protocol can apply to either architecture.

We therefore test the protocol at an explicit within-forward cutoff. The intervention axis is network depth, not emitted-token time. This extends the implemented models and tasks while preserving the distinction between readout change, recovery and necessary adaptation. It is not a matched-task architecture experiment or a test of already-emitted-token revision. No diffusion model is evaluated.

## 2. Model, task and access contract

We use the pretrained [GPT-2 small checkpoint](https://huggingface.co/openai-community/gpt2) at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, with 124,439,808 parameters. The model is frozen, in evaluation mode, on a single CPU thread with FP32 inference. Transformers 4.57.1 uses eager causal attention; scores are computed in FP64. No fine-tuning or outcome-based model selection is performed.

Sixteen pairs of single-token words are specified before model outputs. Prompts follow `The code is WORD. The code is`. A and B differ in the assigned word at one input token. Aplus replaces the initial `The` with `My`, retaining the reference and sequence length. All 48 prompts have eight tokens and a single-token target including its initial space. Accuracy is the global vocabulary argmax, rather than forced choice between the two references. These manually selected copy tasks are neither natural text samples nor difficulty-matched Sudoku sources. Their references are defined by the evaluation task, not by a claim that a language model assigns zero probability to other continuations.

The cutoff is after block six of twelve. The mutable state is the 768-dimensional residual vector at the last prompt position. Earlier-token representations at every remaining block belong to the recipient prompt. Causality ensures they do not depend on the last position. Each resumed block receives these fixed recipient representations and the current intervened last-position vector. Its attention and MLP contributions are saved separately and added in native order. The final layer normalization and language-model head provide the readout.

This is a partial-state intervention with retained context, not a transfer of the complete prompt memory. Obtaining the retained representations requires a native computation; no inference saving is asserted. The contract is analogous to preserving a conditioning channel in the NCM adapter, but the states, tasks and continuation axes differ. A whole-memory swap would define another access regime.

The [protocol](PROTOCOLE.md), data, weights, scripts and imported sources were locally sealed before new inference: 1,291 files at 2026-10-25T15:09:14Z. There was no external registration. All sources, including native failures, are retained. The layer, corruption dose, metrics and functional thresholds were not adjusted after outcomes.

## 3. Interventions and analysis

Regime O reads all thirteen states: input embedding and the twelve block outputs, using the final reader throughout. Such readouts need not be calibrated intermediate predictors and are not emitted tokens. Regime C uses the original five crossed outcomes: native state/native tail, perturbed state/frozen tail, perturbed state/adapted tail, native state/adapted tail, and perturbed state/tail adapted to another perturbation. Two Gaussian directions have disjoint supports of 384 coordinates each, both signs, and norm 0.1 times the anchor state's norm. This yields 128 trials over 32 A/B anchors. The dose is chosen without observing damage and is not escalated.

NLL contrasts D, R, I and S retain the original definitions. We average four perturbations and two anchors within each family, then use 20,000 whole-family bootstrap resamples. Four primary contrasts use 98.75% intervals and the exploratory margin 0.01 nat/token. The margin is inherited, not independently validated as practical utility for this task.

Regime E executes the full 2 contexts × 2 donor states × 3 tail policies for both A/B and A/Aplus: 384 conditions. Tails are native-left, native-right and regenerated. Recipient- and donor-native results are then aligned on the off-diagonal swaps without selecting new outputs. The tails contain continuous neural contributions, not replacement output tokens.

For A/B, response preference is log p(B-word) minus log p(A-word). With recipient context as the first index and donor state as the second, define u, v, w and g as in the original manuscript. The unchanged descriptive gate requires two exact native anchors, g>0.1, |w/g|≤0.2 and u/g in [0,1]. Input dominance requires u/g≥0.8; state dominance requires u/g≤0.2. Otherwise, report two sensitivities or the appropriate abstention. All families remain in aggregate effects. Neither this heuristic nor u+v=g identifies an internal program or a causal resource fraction.

## 4. Results

### 4.1 Native and crossed outcomes

Official/native readout, residual reconstruction and sham resumption agree exactly on all 48 inputs. Native accuracy is 12/16 on A, 8/16 on B and 13/16 on Aplus, or 33/48 overall. Eight families have both A/B anchors correct. Intermediate readouts show corrections and degradations: for B, twelve sources are correct after block ten, versus eight at the final block. This is readout variation within one next-token computation, not revision of previously emitted text.

Under C, mean D is 0.00181115 nat, adjusted interval [0.00141522, 0.00213673]; R is 0.00781555 [0.00088758, 0.01519674]; I is 0.00181959 [0.00113743, 0.00242062]; S is 0.00168670 [0.00070116, 0.00268459]. Mean frozen damage is equivalent within the fixed margin. The joint useful-and-specific recovery criterion is not met.

Y00 and Y10 each give 80/128 correct responses. Y11, Y01 and Y1wrong each give 78/128. A disclosed post-hoc paired exporter confirms zero argmax changes under frozen perturbation and two correctness losses, without gains, in each of the three adapted-tail conditions relative to native. Thus a favorable mean NLL recovery contrast can coexist with reduced accuracy and does not establish useful repair at the prespecified margin. The exporter was added after the primary outcomes; it uses saved logits only and changes no threshold or label.

### 4.2 Input dominance and frozen continuations

Across all sixteen families, mean context effect u is 3.76857 nat, descriptive 95% interval [1.23271, 6.39620]; state effect v is 0.30567 [0.19423, 0.42402]; interaction w is 0.01608 [-0.21121, 0.22946]. The unchanged rule labels **8/16 families input-dominant** and **8/16 non-identifiable due to native-anchor failure**. All eight families eligible through native exactness meet the remaining input-dominance conditions. Reporting only that conditional subset would hide half of the tested sources; we retain the full denominator and every family effect.

On 32 reciprocal A/B state swaps, regenerated tails give 19 recipient-correct outputs, 10 donor-correct and 3 other outputs. Frozen recipient-native tails give 18, 11 and 3; frozen donor-native tails give 9, 20 and 3. A post-hoc paired count identifies one gain and no loss for regeneration versus the frozen recipient control, in `copy-013` with recipient B. This local gain does not establish a general repair mechanism or an architecture class. Native A/B accuracy was only 20/32, so the interpretation also differs from the NCM ceiling case.

For answer-preserving A/Aplus swaps, regenerated, recipient-native and donor-native tails give 25/32, 26/32 and 25/32 correct outputs. Probability distances and NLL differences are retained without applying a preference ratio to identical references. A formulation change can affect model performance while preserving the task's intended answer.

## 5. What this changes in the benchmark claim

The resource now contains executed applications to two learned architectures and two tasks, in addition to constructed calibration. The GPT-2 adapter passes native-parity and crossed-continuation checks on real FP32 states. It also preserves abstention when a native anchor is wrong and retains a case where mean confidence recovery and accuracy move differently. This addresses part of the original single-learned-family limitation; external evaluation and independently developed attribution methods remain absent.

The proposed mapping “AR follows its frozen tail, NCM is input-dominant, diffusion is mixed” is not an admissible ground-truth taxonomy. A tail policy and a functional label are different objects. Moreover, an AR next-token computation can exhibit input-dominant preference when recipient context is retained. This observation does not contradict append-only generation, which concerns the output interface. It also does not make the current diagnostic a correct classifier of arbitrary AR mechanisms.

Comparing 8/16 GPT-2 diagnoses with 15/16 NCM diagnoses would confound task, native accuracy, intervention geometry, depth and retained information. No architectural superiority or universal separation follows. A future matched-task study should calibrate corruptions on separate development data and evaluate explicit context-retention and whole-memory tracks. Diffusion would additionally require a defined conditioning and remaining-noise contract; no label should be assigned before that experiment.

The common-information conclusions of the original manuscript remain unchanged. This deterministic next-token experiment neither measures a Wyner rate nor manipulates pooling. A retained context is a shared conditioning resource; calling it one is not proof of minimal coordination information.

## 6. Audit and availability

The [audit](results/verification.json) independently recomposes 1,024 saved C/E decoder outcomes, recalculates 640 NLLs with scalar log-sum-exp, and reproduces contrasts, bootstrap intervals, labels, exactness, perturbation norms, diagonals and fixed-tail invariances. It replays 16 C and 16 E continuations through explicit attention/MLP calls on the first two families, and six official native forwards. All compared states and logits match exactly. Frozen inputs and the historical NCM/manuscript manifests remain unchanged. These are internal software checks by the same research agent, not external empirical replication.

The [data](data/families.json), [summary](results/summary.json), raw arrays, [journal](JOURNAL.md), implementation and [reproduction instructions](README.md) are retained locally. The single layer and dose, manually selected small corpus, native task failures, uncalibrated intermediate reader and potentially out-of-distribution states limit generalization. No secret test, new training or diffusion experiment was performed.

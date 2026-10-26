# Recovery and Perturbation Matching in GPT-2: A Controlled IOI Case Study

Revision of 25 October 2026. The main text now addresses one attribution question on IOI. Complete numerical records and earlier studies remain in appendices. Figures and an explicitly post-hoc margin display use retained results; no model was run and no historical decision rule was changed.

## Abstract

Does recovery after an intervention reflect a useful change in the remaining computation, and does matching that change to the intervention add an advantage? We study these questions in GPT-2 small on 512 reserved indirect-object-identification families. We replace one Name Mover head's output with a native donor activation and cross the perturbed state with native, adapted and other-donor continuations. For the primary head, frozen-continuation damage is 2.752 nats and adaptation recovers 1.910 nats. Replaying the native outputs of published downstream backups raises loss more than a layer-matched control, supporting their contribution to recovery. Matching the continuation to its donor adds only 0.038 nat. Two declared secondary heads also recover, with matching advantages of 0.025 and 0.008 nat. The latter estimate is positive but its interval spans the historical 0.01-nat margin; it is not evidence of absent specificity. We expose the dependence of such decisions on conventional margins instead of treating them as mechanism labels. Two boundary checks limit the scope: a Pythia final-layer experiment separates reader changes from block adaptation, and an InterpBench prediction test establishes no incremental advantage from adding the mismatched-continuation condition. The contribution is a controlled IOI case study separating recovery, donor matching and component involvement, not a general benchmark of internal mechanisms.

## 1. Introduction

An ablated computation can recover the correct answer without revealing how the recovery occurred. Later blocks may compensate for the intervention, a fixed continuation may already suffice, or the final reader may change its response to the altered residual. Final correctness alone does not identify which later computation was useful.

We focus on one question: **what does matching an adapted continuation to an intervention add beyond recovery itself?** The indirect-object-identification (IOI) circuit in GPT-2 provides a concrete test. Prior work identifies Name Mover heads and backup heads that can support their function after ablation. We resample a Name Mover output, measure the damage exposed by holding the continuation fixed, and then release that continuation. We also exchange adaptations between donors and separately prevent candidate backups from adapting.

The result is a separation of scales. At the primary L9H9 target, adaptation recovers 1.910 nats of a 2.752-nat deficit. Its advantage over another donor's adaptation is 0.038 nat. The secondary L10H0 target recovers 0.425 nat, although its donor-matching advantage is only 0.008 nat. Both quantities can be positive without supporting the same claim. Blocking backups adds evidence about participating components that neither contrast supplies by itself.

Our contribution is a controlled case study, with direct self-repair antecedents, rather than a new discovery of IOI compensation. An InterpBench test also establishes no incremental prediction advantage from the extra matching condition. Earlier NCM, copying and synthetic Transformer studies are retained in Appendix B; the separate common-information argument is in Appendix E. They motivate controls but do not supply independent replications of the central IOI claim.

## 2. A crossed-continuation test on IOI

### 2.1 Five combinations, four distinct questions

A continuation is the ordered collection of remaining attention and MLP contributions to the final-position residual. A **native tail** was computed before intervention; an **adapted tail** is recalculated afterward. The reader is final LayerNorm and unembedding. Figure 1 gives the design in words, while Appendix A retains the algebra and interaction contrast.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/01_crossed_design.pdf}
\caption{Cross the state with the continuation. Frozen versus adapted tails at the same perturbed state measures recovery. A tail generated for another donor tests the matching advantage. The native state with an adapted tail completes the interaction control. These are counterfactual combinations of retained contributions, not five independent model samples.}
\label{fig:design}
\end{figure}

**Damage** is the increase in reference-token negative log likelihood (NLL) under a perturbed state with the native tail. **Recovery** is the reduction obtained by replacing it with the adapted tail. **Matching advantage** compares this adaptation with one generated for another donor at the same site. **Selective block effect** compares replaying native outputs of announced backups with replaying a layer-matched control group. Lower NLL is better.

Recovery establishes that a changed continuation helps under the declared intervention. Matching asks whether the actual donor's adaptation is better than the specified alternative. It does not identify a unique representation of error identity. A fixed response to any perturbation could recover one error without a matching advantage. Conversely, a small matching advantage need not make the larger recovery effect spurious.

In IOI the same reader rule is applied to every combined residual, with its normalization factor recomputed for that residual. Reader changes are therefore included in the output contrast. The separate Pythia check varies that factor explicitly, and the backup intervention tests changed block contributions separately.

### 2.2 Data and intervention boundary

We use frozen GPT-2 small, revision \texttt{607a30d783dfa663caf39e06633721c8d4cfcd7e}, on public [MIB-IOI](https://huggingface.co/datasets/mib-bench/ioi) data. Input-only selection fixes 128 development and 512 test families. Each contains a native prompt and answer-changing and answer-preserving counterfactuals. Candidates are single tokens with initial spaces; variants have equal lengths. No selected source is removed for an incorrect native answer or unfavorable effect. Nine literal templates occur in each partition, with none shared between development and test. We explicitly prepend GPT-2 BOS, without assuming equivalence to the complete MIB pipeline.

The primary intervention replaces L9H9's 64-dimensional pre-projection output at the last prompt position by its native value from another native prompt. Other heads in that attention layer stay native. The boundary follows the complete attention update and precedes its MLP. Recalculation includes that MLP and the remaining blocks, retaining the recipient's earlier-position representations. Contributions are added in native FP32 order. L9H6 and L10H0 are declared secondary targets with corresponding boundaries.

Two donors are assigned by cyclic offsets 37 and 73 within each partition. They are distinct non-self donors, selected without reference to names, norms or effects. The mismatched tail comes from the other donor. Thus the site is matched, but displacement norm and difficulty are not equalized.

The development gate requires mean frozen damage of at least 0.05 nat, damage above 0.01 nat in at least 25% of families, and passing software checks. It passes with 2.462-nat damage and 127/128 families above 0.01 nat. Test evaluation then retains all 512 families.

### 2.3 A separate backup intervention and grouped inference

We prevent published strictly downstream backups from adapting by replaying their native outputs. For layer-9 targets, the group is L10H1/H10/H6/H2 and L11H9/H2. Controls are L10H0/H3/H4/H5 and L11H0/H1, matching count and layer distribution. Simultaneous layer-9 heads are not downstream consequences. For L10H0, only the two layer-11 backups and two controls remain downstream.

Controls are not assumed inert. Corroboration requires that replaying backups increases loss relative to the adapted computation, as well as increasing it more than controls. A relative difference alone could otherwise reflect beneficial controls. This establishes a group contribution under the intervention, not individual necessity or a complete circuit explanation.

Twenty thousand percentile-bootstrap resamples group both donors within each source. Primary damage, recovery, matching and selective-block contrasts retain their historical 98.75% intervals; secondary-head intervals at that level are descriptive. Shared donors induce additional dependence, so intervals are conditional approximations, not guaranteed population coverage. The fixed sample size was not derived from a prospective power calculation. Vocabulary exactness, candidate-name exactness and centred-logit self-repair are reported separately. Official-forward audits are internal software checks.

## 3. Results: large recovery and smaller matching effects

### 3.1 Recovery and backup involvement

For L9H9, frozen damage is **2.752 nat** [2.657, 2.846], and recovery is **1.910 nat** [1.814, 2.006]. Remaining regenerated damage is 0.842 nat. Vocabulary exactness rises from 59/1,024 frozen interventions to 622/1,024 adapted interventions, against 952/1,024 in the repeated native condition. These are 512 source families with two donors, not 1,024 independent native examples.

Both secondary targets recover. L9H6 has 1.205-nat damage and 1.049-nat recovery [0.975, 1.125]; exactness rises from 428 to 904/1,024. L10H0 has 0.706-nat damage and 0.425-nat recovery [0.389, 0.461]; exactness rises from 654 to 867/1,024.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/02_ioi_recovery.pdf}
\caption{IOI effects on the same 512 families. (a) Adaptation recovers part of frozen-tail damage. (b) Replaying backups raises NLL relative to the adapted computation, more than the matched controls. Bars are retained conditional 98.75\% intervals; secondary targets remain descriptive. Panels use different horizontal scales. Layer-9 targets have six downstream backups and L10H0 has two; block effects do not rank circuit importance across boundaries.}
\label{fig:recovery}
\end{figure}

Replaying backups after L9H9 raises NLL by 0.251 nat relative to adaptation, compared with 0.040 nat for controls. The selective difference is **0.211 nat** [0.189, 0.235]. For L9H6, absolute increases are 0.119 and 0.065 nat, giving 0.054 nat [0.042, 0.067]. For L10H0, they are 0.133 and 0.010 nat, giving 0.123 nat [0.108, 0.139]. Both absolute and selective comparisons support a contribution from the announced groups, consistent with prior IOI compensation work.

### 3.2 Positive matching is not the same as exceeding a practical margin

The matched-donor advantage is **0.0380 nat** [0.0240, 0.0540] for L9H9, 0.0246 nat [0.0170, 0.0349] for L9H6, and **0.0082 nat** [0.0060, 0.0109] for L10H0. All three retained intervals lie above zero. The late target supports a positive matching effect under this design, while its interval does not establish an effect exceeding the conventional 0.01-nat margin.

The historical L10H0 decision remains a failure to exceed that margin, not a failure to recover or evidence of absent specificity. Centred-logit self-repair is positive at all targets: 2.355, 1.538 and 0.678. These differently defined logit quantities need not imply a large matching advantage. Their disagreement with a threshold decision does not identify a flaw in the published measure. Appendix A retains full precision, all exactness counts and broader state swaps, which behave differently from one-head replacements.

## 4. Interpreting the conventional choices

Fixing a rule before evaluation limits outcome-driven selection. It does not establish practical significance. The historical margins, donor assignments and corruption grid were operational choices, not values derived from first principles. We retain their decisions but lead with continuous effects.

### 4.1 Effect scale and post-hoc margin sensitivity

A mean reference-token NLL improvement of \(m\) nat corresponds to a geometric-mean probability ratio \(e^m\). Thus 0.01 nat increases the geometric-mean reference probability by about 1.01\%, and 0.05 nat by about 5.13\%. This explains the scale; it does not justify a minimum useful improvement or an accuracy gain.

The 0.01-nat margin was inherited across tasks with different tokens and averaging masks. It was not anchored to independent utility, a minimum resolution gain or a power calculation. The 0.05-nat damage margin and 25\% development gate likewise screen for repair opportunity by convention. IOI damage and recovery lie far from those margins; interpretation of matching is more sensitive.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/03_margin_sensitivity.pdf}
\caption{Continuous matching effects and post-hoc decision sensitivity. (a) Retained means and 98.75\% intervals; the dashed line marks the historical 0.01-nat margin. (b) Yes means the same interval's lower endpoint exceeds the candidate margin. L10H0 changes decision between 0.005 and 0.01 nat without changing its estimated effect. Candidate margins are an interpretive display added after results, not replacement tests. Shared-donor and secondary-analysis qualifications still apply.}
\label{fig:margins}
\end{figure}

The L10H0 lower endpoint exceeds 0.005 but not 0.01 nat. At 0.02 nat only L9H9 clears the margin; at 0.03 nat none does. Figure 3 makes these arithmetic comparisons with unchanged intervals explicit. It is neither a new confirmatory grid nor a post-hoc power analysis. Finite-sample effects and uncertainty carry more information than one pass/fail label.

### 4.2 Donor policy, unanimity and the failed fallback calibration

Offsets 37 and 73 produce reproducible, distinct non-self donors. The particular integers have no causal, semantic or optimality interpretation. They prevent effect-based selection but do not establish robustness to new offsets, name- or norm-matched donors, or an independent donor bank. Those alternatives have not been run. Matching remains conditional on this policy.

InterpBench's threshold 1.00 means every included probe agrees with the native categorical output. It is an exact unanimity predicate, not 100\% confidence or an optimized risk level. The declared 0.50 and 0.75 thresholds remain visible, and equal score thresholds do not mean equal coverage.

The corruption fallback 3 belongs to the earlier synthetic experiment, not IOI. It was the largest point in a fixed grid, after no candidate met the common-damage criterion. Executing the fallback did not create an informative calibrated repair test. There is no first-principles argument that makes 3 informative after that failure. Appendix B retains the negative calibration and supplies no positive repair or general AR-versus-parallel claim.

## 5. Two boundary checks

### 5.1 Reader changes without adaptive downstream blocks

Pythia-160M's final attention and MLP branches are parallel. Changing a final head cannot adapt that MLP, and no later block exists. On 512 reserved WikiText sequences, releasing the normalization factor yields mean reader recovery of -0.00060 nat, descriptive 95\% interval [-0.00159, 0.00014]. Positive average reader recovery is not established.

Centred-logit self-repair matches the reader contribution within the fixed tolerance, and full-forward checks leave the MLP unchanged. This is a structural attribution check, not a strong positive repair result or a reproduction of prior Pile effect percentages. It separates reader changes from block adaptation. Appendix C retains methods and per-head results.

### 5.2 More probe conditions did not establish better predictions

InterpBench tests whether two probes predict invariance under a third donor, after sealing predictions. On four reserved tasks and 5,632 queries, the strict five-condition rule has 50.52\% equal-task-average coverage and 2.35\% neural contradiction risk. Without the mismatched condition, these are 50.90\% and 2.34\%. The extra condition removes 23 assertions without removing a neural contradiction.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/04_interpbench_risk.pdf}
\caption{Finite risk-coverage operating points for the internal probe rules. Lines connect declared thresholds, not calibrated probabilities or a matched-coverage interpolation. Open circles mark threshold 1.00, including coincident adapted-only points. The right panel expands the low-risk region. Adding the mismatched condition establishes no incremental neural prediction advantage. Queries are correlated and the four tasks weighted equally; no population significance or matched-coverage ranking is inferred.}
\label{fig:interpbench}
\end{figure}

Case 18 also disagrees substantially with its programme: 119/256 native sources are exact, or 122/256 with canonical padding. Programme and compiled references agree on all 5,618 scorable future exchanges; fourteen masks are empty. Some programme contradictions therefore reflect resource fidelity and cannot all be assigned to the predictor. This limits a proposed predictive use of matching without negating the corroborated IOI recovery. Appendix D retains the domain, task counts and negative results.

## 6. Relation to prior work

[Rushing and Nanda](https://proceedings.mlr.press/v235/rushing24a.html) study frozen downstream contributions and the final normalization factor. Their centred-logit self-repair subtracts the direct-effect change using the original factor. These are direct antecedents. [IOI backup heads](https://arxiv.org/html/2211.00593v1), the [Hydra effect](https://arxiv.org/abs/2307.15771) and [copy suppression](https://aclanthology.org/2024.blackboxnlp-1.22/) likewise show why ablation outcomes can hide contributions. We claim no priority for these controls or IOI compensation.

[MIB](https://proceedings.mlr.press/v267/mueller25a.html) evaluates circuit and variable localization; we use public IOI prompts without an official method submission. [InterpBench](https://arxiv.org/abs/2407.14494v3) supplies programmes, alignments and networks whose native fidelity and exchange domain still require checks. Ablations of our own rule are not independently developed competing methods.

As emphasized by [Zhang and Nanda](https://openreview.net/pdf?id=Hf17y6u9BC), activation-patching results depend on metrics and intervention design. Direct effects, NLL recovery, donor matching and component blocking remain different estimands. A disagreement does not itself invalidate any measure. [Causal abstractions](https://papers.nips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html) also require specified alignments; the present finite tests are not internal-algorithm ground truth.

The earlier [NCM](https://arxiv.org/html/2610.14162v1) and synthetic studies remain in Appendix B. The parallel model is a one-shot proxy, not an NCM or VQ-VAE implementation. Appendix E's common-information argument measures no Wyner rate in these deterministic evaluations.

## 7. Limitations, provenance and accountability

### 7.1 Scope of the evidence

The central evidence uses one GPT-2 checkpoint, one task, 512 families, two fixed donors and three heads. Shared weights and prompts make the secondary tests intervention replications, not independent investigations. The margins lack prospective power or utility calibration; shared donors and finite templates limit inference. Recovery and group blocking do not identify every compensating component.

InterpBench lacks independent predictors at matched access and budget, excludes some auxiliaries, and retains case 18's fidelity failure. Pythia establishes no positive average reader repair. Heterogeneous appendix studies cannot be pooled into mechanism-classification accuracy.

### 7.2 Substantial AI assistance

OpenAI Codex research agents contributed substantially to planning, experimental code and execution, analyses, audits, literature synthesis and manuscript drafting and revision. This went beyond language editing. Other agents reviewed theory and implementation; their reviews and reruns are internal checks, not independent human replication.

The human project lead set goals and requested revisions. A completed independent human implementation of a principal contrast or comprehensive human source-to-claim verification is not documented and is not claimed. Retained code, protocols, outcomes, numerical incidents and drafts support inspection; automated checks do not certify qualitative interpretations.

AI tools are not authors. Human authors retain responsibility for claims, citations and text; an agent or audit count cannot discharge it. This draft does not assert completed human sign-off or submission readiness. Internal agent reviews are not recast as independent validation.

### 7.3 Reproducibility and status

Freezes are local, without an external immutable registry. Thresholds and model outputs are unchanged; margin sensitivity is post-hoc. Raw records and audits remain in the [IOI](../01_ioi/RAPPORT.md), [secondary](../01_ioi/REPLICATIONS_SECONDAIRES.md), [Pythia](../03_pythia/RAPPORT.md) and [InterpBench](../02_interpbench/RAPPORT_C2.md) reports. Appendix F records numerical corrections. Public release, official integration and submission are not claimed.

## 8. Conclusion

In this IOI case study, adaptation recovers substantial damage from replacing a Name Mover output, and separate replay interventions support a contribution from published backups. Donor matching adds a smaller positive effect. The late head's interval spans the conventional margin; it should not be described as lacking specificity.

Recovery, donor matching and component involvement require different comparisons. Continuous effects and access contracts carry the evidence; thresholds do not define mechanisms. InterpBench supplies no incremental predictive advantage; Pythia supplies a structural attribution check. Independent human replication and matched-access comparisons remain unperformed.


## References

- Rushing, C., and Nanda, N. (2024). [Explorations of Self-Repair in Language Models](https://proceedings.mlr.press/v235/rushing24a.html). ICML, PMLR 235, 42836–42855. [Technical definitions, arXiv v2](https://arxiv.org/html/2402.15390v2).
- Wang, K., Variengien, A., Conmy, A., Shlegeris, B., and Steinhardt, J. (2023). [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/html/2211.00593v1). ICLR.
- McGrath, T., Rahtz, M., Kramar, J., Mikulik, V., and Legg, S. (2023). [The Hydra Effect: Emergent Self-repair in Language Model Computations](https://arxiv.org/abs/2307.15771). arXiv:2307.15771.
- McDougall, C. S., Conmy, A., Rushing, C., McGrath, T., and Nanda, N. (2024). [Copy Suppression: Comprehensively Understanding a Motif in Language Model Attention Heads](https://aclanthology.org/2024.blackboxnlp-1.22/). BlackboxNLP, 337–363.
- Mueller, A., et al. (2025). [MIB: A Mechanistic Interpretability Benchmark](https://proceedings.mlr.press/v267/mueller25a.html). ICML, PMLR 267, 45069–45108.
- Tyen, G., Mansoor, H., Carbune, V., Chen, P., and Mak, T. (2024). [LLMs cannot find reasoning errors, but can correct them given the error location](https://aclanthology.org/2024.findings-acl.826/). Findings of ACL, 13894–13908.
- Kumbong, H., et al. (2026). [Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation](https://arxiv.org/abs/2610.14162v1). arXiv:2610.14162v1.
- Wyner, A. D. (1975). [The Common Information of Two Dependent Random Variables](https://doi.org/10.1109/TIT.1975.1055346). IEEE Transactions on Information Theory 21(2), 163–179. Original publisher full text was not directly available during this audit; the definition was cross-checked against the primary extensions listed here.
- Liu, W., Xu, G., and Chen, B. (2010). [The Common Information of N Dependent Random Variables](https://arxiv.org/abs/1010.3613). Allerton Conference.
- Graczyk, R., Lapidoth, A., and Wigger, M. (2022). [Conditional and Relevant Common Information](https://www.isiweb.ee.ethz.ch/papers/docu/rgra-alap-mwig-2022-2.pdf). Information and Inference. DOI: 10.1093/imaiai/iaab021.
- Cuff, P. (2013). [Distributed Channel Synthesis](https://arxiv.org/abs/1208.4415). IEEE Transactions on Information Theory 59(11), 7071–7096.
- Kumar, G. R., Li, C. T., and El Gamal, A. (2014). [Exact Common Information](https://arxiv.org/abs/1402.0062). IEEE ISIT, 161–165.

- GPT-2 checkpoint and model card. [openai-community/gpt2](https://huggingface.co/openai-community/gpt2), fixed revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.

- Geiger, A., Lu, H., Icard, T., and Potts, C. (2021). [Causal Abstractions of Neural Networks](https://papers.nips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html). NeurIPS 34.
- Zhang, F., and Nanda, N. (2024). [Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://openreview.net/pdf?id=Hf17y6u9BC). ICLR.

- Gupta, R., Arcuschin, I., Kwa, T., and Garriga-Alonso, A. (2024). [InterpBench: Semi-Synthetic Transformers for Evaluating Mechanistic Interpretability Techniques](https://arxiv.org/abs/2407.14494v3). NeurIPS Datasets and Benchmarks; arXiv version 3, 2025.
- Biderman, S., et al. (2023). [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373). [Pythia-160M checkpoint](https://huggingface.co/EleutherAI/pythia-160m), fixed revision recorded in Appendix C.
- Merity, S., Xiong, C., Bradbury, J., and Socher, R. (2016). [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). WikiText corpus; [retained dataset resource](https://huggingface.co/datasets/Salesforce/wikitext), `wikitext-2-raw-v1`.
- MIB–IOI dataset. [Public data and counterfactual variants](https://huggingface.co/datasets/mib-bench/ioi); frozen revision in the IOI provenance record.
- InterpBench implementation and models. [Code](https://github.com/FlyingPumba/InterpBench), [checkpoints](https://huggingface.co/cybershiptrooper/InterpBench), and [case-13 source correction](https://github.com/FlyingPumba/InterpBench/commit/bceb41a5c272fb1b1a91611396c89bbf81c87ee0).


## Appendix A. Formal contrasts and complete IOI records

The main text uses descriptive names. This appendix retains the original notation and additional outcomes. These are different contrasts on the same sources, not additional independent samples.

### Equations and the specificity counterexample

With ordered residual addition suppressed in the notation, define

\[
\begin{aligned}
Y_{00}&=D(z+T_0), &Y_{10}&=D(z+\delta+T_0),\\
Y_{11}&=D(z+\delta+T_\delta), &Y_{01}&=D(z+T_\delta).
\end{aligned}
\]

Add \(Y_{1w}=D(z+\delta+T_{\delta'})\), with a separately declared mismatched perturbation. The historical Gaussian experiments match norm and sign on another support; the native-activation experiments use another donor at the same site. Given reference loss L, measure

\[
\begin{aligned}
d&=L_{10}-L_{00}, &r&=L_{10}-L_{11},\\
i&=L_{10}+L_{01}-L_{11}-L_{00}, &s&=L_{1w}-L_{11}.
\end{aligned}
\]

Lower loss is better. Damage d measures the opportunity to repair; r measures recovery; i compares the continuation change across native and perturbed states; s compares matched and mismatched adaptations. These are contrasts, not mediation fractions or measurements of an intention to repair. The meaning of specificity depends on this control policy. Equal support does not imply equal norm, and matched norm does not imply matched difficulty or latent geometry. A positive s therefore measures an advantage over the declared mismatch, not a unique representation of error identity.

**A specificity trap.** A constructed program detects any nonzero perturbation and always applies the same correction -delta0. For the perturbation delta0, squared-loss values can be L00=0, L10=k, L11=0, L01=k, with k>0. Then r=k and i=2k are positive. Yet another perturbation elicits the same continuation, so s=0. Useful recovery for one intervention therefore does not establish a response specific to perturbation identity.

### Reciprocal input/state/continuation design

For inputs A and B with different known answers, save their native states z_A and z_B at the same cutoff. Execute all combinations of two recipient inputs, two donor states, and three continuation policies: frozen native A, frozen native B, and regeneration from the input/state combination. This gives 12 conditions. Transfer only the donor state; build the prefix cache from the recipient input. With state and tail fixed, changing recipient input cannot affect a decoder that does not directly read that input, providing an implementation check.

Use an additional answer-preserving pair as a control for input changes unrelated to changing the correct solution. A valid Sudoku output must be distinguished from a solution consistent with the recipient's clues. A donor solution can satisfy every Sudoku unit while failing the recipient puzzle.

### Input-dominance diagnostic

For regenerated continuations, let h_ij be the mean log-probability preference for solution B over A, with input i and donor state j. Define

\[
\begin{aligned}
u&=\tfrac12[(h_{BA}-h_{AA})+(h_{BB}-h_{AB})],\\
v&=\tfrac12[(h_{AB}-h_{AA})+(h_{BB}-h_{BA})].
\end{aligned}
\]
\[
w=h_{BB}-h_{BA}-h_{AB}+h_{AA},\qquad g=h_{BB}-h_{AA}.
\]

Effects u and v concern input and state, respectively; w is an interaction contrast. The identity u+v=g makes their normalized ratios complementary, not independent resources or causal percentages.

The prespecified per-family diagnostic requires both native grids exact, g>0.1 nat, |w/g|≤0.2 and u/g in [0,1]. Within this domain, u/g≥0.8 is input-dominant, u/g≤0.2 is state-dominant, and an intermediate value is two sensitivities/mixture-or-unmodeled. A failed gate or nonmonotone ratio does not get forced into a class. All families remain in aggregate effect estimates. These margins instantiate functional tests on real logits; they are not unmodified exact-equality classification rules from the analytical library.

### IOI contrasts and exactness

Before interventions, A/B/Aplus vocabulary exactness is 476/512, 449/512 and 446/512; two-candidate exactness is 512/512, 498/512 and 506/512. These native outputs are shared across targets. The C corruption experiment uses A anchors only; the E comparisons retain all variants and native errors.

**L9H9 primary.** Interaction I=1.27590 nat; direct-effect change=-3.38591 centred logit; self-repair=2.35486 centred logit. Total regenerated ablation damage is 0.84242 nat. B=0.21097 [0.18860, 0.23457]; blocking backups changes loss by 0.25106, controls by 0.04009 nat.

Vocabulary exactness in Y00/Y10/Y11/Y01/Y1wrong/backup-block/control-block order: 952, 59, 622, 1001, 581, 492, 599/1,024. Two-candidate exactness in the same order: 1024, 698, 1015, 1024, 975, 1007, 1016/1,024. The local D/R/S thresholds retain 192/512 families, compared with 507/512 when S is omitted. These are observed effect-threshold counts, not mechanism-classification accuracy.

**L9H6 secondary.** Interaction I=0.58050 nat; direct-effect change=-1.71435 centred logit; self-repair=1.53811 centred logit. Total regenerated ablation damage is 0.15555 nat. B=0.05446 [0.04154, 0.06732]; blocking backups changes loss by 0.11940, controls by 0.06494 nat.

Vocabulary exactness in Y00/Y10/Y11/Y01/Y1wrong/backup-block/control-block order: 952, 428, 904, 983, 883, 874, 895/1,024. Two-candidate exactness in the same order: 1024, 967, 1024, 1019, 1016, 1024, 1024/1,024. The local D/R/S thresholds retain 200/512 families, compared with 494/512 when S is omitted. These are observed effect-threshold counts, not mechanism-classification accuracy.

**L10H0 secondary.** Interaction I=0.17819 nat; direct-effect change=-1.10185 centred logit; self-repair=0.67796 centred logit. Total regenerated ablation damage is 0.28106 nat. B=0.12296 [0.10780, 0.13881]; blocking backups changes loss by 0.13329, controls by 0.01033 nat.

Vocabulary exactness in Y00/Y10/Y11/Y01/Y1wrong/backup-block/control-block order: 952, 654, 867, 982, 851, 800, 864/1,024. Two-candidate exactness in the same order: 1024, 1015, 1023, 1022, 1022, 1020, 1023/1,024. The local D/R/S thresholds retain 100/512 families, compared with 478/512 when S is omitted. These are observed effect-threshold counts, not mechanism-classification accuracy.

### Whole-state exchange is a different intervention

Natural state exchanges give a different result from resampling one head. In the A/B factorial at the layer-9 boundary, regenerated tails produce the recipient's vocabulary target in only 7/1,024 off-diagonal exchanges, compared with 1/1,024 for the recipient-native frozen tail and 4/1,024 for the donor-native tail. For answer-preserving A/Aplus, the respective counts are 430, 832 and 922. At the layer-10 boundary these counts are 11, 0 and 4 for A/B, and 877, 842 and 922 for A/Aplus. Layer-9 E outcomes are identical for the L9H9 and L9H6 runs because E exchanges the same complete boundary vector. Local head recovery therefore does not imply successful recipient recovery after a broader state swap, even with recipient prefix representations retained. All 12 conditions remain in the [summaries](../01_ioi/results/test_summary.json); no global architecture label is assigned.

### Reading precision and the margin display

The preceding historical estimates retain their original precision for traceability. The main text rounds to three or four decimals. A displayed last digit is not a claim of numerical or population precision. Figure 3 compares unchanged 98.75% interval lower endpoints with 0, 0.005, 0.01, 0.02, 0.03 and 0.05 nat. It changes no original margin and tests no new intervention. All three matching intervals exclude zero; only L9H9 and L9H6 exceed 0.01 nat at their lower endpoints.

## Appendix B. Earlier calibration and access-contract studies

These historical experiments motivated the controls but do not add independent evidence for the IOI circuit claim. The analytical library calibrates its own announced programmes; learned models have no mechanism-class labels supplied as ground truth.

The [calibration and NCM paper](../../article_benchmark_coordination/PAPER.md), [NCM instructions](../../benchmark_ncm_reel/README.md), [copying protocol](../../benchmark_multimodel_ar/PROTOCOLE.md), [matched-task protocol](../../benchmark_tache_commune/PROTOCOLE.md) and [access-contract note](../../benchmark_tache_commune/CONTRATS_ET_ESTIMANDS.md) retain full methods. Calibration has 344 constructed instances from 52 families. NCM and GPT-2 copying each use sixteen families. The matched task uses forty reserved rotation/reversal orbits and three paired training seeds.

NCM cuts after level 9, leaving levels 10-12. Its full reconstruction has 89 positions; level 9 has 37 code positions. The fixed local Gaussian dose is approximately 5.3% of global state norm. GPT-2 copying exchanges the final-position state after block six, retaining recipient context, at a dose of 0.1 times state norm. The inherited dominance diagnostic requires both native anchors correct, a native preference gap above 0.1 nat, interaction/gap at most 0.2 in absolute value, and input-effect/gap in [0,1]. Ratios at least 0.8 are input-dominant, at most 0.2 state-dominant; failed gates abstain. These are operational conventions, not architecture classes.

The matched models each have 77,576 parameters and four blocks of width 48. They train for 1,200 updates, batch 64, on identical partitioned rotation data, paired initial parameters and batches. AR uses teacher forcing; parallel queries use fixed placeholders. The parallel model has no NCM tokenizer, VQ-VAE or coarse-to-fine hierarchy. Different training objectives prevent a pure causal effect of generation paradigm.

The development dose grid was 0.1, 0.3, 1 and 3. None met the joint 10-40 percentage-point frozen-accuracy decrease and 0.05-nat loss-increase criterion for both systems. Dose 3 was a declared fallback, not a validated informative dose.

Retained-context regeneration combines seven recipient context positions with four donor queries. Whole-memory regeneration uses all eleven donor positions and no recipient tensor. Its zero recipient effect is a construction check. A frozen recipient-native tail carries information from a prior recipient computation. The boundaries below are interpretable only with these contracts fixed.

### Calibration checks the intended distinctions

The forced trajectory rule makes 52/344 correct singleton attributions, or 15.12%; its class-macro accuracy is 1/6. The cautious observation rule covers every true class but returns all six classes. The crossed diagnostic also covers every true class, with mean set size 2.395 and 52 correct singleton decisions. The complete diagnostic makes 344/344 correct singleton decisions, including 224/224 on reserved task families and mixtures.

The last number is expected by construction: program equations and diagnostics were developed together, and the library is public. It is a calibration check, not independent evidence of a generally accurate mechanism classifier. The 15.12% result is not a universal upper bound for trajectory-based methods. The exact matched observations establish non-identifiability for these constructed systems under the defined access regime.

### NCM native performance is at ceiling

All 48 A/B/Aplus inputs are solved exactly at level 12. No earlier complete 89-token readout is exact, including separators. This does not imply absence of useful information in earlier codes. Corrective and degrading transitions occur across the intermediate trajectories and are retained for all transitions without selecting a favorable peak.

### Local corruption creates no substantial repair opportunity

All five C conditions produce 128/128 exact final grids, preserving clues and separators. The primary mean frozen-damage contrast is d=1.74718×10^-6 nat, adjusted interval [-7.44527×10^-7, 6.31669×10^-6]. The corresponding results are r=-7.45874×10^-6 [-2.97070×10^-5, 1.23516×10^-7], i=9.47467×10^-7 [-2.83004×10^-7, 4.07239×10^-6], and s=9.67951×10^-7 [-2.92560×10^-7, 4.17891×10^-6].

The frozen-damage interval lies within the prespecified equivalence margin. This supports bounded mean robustness to the tested dose, not successful repair after material damage. Approximately 0.448% of suffix codes change, illustrating that code adaptation need not improve answer quality. No alternate dose or cutoff was selected after this result.

### Input-dominant preference does not establish necessary adaptation

Across all families, the input effect is u=29.4676 nat, descriptive 95% interval [28.1586, 30.7572]; the state effect is v=3.1798 [2.1810, 4.1940]; the interaction is w=-0.7311 [-1.4021, 0.0092]. The effect of state is therefore not erased by the final answer counts.

The fixed rule diagnoses input-dominant response preference in 15/16 families. The remaining family has u/g=1.0069949 and remains nonmonotone/non-identifiable without rounding it into the majority. No internal-mechanism ground truth is assigned to any family.

After reciprocal A↔B state swaps, regenerated continuations produce the recipient solution in 32/32 cases and never the donor solution. Crucially, frozen native recipient continuations also produce the recipient solution in 32/32 cases, whereas frozen native donor continuations produce the donor solution in 32/32. Thus regeneration has no exact-answer advantage over the recipient-native frozen control in these swaps. That control was produced by a prior full native recipient run; it is neither free answer access nor an established inference shortcut.

All outputs remain complete valid Sudoku grids, but donor solutions violate the discriminating recipient clue. In A↔Aplus swaps, all 32 off-diagonal outputs under each of recipient-native, donor-native and regenerated continuation policies match the common reference. Small probability changes persist without changing the solution.

The exactness/clue/format measures were planned, but their separately aligned descriptive exporter was added after primary results. It does not change frozen contrasts, thresholds or labels. This reporting distinction is recorded in the [study journal](../../benchmark_ncm_reel/JOURNAL.md).

### Auxiliary analysis separates distinguishability from dominance

We performed a locally sealed but explicitly post-hoc analysis of saved codes, with no new model calls. For each family F, define a balanced selector V between its two unique inputs. On this two-atom design, the exact finite-design quantity I(V;R_k|F=f) is one bit when the two complete code vectors differ, and zero otherwise. This is not an estimated population information measure or a test of independently decodable semantics.

Level 5 is constant at code 11 and distinguishes no family. Nevertheless, levels 8 and 9 each distinguish both inputs in 16/16 families. The concatenated levels 1–9 and 10–12 each distinguish the branch in all families, so their finite-design conditional mutual information given family is one bit. Conditioning instead on the complete input yields zero analytically under deterministic argmax; the conditioning variables define different questions.

Distinct code vectors can reflect the changed clue without providing a reusable semantic decoder. Code distinguishability alone also does not guarantee distinguishability of the accumulated reconstruction: the code tables and transformations need not be injective. This result rules out an account based simply on absence of all pair identity in the tested coarse codes, not all forms of information loss. It also shows why dominance of a retained input cannot by itself establish an informational deficit in the earlier state.

All planned post-hoc sensitivity cutoffs are retained: 15/16 families meet input ratio 0.70 or 0.80; 8/16 meet 0.90 and 4/16 meet 0.95, with the same eligibility gates. The historical threshold remains 0.80. These counts expose the descriptive threshold dependence rather than redefine the principal result.

### GPT-2: context dominance is compatible with autoregression

Official/native reconstruction and sham continuation agree exactly for all 48 prompts. Native accuracy is 12/16 on A, 8/16 on B and 13/16 on Aplus: 33/48 overall. Both A/B anchors are correct in eight families. Intermediate readings contain corrections and degradations: B is correct in twelve sources after block ten and eight at the final block. These are internal readouts, not revisions of emitted text.

For C, D=0.00181115 nat/token, adjusted interval [0.00141522, 0.00213673]; R=0.00781555 [0.00088758, 0.01519674]; I=0.00181959 [0.00113743, 0.00242062]; S=0.00168670 [0.00070116, 0.00268459]. Mean frozen damage falls within the fixed equivalence margin and the joint useful-and-specific recovery criterion is not met. Y00 and Y10 are correct in 80/128 trials; Y11, Y01 and Y1wrong in 78/128. A disclosed post-hoc paired exporter finds no argmax change under frozen corruption, but two correctness losses and no gains in each adapted-tail condition. Thus improved mean NLL under regeneration coexists with reduced exactness. The exporter uses saved outputs and changes no frozen diagnostic.

Mean E context effect u is 3.76857 nat, descriptive 95% interval [1.23271, 6.39620]; state effect v is 0.30567 [0.19423, 0.42402]; interaction w is 0.01608 [-0.21121, 0.22946]. The unchanged rule diagnoses input-dominant preference in **8/16 families**, while **8/16 remain non-identifiable due to native-anchor failure**. Every family passing the native gate meets the other input-dominance conditions; reporting only this eligible subset would hide half of the corpus. All sixteen effects are retained.

Across 32 off-diagonal A/B swaps, regenerated tails give 19 recipient-correct outputs, 10 donor-correct and three others. Recipient-native frozen tails give 18, 11 and three; donor-native frozen tails give nine, 20 and three. A paired descriptive export finds one gain and no loss for regeneration against the recipient-native control. This local difference does not establish a universal repair mechanism. In A/Aplus swaps, regenerated, recipient-native and donor-native tails give 25/32, 26/32 and 25/32 correct outputs. Identical references are not normalized into a response-preference ratio.

The AR [audit](../../benchmark_multimodel_ar/results/verification.json) recomposes all 1,024 C/E decoder outcomes, recalculates 640 scalar NLLs and reproduces scores, grouped C intervals, labels, norms, diagonals and invariances. Explicit sublayer calls replay 16 C and 16 E continuations on the first two families, plus six official native forwards. States and logits match exactly. Historical manifests and frozen inputs remain unchanged. This is an internal software audit, not an independent empirical replication.

### Matched-task and access-control results

#### Native success does not ensure an informative corruption

Every model solves all 40 A, 40 B and 40 Aplus examples at the first symbol and full freely generated sequence: 120/120 per training. Thus native-anchor failure no longer explains abstention in this experiment. The task remains easy and synthetic.

No candidate dose passes the common-damage criterion on development. At dose 3, AR has mean NLL damage 0.00164032 nat and zero accuracy loss; parallel has 0.05683645 nat and a 0.278-percentage-point loss, far below the required ten points. We retain the fixed fallback without widening the grid. Consequently this campaign does **not** establish a successfully calibrated common informative corruption.

On evaluation at the primary cutoff, frozen/adapted exactness over 320 trials is 320/237, 320/267 and 320/210 for the three AR seeds; 316/237, 316/244 and 317/255 for the parallel seeds. Mean recovery contrasts R are respectively −0.801597, −0.556550, −1.035251 and −0.670127, −0.596030, −0.532856 nat. Each model's adjusted R interval is wholly negative. I and S means are also negative at all six primary cells. The three AR frozen-damage means are equivalent within the 0.01 margin; parallel damage means range from 0.058768 to 0.072958 nat. No model meets the useful-and-specific recovery criterion. Regeneration can amplify perturbations better tolerated by a previously computed frozen tail, conditional on this large dose and its unsuccessful calibration.

#### Primary exchanged states follow the donor

After block two, retained-context regeneration produces the donor's first symbol in all 80 off-diagonal A/B swaps for every model, with zero recipient-correct outputs. Donor-native frozen tails also give 80/80 donor outputs. Recipient-native frozen tails give 80/80 recipient outputs for every AR seed, and 56/80,44/80,36/80 for the parallel seeds. Regeneration therefore provides no recipient-exactness gain against these controls. Frozen tails required a prior full native computation; no efficiency claim follows.

The strict retained-context diagnostic labels 20,25,27 of 40 families state-dominant for AR seeds 17,29,43 and abstains as nonmonotone in the remaining 20,15,13. Parallel counts are 18,17,28 state-dominant and 22,23,12 nonmonotone. Native anchors are all exact. Negative input ratios are preserved even when close to zero; continuous effects and uniformly donor-following argmax outcomes remain visible alongside the strict labels. Abstention here is neither evidence of task incompetence nor a complexity theorem.

#### Earlier cutoffs and access contracts change the result

At the declared secondary cutoff after one block, retained-context regeneration gives 0/80,80/80,0/80 recipient outputs across AR seeds and 0/80,40/80,80/80 across parallel seeds. These seed effects are reported in full. AR 29 has 20 input-dominant and 20 nonmonotone families; parallel 43 has 37 input-dominant and three nonmonotone families. Parallel 29 fails the interaction gate in all 40 families; its intermediate output counts are not classified as a known mixture mechanism. The remaining models have mostly state-dominant or abstaining labels, detailed in the full report.

Recipient-native frozen tails already give 80/80 recipient answers for all six models at this earlier cutoff. Early regenerated success does not establish its necessity for exactness. In AR 29 and parallel 43, identical weights switch from recipient-following early regeneration to donor-following regeneration after block two. This is a boundary-dependent functional result, not a universal architecture signature.

Under whole-memory regeneration, every model/cut exactly reproduces donor-native logits for every recipient/donor combination. The receiver effect and interaction are exactly zero; each cell gives 40/40 state-dominant labels under the correct native anchors, as expected by construction. Early AR 29 and parallel 43 therefore also change from recipient-following to donor-following behavior when the access contract changes on the same weights. Other seeds were already donor-following, so that access effect is not asserted to be universal in the retained track.

## Appendix C. Reader-only Pythia control

### Pythia: changing the reader without an adaptive block continuation

We use the frozen [Pythia-160M checkpoint](https://huggingface.co/EleutherAI/pythia-160m), revision `50f5173d932e8e61f858120bcb800b97af589f46`, with 162,322,944 parameters. The final attention and MLP branches are parallel. Replacing a final attention-head output therefore cannot change the same layer's MLP, and no later attention/MLP block exists. This gives a structural control for attributing reader effects to downstream adaptation.

The fixed targets are L11H2 and L11H10. We sample 128 development rows and 512 test rows from the official validation and test partitions of [WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext), after input-only eligibility and duplicate checks. The first 32 tokens provide context and token 33 is the reference. Two other sequences, at fixed offsets 1 and 17, supply native activations at the same final-position head. There is no high-direct-effect position selection or ratio clipping. Multiple rows may come from one article; shared donors and document dependence limit the descriptive bootstrap.

The five conditions cross native/corrupted residuals with native, corruption-matched and other-donor final-normalization denominators. Residual centering is recomputed; only the denominator is fixed or exchanged. Here R, I and S are explicitly reader contrasts. The corresponding adaptive block-continuation effect is absent by the computational graph. We report NLL, vocabulary exactness, centred logits, raw self-repair and scale ratios, using 10,000 bootstrap resamples of whole sequences with both heads and donors grouped. The development gate requires technical parity, not a favorable recovery effect. WikiText and unselected positions differ from the prior Pile experiment, so this is a targeted structural test, not a reproduction of its reported effect percentages. [Protocol](../03_pythia/protocol.json).

### Full reserved outcome


On 512 reserved sequences, the native model predicts 151 next tokens exactly. Four interventions per sequence—two heads and two donors—give 2,048 correlated trials. Frozen-factor damage is 0.002391 nat, descriptive 95% interval [−0.001605, 0.006869]. Releasing the final-normalization factor yields R=−0.000601 nat [−0.001589, 0.000141]. No positive average reader recovery is established. Exactness is 604/2,048 in the repeated native condition, 598/2,048 under both frozen and adapted factors, 604/2,048 for the native residual with adapted factor, and 597/2,048 with the other donor's factor. Frozen and adapted exactness has no paired gains or losses.

Mean centred-logit self-repair is 0.000824 [−0.005052, 0.006706]; the measured reader contribution is 0.000824 to the same displayed precision. The two heads differ in sign: H2 has mean self-repair 0.003459 and H10 −0.001812. Neither result was used to replace a target or select high-effect positions. The maximum per-trial discrepancy between self-repair and the reader identity is 0.000171, below the fixed 0.0002 numerical tolerance. The final MLP contribution is unchanged in the full-forward checks. Thus the experiment separates an observable reader effect from a nonexistent adaptive downstream block continuation; it does not recover a large mean repair effect on this unselected WikiText sample. Remaining reader contrasts and intervals are given below and the [retained summary](../03_pythia/results/test_summary.json).

### Pythia reader contrasts

Across both heads, I=0.000007095 nat, descriptive 95% interval [−0.000230012, 0.000147205], and S=−0.000109216 [−0.000355497, 0.000019042]. The mean native/corrupted scale ratio is 1.00013290 [0.99988247, 1.00037891]; 1,035/2,048 ratios exceed one. Positive individual reader recovery occurs in 1,016/2,048 trials and positive raw self-repair in 1,053/2,048; their signs do not imply positive mean recovery.

H2 has D=0.00555188, R=−0.00098630 and S=−0.00019393 nat. H10 has D=−0.00077030, R=−0.00021589 and S=−0.00002450 nat. The corresponding direct-effect changes are −0.02926254 and −0.00723953 centred logit. The [summary](../03_pythia/results/test_summary.json) retains all per-head intervals. No favorable head or high-effect position subset replaces these results.

## Appendix D. InterpBench prediction test and oracle domain

### InterpBench: predicting a future exchange rather than assigning a mechanism class

The external-reference evaluation uses published InterpBench checkpoints: cases 8 and 33 for development, then 256 distinct source sequences each for reserved cases 13, 18, 19 and 21. Programmes and checkpoint files are pinned. Programmes are executed through the published RASP functions and the official Tracr-to-Craft compiler. The learned networks run in the official TransformerLens backend. Neither the oracle nor the chosen programme is inferred from diagnostic predictions.

Provenance is important. The public code corrected case 13 on 9 September 2024, after the retained checkpoint and circuit files were already published. Their file identities are unchanged across the compared Hugging Face snapshots. We therefore use the pre-correction programme, recovered from the parent of that source change before reserved inference. Its operational labels retain the historical function, including its defects; they are not silently relabeled to match a corrected intuitive definition of numerical trend. For cases 18 and 21, some compiler auxiliaries lack directly exchangeable RASP nodes. They remain in the reference computation but are excluded from positive interventions whose semantic masks cannot be verified. Negative controls are absent from the complete Craft model, not merely absent from an exported edge list. The selected units and exclusions are [sealed in the C2 plan](../02_interpbench/data/c2_plan.json).

For each source and eligible head/MLP, two same-length donors provide test-time probes and a third donor supplies the future intervention. These are distinct donors within a query; the bank is reused across queries. Probes are available when making a prediction on that source. This is within-family extrapolation to a new donor, not prediction on an entirely unobserved family. The C2 inputs contain BOS and useful tokens without PAD. All categorical comparisons exclude BOS and padding; oracle comparisons also exclude positions whose RASP result is None, using the mask of the actual intervention.

The prediction is deliberately narrow: retain the native categorical output as a forecast of invariance, or abstain. Cautious O always abstains; forced O always forecasts invariance. The Y11-only rule uses the fraction of its two adapted probe outcomes that match the native argmax sequence. A second rule averages agreement across Y00/Y10/Y11/Y01; the five-condition rule also includes Y1wrong. Agreement uses all useful positions, including those without a defined RASP reference. These scores are agreement fractions, not calibrated probabilities. Thresholds 0.50, 0.75 and 1.00 are fixed in advance, with 1.00 primary. At the strict point, every included probe outcome must agree. All predictions are sealed before programme, Craft or learned-network responses to the third donor are computed. None of these rules is a six-way mechanism classifier, and the ablated rules are internal controls rather than independently developed literature methods.

We report coverage and contradictions against the neural future output, and separately against the programme on its defined mask. Programme risk can include native network–programme disagreement as well as a failed future forecast. Interchange accuracy for aligned variables and strict interchange accuracy for unaligned variables quantify network–programme conformity; they are not competing predictors. Results are reported per task and averaged equally across four tasks, without population p-values over correlated queries. E is a separate whole-memory 2-input × 2-state × 3-tail check and is never used secretly by the C predictor.

After the C2 results revealed weak native conformity for case 18, C3 compared the same 1,024 sources under the official BOS/PAD input convention. The programme reference still uses the useful tokens, as in the published data generator. This control was sealed before its own calculation but is explicitly post-hoc to C2; it changes no C2 prediction, threshold or result. [C2 protocol](../02_interpbench/PROTOCOLE_C2.md), [C3 protocol](../02_interpbench/PROTOCOLE_C3_PADDING.md).

### InterpBench: a sealed prediction test with no demonstrated incremental advantage

The development oracle checks cover 256 sources and 1,536 exchanges, including 499 changed programme answers, with agreement among programme, Craft and learned-network outputs on all exchanges. The official neural backend passes its native replay and sham checks. On the reserved tasks, all 5,618 queries with a nonempty semantic mask agree between RASP and Craft. Fourteen further queries have no defined categorical target; they are retained as unscorable, not credited as successes.

Network–programme conformity varies by task. Historical case 13 solves 256/256 native sources and matches the programme on 1,024/1,024 aligned and 512/512 unaligned future exchanges. Case 19 solves all 253 native sources with defined outputs; three have no defined native target. Its evaluated exchanges match in 507/507 aligned and 759/759 unaligned cases. Case 21 solves 256/256 native sources and matches in 768/768 aligned and 1,277/1,280 unaligned exchanges. **Case 18 fails the fidelity check:** only 119/256 native sequences are exact, with 116/256 aligned and 232/512 unaligned exchanges matching the programme. We retain the case in the aggregate instead of removing an inconvenient task.

At the strict agreement threshold, cautious O emits no claims and has undefined risk. Forced O covers every query and has task-averaged neural contradiction risk 41.04%. The Y11-only rule covers 55.74% with 2.39% risk; omitting only the mismatched condition gives 50.90% coverage and 2.34% risk. The five-condition rule covers 50.52% with 2.35% risk. These are finite task averages, not a statistical ranking at matched coverage. The additional mismatch check removes 23 assertions without removing a neural contradiction. Two programme contradictions are removed, both in the already nonconforming case 18. **An incremental predictive advantage from all five conditions is not established.** The lower-threshold results are also retained rather than selecting the most favorable point.

For the five-condition rule, programme contradictions are 14/578 in case 13, 217/439 in case 18, 1/545 in case 19 and 3/1,316 in case 21. Their equal-task mean risk is 13.07%. This is not directly interchangeable with neural risk: programme scoring uses its defined-position mask and can reflect native model–programme mismatch. A separately labeled post-hoc audit puts neural scoring on the same masks; it does not change any prediction. The high risk in case 18 cannot be assigned entirely to a failed attribution rule.

The padding control does not resolve this discrepancy. With the official padded input convention, case 18 is exact in 122/256 native sources, while the padded Craft reference matches the useful-token programme in 256/256. Cases 13, 19 and 21 remain exact on their defined native references with and without padding. C3 therefore rules out padding alone as an explanation in this comparison; the cause of case 18's mismatch remains unestablished. Its post-hoc results do not replace C2.

The separate E check contains 55,296 conditions. All 18,432 regenerated whole-memory conditions reproduce donor-native logits exactly, as required by the blocked recipient path. This validates the adapter's stated access contract, not a discovered internal algorithm. Full task counts, exclusions, risk–coverage points and [verification records](../02_interpbench/results/c2_verification.json) are retained with the [C2/C3 report](../02_interpbench/RAPPORT_C2.md).

### InterpBench agreement-threshold grid

Coverage, neural contradiction risk and programme contradiction risk below are equal-task means in percent. Neural risk uses all useful positions; programme risk uses the intervention's defined mask. Scores are agreement fractions with different included conditions, so equal score thresholds do not mean equal coverage or calibrated confidence.

- **Y11 only**, thresholds 0.50/0.75/1.00: 0.5: 62.53, 9.58, 17.70; 0.75: 55.74, 2.39, 13.19; 1.0: 55.74, 2.39, 13.19.
- **Without Y1wrong**, thresholds 0.50/0.75/1.00: 0.5: 93.07, 36.01, 36.51; 0.75: 62.13, 9.31, 17.43; 1.0: 50.90, 2.34, 13.12.
- **Five conditions**, thresholds 0.50/0.75/1.00: 0.5: 62.84, 9.88, 17.91; 0.75: 53.77, 2.52, 13.22; 1.0: 50.52, 2.35, 13.07.

Forced O has 100% coverage, 41.04% neural risk and 41.88% programme risk at every threshold. Cautious O has zero coverage and undefined risks. The [post-hoc mask audit](../02_interpbench/results/c2_mask_and_c3_audit.json) preserves neural scoring on the programme-defined positions as well; its values are not substituted for the original grid. Finite source/task counts, excluded auxiliaries and empty masks remain visible in the [full summary](../02_interpbench/results/c2_summary.json).

## Appendix E. A separate conditional common-information argument

This note is not needed for the IOI result and does not explain its recovery. It states assumptions for a future stochastic coordination experiment. No present measurement estimates a Wyner common-information rate.

For discrete output variables Y_1,...,Y_m and shared side information P, define the conditional multivariate extension

\[
C_W(Y_1;\ldots;Y_m\mid P)
=\inf_{U:\,P(Y\mid U,P)=\prod_iP(Y_i\mid U,P)} I(Y;U\mid P).
\]

This combines established multivariate and conditional definitions; it is not a new quantity. For local producers with mutually independent private randomness conditioned on (U,P), their U is an admissible candidate, hence C_W≤I(Y;U|P)≤H(U|P). This bounds a discrete shared-information budget only when all shared resources and the local-readout restriction are accounted for. Exact finite synthesis and asymptotic approximate rate interpretations must also be distinguished; see [Exact Common Information](https://arxiv.org/abs/1402.0062). Input-dependent side information is not automatically the independent common randomness considered in [distributed channel synthesis](https://arxiv.org/abs/1208.4415).

Two elementary targets clarify the issue. The uniform even-parity distribution on n bits has C_W=n−1 bits under mutually conditionally independent production, although total correlation is one bit and pairwise information vanishes for n≥3. A uniform mixture of two solutions of one ambiguous Sudoku base has C_W=1 bit when at least two cells differ. A complete distinguishing clue reveals that branch; conditional on the resulting unique puzzle, C_W is zero. Proofs and finite enumeration checks are provided in the [theory note](../../article_benchmark_coordination/THEORIE_WYNER.md).

In the executed deterministic NCM, both coarse and fine generated codes are functions of the full input, so I(coarse;fine|input)=0. In a future stochastic campaign, if per-position sampling is audited to use independent private draws, codes would factorize conditioned on input and previous state. That property would concern generated codes, not arbitrary codes produced by the frozen solution encoder, and not necessarily the final digits. A global decoder can create dependent output coordinates from independent code sources: a parity-computing decoder gives an immediate counterexample to applying the local-producer bound directly to NCM outputs.

The 15/16 functional diagnoses therefore measure neither C_W nor a common-information rate. They are consistent with the input acting as a shared conditioning signal, but do not establish minimality, coordination cost or compensation of pooling loss. The historical pooling result concerns an exact raw averaging symmetry and a specific empirical constant level; the 3a experiment did not manipulate pooling. The raw length-nine invariant concerns level 5, whereas the present level-9 cutoff has 37 positions. The solution encoder is not called by native generation. No causal chain from pooling to a need for shared information has been identified. The finite paired-code analysis shows that a constant level can coexist with branch-distinguishing coarse codes.

A nontrivial future test must specify a stochastic target distribution at fixed ambiguous input, audit the sampler, preserve output diversity, and separate local from global readout. Pooling would require its own matched intervention. These requirements convert the common-information connection into a testable program rather than an explanation asserted after observing input sensitivity.

## Appendix F. Numerical incidents, inventory and evidence

### Numerical incidents and internal checks

Numerical failures are part of the retained record. In IOI, a nonfinite low-dimensional FP32 vector–matrix measurement was localized after development, without opening test outputs. The direct-effect measurement was moved to FP64 and checked against NumPy and scalar arithmetic; the neural model and interventions remained FP32. The behavior was not reproduced in every memory arrangement, so its low-level cause is not claimed. In Pythia, the initial manual reader missed the fixed parity tolerance. A native-kernel implementation of the same reader and a separately logged FP64 direct-effect measurement passed without relaxing thresholds. InterpBench's first FP32–FP64 comparison also failed its tolerance; equal-precision comparison with the official backend then separated numerical differences from equation mismatch. None of these development corrections silently changes the original failure or test criterion.

For each IOI head, a supplementary audit compares 40 full official forwards on four fixed sources, including blocking and E, with the stored results; logits agree bitwise. The reader/direct-effect identities are checked separately. These additional audits were added after experimental freezing as software checks, not new hypothesis tests. Pythia's native reader is exact and its corrupted full-forward replay stays within the fixed 0.0002-logit tolerance. The independent-formula test audit uses the same first 32 records and tolerance through a memory-mapped implementation added after results. InterpBench's audit reconstructs predictions and contradiction counts from the saved logits and verifies seals; the mask clarification and padding control are explicitly post-hoc.

### Study inventory and retained evidence

The broader record contains fifteen learned checkpoints: one NCM, one GPT-2 reused for copying and IOI, six small matched-task Transformers, one Pythia and six InterpBench networks including development. These heterogeneous objects are not fifteen independent replications of the central claim. Main-text IOI evidence uses one checkpoint.

The [integration ledger](PREUVES_INTEGRATION.md) maps historical numerical statements to retained summaries. The [previous manuscript](archives/avant_revision_critiques_20261025/PAPER_REVISE.md) and typesetting sources are archived. The [revision response](REPONSE_CRITIQUES.md) describes the scope, thresholds, figures and provenance changes; the [margin display data](SENSIBILITE_MARGES.json) records the post-hoc comparisons. Raw observations, protocols and thresholds remain unchanged.

Measured IOI runner times are approximately 432, 422 and 436 seconds for primary and secondary targets. Pythia's reserved runner takes 63.5 seconds; InterpBench's probe/prediction and falsification phases take 83.8 and 103.8 seconds. Loading and phase-specific work are included, audits and installation separate, and other work sometimes shared the machine. These are session costs, not isolated latency benchmarks.

Remaining work includes independent implementation on new families, donor-policy sensitivity using fresh interventions, independent-method comparisons at matched access and coverage, semantic oracle coverage for excluded auxiliaries, and investigation of case 18 in a separately specified campaign. None is represented as completed.

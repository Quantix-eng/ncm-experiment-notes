# Corrections du manuscrit — texte prêt pour une nouvelle version

25 octobre 2026. Rédaction anglaise destinée à une nouvelle version éditoriale. Les paragraphes décrivent l'état antérieur et le programme de validation ; ils ne doivent pas présenter une campagne nouvelle comme exécutée avant son rapport. Le PDF V3 reste conservé.

## 1. Antécédents directs — insertion dans Related Work

**Self-repair and direct-effect controls.** Rushing and Nanda study changes following attention-head resampling, including frozen downstream contributions and the final normalization factor. Their self-repair measure is the change in the centred correct-token logit minus the change in the ablated head's direct effect, evaluated using the original normalization factor. These controls are direct antecedents of our continuation and reader controls; freezing a continuation alone is not our novelty claim. IOI backup heads, the Hydra effect and copy suppression provide earlier evidence that ablation outcomes can conceal contributions or mix several sources of compensation. Our open question is whether jointly evaluating damage, recovery, specificity and accessible information changes conclusions that are corroborated by reserved interventions. [Rushing and Nanda, ICML 2024](https://proceedings.mlr.press/v235/rushing24a.html), [Wang et al., IOI](https://arxiv.org/html/2211.00593v1), [McGrath et al., 2023](https://arxiv.org/abs/2307.15771), [McDougall et al., 2024](https://aclanthology.org/2024.blackboxnlp-1.22/)

## 2. Question principale — remplacement d'une revendication de supériorité

**Claim and validation target.** We evaluate a restricted scientific claim: do additional controls distinguish useful downstream adaptation from tolerance and reader effects, and do those distinctions predict independent intervention outcomes? The constructed systems calibrate our implementation within a finite announced library. Learned-model results are functional findings conditional on access contracts, without labels for the model's complete internal mechanism. Additional models do not, by themselves, establish independent diagnostic accuracy or benchmark utility.

## 3. Comparaison de méthodes — insertion avant les nouveaux résultats

**Different estimands are not competing truths.** Direct logit attribution, total ablation effects, published self-repair and NLL recovery measure different objects. We retain these quantities separately on the same cases. A numerical disagreement is not evidence that a prior measure is wrong. Added value requires a specific additional conclusion, stated before its corroboration data are examined, and supported by an intervention not used to generate it. Versions omitting our frozen-tail or mismatched-tail control are ablations of our protocol, not independently developed literature baselines.

## 4. Objectif AR/parallèle — clarification de portée

**Architectural scope.** Establishing a general autoregressive-versus-parallel effect is outside our immediate validation target. The earlier matched-task comparison documents boundary, access and training-realization dependence. Its parallel Transformer is a proxy for one-shot generation, not a test of the NCM tokenizer, VQ-VAE or concept hierarchy. Different training objectives remain a confound for attributing the observed differences to the generation paradigm itself. We retain that experiment as a scoped case study rather than a general architecture taxonomy.

## 5. Conclusion — remplacement de la priorité méthodologique

**Conclusion.** Frozen-contribution and reader-normalization controls have direct precedents. The contribution to be tested is the practical utility of combining them with material-damage gates, same-site donor controls, explicit information-access contracts and reserved corroborating interventions. Current calibration and neural case studies establish executable measurements and bounded functional observations. Independent method comparisons and external replication must establish whether these measurements improve a scientific conclusion. Failure to damage the system, absence of recovery, contradicted component attributions and lack of added value remain reportable outcomes.

## 6. Transparence de la validation

**Externality and independence.** Public datasets, published hypotheses and independently constructed reference programs provide external evaluation objects. They do not make an evaluation implemented by the diagnostic's authors an independent replication. A separate person must implement the main contrast from a specification and evaluate new families before we can claim that form of validation. Internal software audits and reviews by additional research agents remain explicitly internal.

## Références à ajouter

- Cody Rushing and Neel Nanda. 2024. *Explorations of Self-Repair in Language Models*. ICML, PMLR 235, 42836–42855. [Notice officielle](https://proceedings.mlr.press/v235/rushing24a.html).
- Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris and Jacob Steinhardt. 2023. *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*. ICLR. [Texte primaire arXiv v1](https://arxiv.org/html/2211.00593v1).
- Thomas McGrath, Matthew Rahtz, Janos Kramar, Vladimir Mikulik and Shane Legg. 2023. *The Hydra Effect: Emergent Self-repair in Language Model Computations*. arXiv:2307.15771. [Notice primaire](https://arxiv.org/abs/2307.15771).
- Callum Stuart McDougall, Arthur Conmy, Cody Rushing, Thomas McGrath and Neel Nanda. 2024. *Copy Suppression: Comprehensively Understanding a Motif in Language Model Attention Heads*. BlackboxNLP, 337–363. [Notice officielle](https://aclanthology.org/2024.blackboxnlp-1.22/).

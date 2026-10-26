"""Editorial integration from archived prose and retained experimental summaries.

No model import, new experiment, threshold change or scientific-artifact write.
"""
from pathlib import Path
import json,re,hashlib

D=Path(__file__).resolve().parent
V=D.parent
BASE=D/'archives/avant_integration_resultats_20261025/PAPER_REVISE.md'
claims=[]
def read(p):return json.loads((V/p).read_text())
def val(source,*keys,fmt='.5f'):
    value=read(source)
    for key in keys:value=value[key]
    rendered=format(value,fmt)
    claims.append(dict(source=source,keys=list(keys),value=value,format=fmt,rendered=rendered))
    return rendered
def ioi(suffix,k):return val('01_ioi/results/test'+suffix+'_summary.json','contrasts',k,'mean')
def ci(suffix,k):
    p='01_ioi/results/test'+suffix+'_summary.json'
    return '['+val(p,'contrasts',k,'ci9875',0)+', '+val(p,'contrasts',k,'ci9875',1)+']'
def count(suffix,condition):return val('01_ioi/results/test'+suffix+'_summary.json','accuracy',condition,'correct_vocab',fmt='d')

paper=BASE.read_text()
paper=paper.replace(paper.splitlines()[0],'# When Does a Continuation Repair a Perturbation? Crossed Controls and Tests on Learned Models',1)
old_note=paper.splitlines()[2]
paper=paper.replace(old_note,'Research manuscript updated 25 October 2026. This version integrates the completed IOI, Pythia and InterpBench evaluations with the earlier calibration and case studies. Protocols, observations and historical manuscript versions are retained separately. This update changes the presentation, not the experimental results.',1)

abstract=f'''A model can recover a correct answer after a perturbation without establishing which computation produced the recovery. We evaluate compensation using crossed native and adapted continuations, mismatched adaptations, and explicit contracts for retained information. Analytical controls calibrate the expected distinctions; earlier NCM and Transformer case studies show why successful outputs and weak corruptions are insufficient tests of repair. We then evaluate GPT-2 small on 512 reserved MIB–IOI families. Resampling L9H9 causes {ioi('','D')} nats of frozen-continuation damage, of which adaptation recovers {ioi('','R')}. Blocking published downstream backup heads reduces recovery more than a layer-matched control. Two declared secondary targets also recover, but L10H0 does not meet the fixed margin for matched-versus-mismatched adaptation. A Pythia-160M final-layer control separates reader normalization from downstream block adaptation and establishes no positive mean reader recovery on 512 WikiText sequences. Finally, an InterpBench extension tests 5,632 future interventions across four tasks after sealing predictions. At the strict threshold, the five-condition rule has 50.52% task-averaged coverage and 2.35% neural contradiction risk, compared with 50.90% and 2.34% when its mismatched condition is omitted. A checkpoint–program discrepancy persists in one task under canonical padding. The evidence supports bounded compensation claims on IOI and exposes limits of the proposed evaluation: the extra controls do not establish a general predictive advantage or a universal classifier of internal mechanisms.'''
abstract=abstract.replace('and establishes no positive mean reader recovery','and does not establish positive mean reader recovery')
paper=re.sub(r'(?s)(## Abstract\n\n).*?(?=\n## 1\.)',lambda m:m[1]+abstract+'\n',paper,count=1)

intro='''## 1. Introduction

An intermediate answer can be wrong, become correct, deteriorate and recover again. It is tempting to read this trajectory as a record of self-correction. Yet the same observations can arise when a reader progressively reveals an answer that was already present, when a perturbation is ignored, or when later computation reconstructs the answer from information that remained available. Final accuracy alone cannot separate these accounts.

The practical question is narrower: what evidence warrants attributing recovery to a changed continuation? A successful regenerated continuation is not enough if a previously computed native continuation works equally well. Nor can a nearly harmless intervention provide a strong test of repair. To study these cases, we cross native and perturbed states with native, adapted and mismatched continuations, then state exactly which input-derived representations each continuation can still access.

Our initial calibration and neural case studies established that these interventions could be executed and audited, but also exposed a central weakness: several corruptions caused little damage. The IOI experiments address that weakness using public prompts and components implicated in compensation by prior work. They produce substantial damage, measurable recovery and a separate intervention on candidate compensating heads. The Pythia experiment asks a complementary question: can an apparent self-repair effect arise entirely at the reader when no downstream block can adapt?

These are tests of attribution, not of forecasting performance. We therefore add a separate evaluation on InterpBench. A rule receives two intervention probes, predicts whether an exchange with a third donor will preserve the output, and is then tested against both the learned network and an external programme oracle. This evaluation does not establish an additional predictive benefit from our full set of conditions. It also reveals why checkpoint–programme agreement must be measured rather than inferred from a resource's ground-truth label.

The contribution is an auditable account of which compensation claims survive these controls and where they fail. Frozen-contribution and reader-normalization controls already have direct antecedents in self-repair research. We do not claim their invention. The earlier AR/parallel study remains a case study of access and boundary dependence, and the common-information discussion states conditions for a future coordination experiment rather than explaining the present results through a measured Wyner rate.
'''
paper=re.sub(r'(?s)## 1\. Introduction\n.*?(?=\n## 2\.)',lambda _:intro,paper,count=1)
paper=paper.replace('Our target is narrower: attribution of temporal answer change when the readout itself can create apparent revision. Known controls and counterfactual interventions are therefore antecedents, not priority claims. We have not compared our diagnostics against independent MIB methods or established that this resource improves their evaluation.',
    'We use its public IOI data for compensation experiments, rather than submitting a circuit-localization or causal-variable solution. Our target is attribution of recovery when both the continuation and its reader can change. We have not established improved evaluation of independent MIB methods.')
insert='''
[InterpBench](https://arxiv.org/abs/2407.14494v3) trains semi-synthetic networks to align with specified programmes. We use its published weights and RASP/Tracr references to evaluate narrow counterfactual predictions, with head-level and MLP-level interventions. This supplies an external reference for a declared variable exchange; it does not make every network output or every out-of-domain intervention an oracle. The distinction matters empirically in Section 5.10.
'''
paper=paper.replace('\n[Tyen et al.]',insert+'\n[Tyen et al.]',1)
paper=paper.replace("Add \\(Y_{1w}=D(z+\\delta+T_{\\delta'})\\), where delta-prime perturbs a different support with the same norm and sign. Given reference loss L, measure",
    "Add \\(Y_{1w}=D(z+\\delta+T_{\\delta'})\\), with a separately declared mismatched perturbation. The historical Gaussian experiments match norm and sign on another support; the native-activation experiments use another donor at the same site. Given reference loss L, measure")
paper=paper.replace('The mismatched control matches norm and sign, not difficulty or full latent geometry.',
    'The meaning of specificity depends on this control policy. Equal support does not imply equal norm, and matched norm does not imply matched difficulty or latent geometry. A positive s therefore measures an advantage over the declared mismatch, not a unique representation of error identity.')
paper=paper.replace('The historical mismatched-tail controls below exchange perturbation supports. The prospective IOI extension instead plans distinct native donors at the same head and token position, reducing that support change while retaining actual donor norms.',
    'The historical Gaussian controls exchange perturbation supports. IOI instead uses distinct native donors at the same head and token position, reducing that support change without equalizing donor norms.')

methods=r'''### 4.6 IOI: native head resampling and a separate backup-head intervention

We use the same frozen GPT-2 small checkpoint as in Section 4.4, without further training, and public [MIB–IOI](https://huggingface.co/datasets/mib-bench/ioi) data. A deterministic input-only selection retains 128 official development families and 512 public test families. Each includes the native prompt A, an answer-changing `s2_io_flip_counterfactual` prompt B, and an answer-preserving `s1_io_flip_counterfactual` prompt Aplus. Metadata, candidate order and tokenization are checked before inference. Candidates must each be one token with an initial space, and variants must have equal token lengths. No selected source is excluded for an incorrect native answer or an unfavorable effect. Nine literal templates occur in each selection, with no template string shared between development and test. We explicitly prepend GPT-2 BOS; this is not a claim of bitwise equivalence to the complete MIB evaluation pipeline.

The primary intervention replaces the 64-dimensional pre-projection output of L9H9 at the last prompt position with its native value from another A prompt. Other heads at that layer remain native. The cutoff follows the layer's complete attention update and precedes its MLP. The continuation recalculates that MLP and the remaining blocks, while retaining the recipient's earlier-position representations at each block. The final LayerNorm and unembedding form the reader. Two donors are assigned by fixed offsets 37 and 73 within the partition, without selection on names or effects. The five conditions are evaluated for each A anchor and donor. L9H6 and L10H0 are declared secondary targets, each with its own cutoff; their natural state-swap experiments coincide when the cutoff is the same, so those outcomes are not independent replications.

Development must show mean frozen-tail damage of at least 0.05 nat and damage above 0.01 nat in at least 25% of families, with software checks passing. Otherwise this branch stops before test. The test uses the same practical margins: 0.05 nat for damage, and 0.01 nat for recovery, specificity and the selective block effect. Twenty thousand bootstrap resamples retain the two donors within each source family. The four primary intervals for D, R, S and B use 98.75% percentiles; interactions, logit measures and the two secondary targets are descriptive. Reusing sources as donors creates dependencies between families that this bootstrap does not fully remove. Its intervals are conditional approximations, not guaranteed population coverage.

For corroboration, we prevent the published strictly downstream backup heads from adapting by replaying their native outputs. For the layer-9 targets, the group is L10H1/H10/H6/H2 and L11H9/H2. Its control group is L10H0/H3/H4/H5 and L11H0/H1, matching both count and layer distribution. L9H0/H7 are simultaneous with the target and are not treated as downstream. For L10H0, only the two layer-11 backups and their two controls remain downstream. The control group is not assumed inert. We report both absolute changes from Y11 and (B=L_{mathrm{backups}}-L_{mathrm{controls}}); a positive B alone would not establish that blocking backups hurts recovery.

Alongside NLL and candidate-name logit difference, we reproduce centred-logit self-repair as (Deltaell_y^{c}-Delta DE), with the original final-normalization factor used for DE. The direct-effect change uses the actual donor-minus-recipient activation, not the zero-ablation approximation (-DE_{mathrm{native}}). These logit quantities are not NLL recovery. Exactness among the two names and over the whole vocabulary is reported separately. E additionally executes the full input/state/tail factorial for A/B and A/Aplus; its state exchange concerns the entire last-position boundary vector, not a single head. Protocols and scoring definitions are retained in the [IOI study](../01_ioi/PROTOCOLE_TEST.md).

### 4.7 Pythia: changing the reader without an adaptive block continuation

We use the frozen [Pythia-160M checkpoint](https://huggingface.co/EleutherAI/pythia-160m), revision `50f5173d932e8e61f858120bcb800b97af589f46`, with 162,322,944 parameters. The final attention and MLP branches are parallel. Replacing a final attention-head output therefore cannot change the same layer's MLP, and no later attention/MLP block exists. This gives a structural control for attributing reader effects to downstream adaptation.

The fixed targets are L11H2 and L11H10. We sample 128 development rows and 512 test rows from the official validation and test partitions of [WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext), after input-only eligibility and duplicate checks. The first 32 tokens provide context and token 33 is the reference. Two other sequences, at fixed offsets 1 and 17, supply native activations at the same final-position head. There is no high-direct-effect position selection or ratio clipping. Multiple rows may come from one article; shared donors and document dependence limit the descriptive bootstrap.

The five conditions cross native/corrupted residuals with native, corruption-matched and other-donor final-normalization denominators. Residual centering is recomputed; only the denominator is fixed or exchanged. Here R, I and S are explicitly reader contrasts. The corresponding adaptive block-continuation effect is absent by the computational graph. We report NLL, vocabulary exactness, centred logits, raw self-repair and scale ratios, using 10,000 bootstrap resamples of whole sequences with both heads and donors grouped. The development gate requires technical parity, not a favorable recovery effect. WikiText and unselected positions differ from the prior Pile experiment, so this is a targeted structural test, not a reproduction of its reported effect percentages. [Protocol](../03_pythia/protocol.json).

### 4.8 InterpBench: predicting a future exchange rather than assigning a mechanism class

The external-reference evaluation uses published InterpBench checkpoints: cases 8 and 33 for development, then 256 distinct source sequences each for reserved cases 13, 18, 19 and 21. Programmes and checkpoint files are pinned. Programmes are executed through the published RASP functions and the official Tracr-to-Craft compiler. The learned networks run in the official TransformerLens backend. Neither the oracle nor the chosen programme is inferred from diagnostic predictions.

Provenance is important. The public code corrected case 13 on 9 September 2024, after the retained checkpoint and circuit files were already published. Their file identities are unchanged across the compared Hugging Face snapshots. We therefore use the pre-correction programme, recovered from the parent of that source change before reserved inference. Its operational labels retain the historical function, including its defects; they are not silently relabeled to match a corrected intuitive definition of numerical trend. For cases 18 and 21, some compiler auxiliaries lack directly exchangeable RASP nodes. They remain in the reference computation but are excluded from positive interventions whose semantic masks cannot be verified. Negative controls are absent from the complete Craft model, not merely absent from an exported edge list. The selected units and exclusions are [sealed in the C2 plan](../02_interpbench/data/c2_plan.json).

For each source and eligible head/MLP, two same-length donors provide test-time probes and a third donor supplies the future intervention. These are distinct donors within a query; the bank is reused across queries. Probes are available when making a prediction on that source. This is within-family extrapolation to a new donor, not prediction on an entirely unobserved family. The C2 inputs contain BOS and useful tokens without PAD. All categorical comparisons exclude BOS and padding; oracle comparisons also exclude positions whose RASP result is None, using the mask of the actual intervention.

The prediction is deliberately narrow: retain the native categorical output as a forecast of invariance, or abstain. Cautious O always abstains; forced O always forecasts invariance. The Y11-only rule uses the fraction of its two adapted probe outcomes that match the native argmax sequence. A second rule averages agreement across Y00/Y10/Y11/Y01; the five-condition rule also includes Y1wrong. Agreement uses all useful positions, including those without a defined RASP reference. These scores are agreement fractions, not calibrated probabilities. Thresholds 0.50, 0.75 and 1.00 are fixed in advance, with 1.00 primary. At the strict point, every included probe outcome must agree. All predictions are sealed before programme, Craft or learned-network responses to the third donor are computed. None of these rules is a six-way mechanism classifier, and the ablated rules are internal controls rather than independently developed literature methods.

We report coverage and contradictions against the neural future output, and separately against the programme on its defined mask. Programme risk can include native network–programme disagreement as well as a failed future forecast. Interchange accuracy for aligned variables and strict interchange accuracy for unaligned variables quantify network–programme conformity; they are not competing predictors. Results are reported per task and averaged equally across four tasks, without population p-values over correlated queries. E is a separate whole-memory 2-input × 2-state × 3-tail check and is never used secretly by the C predictor.

After the C2 results revealed weak native conformity for case 18, C3 compared the same 1,024 sources under the official BOS/PAD input convention. The programme reference still uses the useful tokens, as in the published data generator. This control was sealed before its own calculation but is explicitly post-hoc to C2; it changes no C2 prediction, threshold or result. [C2 protocol](../02_interpbench/PROTOCOLE_C2.md), [C3 protocol](../02_interpbench/PROTOCOLE_C3_PADDING.md).

'''
paper=paper.replace('## 5. Results',methods+'## 5. Results',1)
paper=paper.replace('### 5.2 Native performance is at ceiling','### 5.2 NCM native performance is at ceiling',1)
paper=paper.replace('### 4.2 Learned-model instantiation','### 4.2 NCM instantiation',1)
paper=paper.replace('### 4.5 Matched task, repeated training and prospective boundaries','### 4.5 Matched task, repeated training and declared boundaries',1)

ioi_results=f'''### 5.8 IOI: material damage, partial recovery and a failed specificity criterion

The development damage gate passes: mean D is 2.46181 nat and 127/128 families exceed 0.01 nat. We then evaluate all 512 reserved families without filtering native failures or perturbation outcomes. For the primary L9H9 intervention, D is **{ioi('','D')} nat**, interval {ci('','D')}; R is **{ioi('','R')} nat**, interval {ci('','R')}. Recovery is substantial but incomplete: total regenerated ablation damage remains {ioi('','ordinary_ablation_damage')} nat. Vocabulary exactness is {count('','Y00')}/1,024 natively, {count('','Y10')}/1,024 with the native frozen tail, and {count('','Y11')}/1,024 with adaptation. Native outcomes are repeated for two donors, so these are 512 source families rather than 1,024 independent native examples.

Matching the adaptation to its donor adds a smaller effect, S={ioi('','S')} nat, interval {ci('','S')}, compared with the alternative donor at the same site. Blocking the announced backups raises NLL by {ioi('','block_backups_minus_adapted')} nat relative to Y11; the layer-matched control raises it by {ioi('','block_controls_minus_adapted')} nat. Their difference B={ioi('','B')} nat, interval {ci('','B')}, corroborates a contribution from this group under the declared intervention. It does not show that the backups are the only source of compensation or that every member is individually necessary.

The secondary heads reproduce recovery but not every criterion. L9H6 has D={ioi('_L9H6','D')}, R={ioi('_L9H6','R')} with interval {ci('_L9H6','R')}, and S={ioi('_L9H6','S')} with interval {ci('_L9H6','S')}. Vocabulary exactness improves from {count('_L9H6','Y10')}/1,024 to {count('_L9H6','Y11')}/1,024. For L10H0, D={ioi('_L10H0','D')} and R={ioi('_L10H0','R')} with interval {ci('_L10H0','R')}, while S={ioi('_L10H0','S')} has interval {ci('_L10H0','S')}. **The fixed 0.01-nat specificity criterion is not met for L10H0.** This does not establish zero specificity. Its vocabulary exactness still improves from {count('_L10H0','Y10')}/1,024 to {count('_L10H0','Y11')}/1,024. The selective block effects remain positive, with B={ioi('_L9H6','B')} for L9H6 and {ioi('_L10H0','B')} for L10H0; the later cutoff has only two downstream backups rather than six.

Centred-logit self-repair is positive at all three targets: {ioi('','self_repair')}, {ioi('_L9H6','self_repair')} and {ioi('_L10H0','self_repair')}. These values have different units and definitions from NLL recovery. L10H0 illustrates the distinction directly: a positive self-repair measure and useful NLL recovery coexist with failure to exceed the chosen matched-tail margin. This is not an error in the published measure; it is a narrower claim that the additional contrast tests.

Natural state exchanges give a different result from resampling one head. In the A/B factorial at the layer-9 boundary, regenerated tails produce the recipient's vocabulary target in only 7/1,024 off-diagonal exchanges, compared with 1/1,024 for the recipient-native frozen tail and 4/1,024 for the donor-native tail. For answer-preserving A/Aplus, the respective counts are 430, 832 and 922. At the layer-10 boundary these counts are 11, 0 and 4 for A/B, and 877, 842 and 922 for A/Aplus. Layer-9 E outcomes are identical for the L9H9 and L9H6 runs because E exchanges the same complete boundary vector. Local head recovery therefore does not imply successful recipient recovery after a broader state swap, even with recipient prefix representations retained. All 12 conditions remain in the [summaries](../01_ioi/results/test_summary.json); no global architecture label is assigned.

### 5.9 Pythia: reader effects without evidence of positive average repair

On 512 reserved sequences, the native model predicts 151 next tokens exactly. Four interventions per sequence—two heads and two donors—give 2,048 correlated trials. Frozen-factor damage is 0.002391 nat, descriptive 95% interval [−0.001605, 0.006869]. Releasing the final-normalization factor yields R=−0.000601 nat [−0.001589, 0.000141]. No positive average reader recovery is established. Exactness is 604/2,048 in the repeated native condition, 598/2,048 under both frozen and adapted factors, 604/2,048 for the native residual with adapted factor, and 597/2,048 with the other donor's factor. Frozen and adapted exactness has no paired gains or losses.

Mean centred-logit self-repair is 0.000824 [−0.005052, 0.006706]; the measured reader contribution is 0.000824 to the same displayed precision. The two heads differ in sign: H2 has mean self-repair 0.003459 and H10 −0.001812. Neither result was used to replace a target or select high-effect positions. The maximum per-trial discrepancy between self-repair and the reader identity is 0.000171, below the fixed 0.0002 numerical tolerance. The final MLP contribution is unchanged in the full-forward checks. Thus the experiment separates an observable reader effect from a nonexistent adaptive downstream block continuation; it does not recover a large mean repair effect on this unselected WikiText sample. Remaining reader contrasts and intervals are in Appendix A and the [retained summary](../03_pythia/results/test_summary.json).

### 5.10 InterpBench: a sealed prediction test with no demonstrated incremental advantage

The development oracle checks cover 256 sources and 1,536 exchanges, including 499 changed programme answers, with agreement among programme, Craft and learned-network outputs on all exchanges. The official neural backend passes its native replay and sham checks. On the reserved tasks, all 5,618 queries with a nonempty semantic mask agree between RASP and Craft. Fourteen further queries have no defined categorical target; they are retained as unscorable, not credited as successes.

Network–programme conformity varies by task. Historical case 13 solves 256/256 native sources and matches the programme on 1,024/1,024 aligned and 512/512 unaligned future exchanges. Case 19 solves all 253 native sources with defined outputs; three have no defined native target. Its evaluated exchanges match in 507/507 aligned and 759/759 unaligned cases. Case 21 solves 256/256 native sources and matches in 768/768 aligned and 1,277/1,280 unaligned exchanges. **Case 18 fails the fidelity check:** only 119/256 native sequences are exact, with 116/256 aligned and 232/512 unaligned exchanges matching the programme. We retain the case in the aggregate instead of removing an inconvenient task.

At the strict agreement threshold, cautious O emits no claims and has undefined risk. Forced O covers every query and has task-averaged neural contradiction risk 41.04%. The Y11-only rule covers 55.74% with 2.39% risk; omitting only the mismatched condition gives 50.90% coverage and 2.34% risk. The five-condition rule covers 50.52% with 2.35% risk. These are finite task averages, not a statistical ranking at matched coverage. The additional mismatch check removes 23 assertions without removing a neural contradiction. Two programme contradictions are removed, both in the already nonconforming case 18. **An incremental predictive advantage from all five conditions is not established.** The lower-threshold results are also retained rather than selecting the most favorable point.

For the five-condition rule, programme contradictions are 14/578 in case 13, 217/439 in case 18, 1/545 in case 19 and 3/1,316 in case 21. Their equal-task mean risk is 13.07%. This is not directly interchangeable with neural risk: programme scoring uses its defined-position mask and can reflect native model–programme mismatch. A separately labeled post-hoc audit puts neural scoring on the same masks; it does not change any prediction. The high risk in case 18 cannot be assigned entirely to a failed attribution rule.

The padding control does not resolve this discrepancy. With the official padded input convention, case 18 is exact in 122/256 native sources, while the padded Craft reference matches the useful-token programme in 256/256. Cases 13, 19 and 21 remain exact on their defined native references with and without padding. C3 therefore rules out padding alone as an explanation in this comparison; the cause of case 18's mismatch remains unestablished. Its post-hoc results do not replace C2.

The separate E check contains 55,296 conditions. All 18,432 regenerated whole-memory conditions reproduce donor-native logits exactly, as required by the blocked recipient path. This validates the adapter's stated access contract, not a discovered internal algorithm. Full task counts, exclusions, risk–coverage points and [verification records](../02_interpbench/results/c2_verification.json) are retained with the [C2/C3 report](../02_interpbench/RAPPORT_C2.md).

'''
paper=paper.replace('## 6. What the case studies validate',ioi_results+'## 6. What the case studies validate',1)

discussion='''## 6. What the experiments support

The IOI results address the main weakness of the early case studies: there is now a substantial deficit to recover. Freezing downstream contributions exposes large damage, adaptation reduces that damage, and preventing the announced backups from adapting is more harmful than a layer-matched control. Together these interventions support a bounded functional contribution from those heads. They are consistent with the established IOI compensation literature rather than a new discovery of self-repair.

Recovery and perturbation matching remain separate findings. The matched-tail advantage is small relative to recovered NLL, and the late L10H0 target misses the practical specificity margin despite positive self-repair and recovery. The mismatch donor shares the intervention site but need not have the same norm or difficulty. Our inference concerns this donor policy; it does not establish a unique error code or semantic description inside the network. The choice of a useful-effect margin also matters, so a failed threshold should not be rewritten as proof of an absent effect.

The choice of intervention matters just as much. Resampling one IOI head is recoverable; exchanging the complete last-position boundary state often fails to restore the recipient, even for an answer-preserving input change. In the earlier NCM study, successful regenerated swaps were already matched by a particular native frozen continuation. These findings do not conflict. They concern different state units, remaining information and tails. A frozen recipient-native tail is an intervention that carries information from a prior recipient run, not an architecture class or a free source of answers.

Pythia supplies a complementary structural check. A final reader can change its response to an intervened residual even when no downstream attention/MLP computation can respond. Our unselected WikiText sample establishes no positive mean reader recovery, but the graph and numerical checks still separate reader changes from block adaptation. This is why a centred-logit compensation quantity should not be used alone to identify the locus or utility of adaptation.

InterpBench tests an additional ambition: whether more probe controls yield better predictions of an unseen intervention. The cautious five-condition rule makes far fewer forced invariance claims than the naive observation rule. Yet its advantage over the less restrictive intervention rules is not demonstrated at comparable coverage. In particular, adding the mismatched condition prunes claims without removing a neural contradiction at the strict point. This negative result limits the proposed predictor; it does not make specificity irrelevant to every attribution question, nor does it invalidate the separately corroborated IOI effect.

The programme reference also has a domain. Recovering the historical case-13 source was necessary to align the question with the retained checkpoint. Compiler auxiliaries without a verified semantic mask remain outside this evaluation, and case 18 is an observed model–programme fidelity failure under both tested input conventions. Treating every discrepancy as an attribution error would conflate resource conformity with diagnostic quality. Our external objects are independently constructed; the present implementation and interpretation are still internal.

Finally, none of these outcomes identifies a general AR-versus-parallel signature. The matched-task experiment controls parameter count, initialization and update budget, but its objectives differ, and its parallel network is a conditional one-shot proxy rather than an implementation of the NCM tokenizer or VQ-VAE. Boundary and access findings are supported; a pure causal effect of the generation paradigm is not. Similarly, additional text-model evidence does not measure a Wyner rate or establish that pooling caused a coordination deficit.
'''
paper=re.sub(r'(?s)## 6\. What the case studies validate\n.*?(?=\n## 7\.)',lambda _:discussion,paper,count=1)

limitations='''## 8. Limitations and reproducibility

Across the studies we use 15 distinct learned checkpoints: one NCM, one GPT-2 reused for copying and IOI, six small Transformers trained on the matched task, one Pythia, and six InterpBench networks including development. This is broader than one learned architecture but still a small, selected collection. Only four InterpBench tasks form reserved programme evaluation, and some semantic components are excluded. The synthetic task shares its alphabet and rule across partitions, the earlier copying prompts are manually selected, and Sudoku contamination checks cover only some isomorphisms. Swapped states can be out of distribution conditional on the recipient; native activation donors do not make every state combination natural.

The source family, not a token, donor exchange or software assertion, is the relevant grouping unit. Reused donor banks induce dependencies beyond the family bootstrap. WikiText rows may share an article. The three small-model seeds share evaluation families. We therefore report conditional approximate intervals for IOI and descriptive sequence intervals for Pythia, and finite per-task counts for InterpBench. Secondary heads and post-hoc checks do not become independent confirmatory evaluations through repetition. Nor are the risk–coverage points a matched-budget or matched-coverage competition between independently developed methods.

### 8.1 Protocols, numerical corrections and audits

The freezes are local timestamped records, not externally registered immutable protocols. The constructed library and its diagnostics share an author, and the experiment adapters, analyses and audits were implemented by research agents. Other agents reviewed theory and implementation, but no independent human replication has been performed. Programmes, public datasets and hypotheses from prior work provide external evaluation objects; they do not make this an externally conducted evaluation.

Numerical failures are part of the retained record. In IOI, a nonfinite low-dimensional FP32 vector–matrix measurement was localized after development, without opening test outputs. The direct-effect measurement was moved to FP64 and checked against NumPy and scalar arithmetic; the neural model and interventions remained FP32. The behavior was not reproduced in every memory arrangement, so its low-level cause is not claimed. In Pythia, the initial manual reader missed the fixed parity tolerance. A native-kernel implementation of the same reader and a separately logged FP64 direct-effect measurement passed without relaxing thresholds. InterpBench's first FP32–FP64 comparison also failed its tolerance; equal-precision comparison with the official backend then separated numerical differences from equation mismatch. None of these development corrections silently changes the original failure or test criterion.

For each IOI head, a supplementary audit compares 40 full official forwards on four fixed sources, including blocking and E, with the stored results; logits agree bitwise. The reader/direct-effect identities are checked separately. These additional audits were added after experimental freezing as software checks, not new hypothesis tests. Pythia's native reader is exact and its corrupted full-forward replay stays within the fixed 0.0002-logit tolerance. The independent-formula test audit uses the same first 32 records and tolerance through a memory-mapped implementation added after results. InterpBench's audit reconstructs predictions and contradiction counts from the saved logits and verifies seals; the mask clarification and padding control are explicitly post-hoc.

### 8.2 Evidence and availability

The archived version-3 Markdown and PDF remain a record of the earlier studies; they do not contain the experiments newly integrated here. This manuscript's previous editorial version is [preserved](archives/avant_integration_resultats_20261025/PAPER_REVISE.md). No new model run or scientific threshold was introduced by integrating the text.

The [original evidence ledger](../../article_benchmark_coordination/PREUVES.md) covers calibration and NCM. The [copying report](../../benchmark_multimodel_ar/RAPPORT.md), [matched-task report](../../benchmark_tache_commune/RAPPORT.md) and [access-contract note](../../benchmark_tache_commune/CONTRATS_ET_ESTIMANDS.md) document the earlier neural studies. New evidence is mapped in [the integration ledger](PREUVES_INTEGRATION.md), with [IOI protocols and reports](../01_ioi/RAPPORT.md), [secondary replications](../01_ioi/REPLICATIONS_SECONDAIRES.md), [Pythia results](../03_pythia/RAPPORT.md), and [InterpBench C2/C3](../02_interpbench/RAPPORT_C2.md). Each retains raw outputs, freezes and audits. These links identify local study artifacts; a public release, official MIB integration and a submitted manuscript are not claimed.

The measured IOI runner times are approximately 432, 422 and 436 seconds for the primary and two secondary heads. Pythia's reserved runner takes 63.5 seconds; InterpBench's probe/prediction and falsification phases take 83.8 and 103.8 seconds. Loading and phase-specific work are included, audits and installation are separate, and other work sometimes shared the machine. These are session costs, not isolated latency benchmarks or evidence of a method's computational superiority.

The remaining validation questions are specific: reproduce a principal contrast independently on new families; compare methods at matched access and coverage; extend semantic oracle coverage to the excluded compiler auxiliaries; and investigate the case-18 discrepancy in a new, separately specified campaign. Choosing replacement tasks or thresholds after these results would not answer them on the present reserve.
'''
paper=re.sub(r'(?s)## 8\. Limitations, reproducibility and research status\n.*?(?=\n## 9\.)',lambda _:limitations,paper,count=1)
conclusion='''## 9. Conclusion

The strongest positive result is conditional and concrete: GPT-2's continuation recovers much of the damage from resampling each of three IOI Name Mover heads, and a separate blocking intervention supports a contribution from published backups. That recovery is not a universal signature. The late target misses the declared specificity margin, larger state exchanges behave differently, and Pythia shows why the final reader must be distinguished from an adaptive block continuation.

The InterpBench prediction test is less favorable. All five probe conditions do not establish an incremental predictive advantage over the ablated intervention rules at comparable coverage. A checkpoint–programme mismatch also survives the padding check, limiting which oracle-based errors can be assigned to a diagnostic. These outcomes belong in the evaluation rather than being removed to produce a single success rate.

Our parallel Transformer remains a proxy for one-shot generation: all target symbols are predicted together. It does not test the NCM VQ-VAE, tokenizer or concept hierarchy, and the differing training objectives prevent a pure causal AR-versus-parallel conclusion. Likewise, common-information theory supplies assumptions for a future experiment, not a measured coordination rate in the models studied here.

The resulting resource supports bounded claims about damage, recovery, matching and access. It does not yet establish a reference benchmark or a generally accurate classifier of internal mechanisms. Its next test is comparative use and independent replication, with the present positive, negative and uninformative outcomes left intact.
'''
paper=re.sub(r'(?s)## 9\. Conclusion\n.*?(?=\n## References)',lambda _:conclusion,paper,count=1)

refs='''
- Gupta, R., Arcuschin, I., Kwa, T., and Garriga-Alonso, A. (2024). [InterpBench: Semi-Synthetic Transformers for Evaluating Mechanistic Interpretability Techniques](https://arxiv.org/abs/2407.14494v3). NeurIPS Datasets and Benchmarks; arXiv version 3, 2025.
- Biderman, S., et al. (2023). [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373). [Pythia-160M checkpoint](https://huggingface.co/EleutherAI/pythia-160m), fixed revision recorded in Section 4.7.
- Merity, S., Xiong, C., Bradbury, J., and Socher, R. (2016). [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). WikiText corpus; [retained dataset resource](https://huggingface.co/datasets/Salesforce/wikitext), `wikitext-2-raw-v1`.
- MIB–IOI dataset. [Public data and counterfactual variants](https://huggingface.co/datasets/mib-bench/ioi); frozen revision in the IOI provenance record.
- InterpBench implementation and models. [Code](https://github.com/FlyingPumba/InterpBench), [checkpoints](https://huggingface.co/cybershiptrooper/InterpBench), and [case-13 source correction](https://github.com/FlyingPumba/InterpBench/commit/bceb41a5c272fb1b1a91611396c89bbf81c87ee0).
'''
paper=paper.rstrip()+'\n'+refs

appendix='''
## Appendix A. Additional retained numerical results

The main text gives the material-effect contrasts and conclusions. This appendix retains the other measured quantities without treating them as additional independent tests. Full family-, template- and intervention-level outcomes remain in the linked summaries.

### A.1 IOI contrasts and exactness

Before interventions, A/B/Aplus vocabulary exactness is 476/512, 449/512 and 446/512; two-candidate exactness is 512/512, 498/512 and 506/512. These native outputs are shared across targets. The C corruption experiment uses A anchors only; the E comparisons retain all variants and native errors.

'''
for suffix,name in [('', 'L9H9 primary'),('_L9H6','L9H6 secondary'),('_L10H0','L10H0 secondary')]:
    appendix+=f"**{name}.** Interaction I={ioi(suffix,'I')} nat; direct-effect change={ioi(suffix,'delta_DE')} centred logit; self-repair={ioi(suffix,'self_repair')} centred logit. Total regenerated ablation damage is {ioi(suffix,'ordinary_ablation_damage')} nat. B={ioi(suffix,'B')} {ci(suffix,'B')}; blocking backups changes loss by {ioi(suffix,'block_backups_minus_adapted')}, controls by {ioi(suffix,'block_controls_minus_adapted')} nat.\n\n"
    s=read('01_ioi/results/test'+suffix+'_summary.json')
    appendix+='Vocabulary exactness in Y00/Y10/Y11/Y01/Y1wrong/backup-block/control-block order: '+', '.join(str(s['accuracy'][n]['correct_vocab']) for n in s['accuracy'])+'/1,024. Two-candidate exactness in the same order: '+', '.join(str(s['accuracy'][n]['correct_candidate']) for n in s['accuracy'])+'/1,024. '
    appendix+=f"The local D/R/S thresholds retain {s['claim_coverage']['complete']}/512 families, compared with {s['claim_coverage']['without_specificity']}/512 when S is omitted. These are observed effect-threshold counts, not mechanism-classification accuracy.\n\n"
appendix+='''### A.2 Pythia reader contrasts

Across both heads, I=0.000007095 nat, descriptive 95% interval [−0.000230012, 0.000147205], and S=−0.000109216 [−0.000355497, 0.000019042]. The mean native/corrupted scale ratio is 1.00013290 [0.99988247, 1.00037891]; 1,035/2,048 ratios exceed one. Positive individual reader recovery occurs in 1,016/2,048 trials and positive raw self-repair in 1,053/2,048; their signs do not imply positive mean recovery.

H2 has D=0.00555188, R=−0.00098630 and S=−0.00019393 nat. H10 has D=−0.00077030, R=−0.00021589 and S=−0.00002450 nat. The corresponding direct-effect changes are −0.02926254 and −0.00723953 centred logit. The [summary](../03_pythia/results/test_summary.json) retains all per-head intervals. No favorable head or high-effect position subset replaces these results.

### A.3 InterpBench agreement-threshold grid

Coverage, neural contradiction risk and programme contradiction risk below are equal-task means in percent. Neural risk uses all useful positions; programme risk uses the intervention's defined mask. Scores are agreement fractions with different included conditions, so equal score thresholds do not mean equal coverage or calibrated confidence.

'''
mac=read('02_interpbench/results/c2_summary.json')['macro_average_over_tasks']
for method,label in [('C_without_frozen','Y11 only'),('C_without_mismatch','Without Y1wrong'),('C_full','Five conditions')]:
    parts=[]
    for t in ['0.5','0.75','1.0']:
        x=mac[method][t];parts.append(t+': '+', '.join(f"{100*x[k]:.2f}" for k in ['coverage','risk_network','risk_oracle']))
    appendix+=f"- **{label}**, thresholds 0.50/0.75/1.00: "+'; '.join(parts)+'.\n'
appendix+='''
Forced O has 100% coverage, 41.04% neural risk and 41.88% programme risk at every threshold. Cautious O has zero coverage and undefined risks. The [post-hoc mask audit](../02_interpbench/results/c2_mask_and_c3_audit.json) preserves neural scoring on the programme-defined positions as well; its values are not substituted for the original grid. Finite source/task counts, excluded auxiliaries and empty masks remain visible in the [full summary](../02_interpbench/results/c2_summary.json).
'''
paper=paper.rstrip()+'\n'+appendix

assert 'Prospective external-validation programme' not in paper
assert 'It reports no new experiment' not in paper
assert '### 5.8 IOI' in paper and '### 5.10 InterpBench' in paper
(D/'PAPER_REVISE.md').write_text(paper.rstrip()+'\n')
(D/'CLAIMS_INTEGRATION.json').write_text(json.dumps(dict(
    archived_manuscript=str(BASE.relative_to(D)),source_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
    generated_numeric_claims=claims,scientific_computations_executed=False,
    note='Direct JSON values used for key IOI claims; remaining static rounded counts are checked in the editorial audit.'),indent=2)+'\n')
print('Integrated manuscript:',len(paper.split()),'words;',len(claims),'source-linked numerical insertions')

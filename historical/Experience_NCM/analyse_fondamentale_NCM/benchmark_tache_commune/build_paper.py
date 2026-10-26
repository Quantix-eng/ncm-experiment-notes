"""Post-results editorial assembly; never changes source manuscripts or scores."""
from pathlib import Path
import os
import re

D=Path(__file__).resolve().parent
SOURCE=D.parent/'benchmark_multimodel_ar/PAPER_MULTIMODEL.md'


def once(text,old,new):
    assert text.count(old)==1,old[:90]
    return text.replace(old,new)


def rebase(match):
    link=match.group(1)
    if link.startswith(('https://','http://','#')):return ']('+link+')'
    path,sep,fragment=link.partition('#')
    destination=os.path.relpath((SOURCE.parent/path).resolve(),D)
    return ']('+destination+(sep+fragment if sep else '')+')'


if __name__=='__main__':
    text=re.sub(r'\]\(([^)]+)\)',rebase,SOURCE.read_text())
    text='# Attributing Apparent Revision: Calibration, Matched Tasks and Access Controls\n'+text.split('\n',1)[1]
    first=text.index('Working manuscript, version 2,')
    end=text.index('\n',first)
    text=text[:first]+('Working manuscript, version 3, 25 October 2026. This version retains the analytical, NCM and GPT-2 evidence and adds six prospectively specified small-model trainings on a matched task. Corruption calibration failed its declared criterion and the fixed fallback is explicitly retained. Historical protocols, outcomes and manuscripts remain unchanged. No external validation or submission is implied.')+text[end:]
    a=text.index('A changing intermediate answer');b=text.index('\n## 1. Introduction',a)
    abstract='''A changing intermediate answer does not identify the computation responsible for that change. We develop a revision-attribution protocol with analytical calibration, crossed continuations and explicit access contracts. A complete diagnostic recognizes all 344 announced constructed-program instances as expected by construction; learned models receive functional descriptions rather than mechanism ground truth. A Sudoku NCM has input-dominant preference in 15/16 families, with regenerated swap success matched by native frozen controls. GPT-2 has input-dominant preference in 8/16 copy-task families and abstains on eight native failures. We then train paired small autoregressive and parallel Transformers on a common rotation task with three seeds, separate development and 40 evaluation families. All six solve all 120 held-out variants. A development-only dose grid fails the common informative-damage criterion, and its prespecified fallback is reported without retuning. At the primary cutoff after two blocks, regenerated state swaps always follow the donor for both systems; at an earlier declared cutoff, outcomes differ by training and context access. Exchanging the entire memory blocks the recipient path exactly as specified. Many strict labels abstain despite perfect native task accuracy, and adapted continuations can worsen perturbed outputs. These studies support executable evaluation across learned systems and reveal dependence on corruption, boundary and access. They do not establish a universal architecture taxonomy, independent classifier validity or a measured Wyner coordination rate.'''
    text=text[:a]+abstract+'\n'+text[b:]
    text=once(text,'The contribution is a protocol and two case studies;',
              'A third empirical study adds a matched task, three paired trainings, a separate corruption-development phase and a blocked-path access control. The contribution is a protocol and three empirical studies;')
    addendum=(D/'PAPER_ADDENDUM.md').read_text()
    methods=addendum.split('## 2. Models, data and prospective boundaries\n',1)[1].split('## 4. Results',1)[0]
    methods=methods.replace('## 3. States, contracts and outcomes','#### States, contracts and outcomes')
    text=once(text,'## 5. Results','### 4.5 Matched task, repeated training and prospective boundaries\n\n'+methods+'## 5. Results')
    results=addendum.split('## 4. Results\n',1)[1].split('## 5. Audit and implications',1)[0]
    results=results.replace('### 4.1','####').replace('### 4.2','####').replace('### 4.3','####')
    text=once(text,'## 6. What the case studies validate','### 5.7 Matched-task and access-control results\n'+results+'## 6. What the case studies validate')
    discussion='''The matched-task study controls data, model size, paired initialization, batches and update budget across six trainings. This improves on the earlier cross-task comparison, but training objectives still differ and the parallel Transformer is not a faithful NCM. Three seeds expose training dependence rather than justify population inference over all trained models. Their shared evaluation families cannot be pooled as 120 independent sources.

The blocked-path track supplies a restricted structural truth on learned networks: whole-memory regeneration has no recipient channel. Its zero effect is expected and audits the adapter; it is not 100% accuracy against an internal-algorithm label. Recipient-native frozen tails remain external carriers of recipient information, so their success is compatible with that invariance. Different access contracts change the causal quantity being estimated, not the learned weights. See the [estimand note](CONTRATS_ET_ESTIMANDS.md).

The study also reveals limitations in the evaluation itself. Its development dose grid fails the prespecified common-damage criterion; this is not a successfully calibrated repair benchmark. Strict ratios yield abstentions even under perfect native accuracy and donor-following argmax outcomes. No post-hoc clipping, threshold relaxation or stronger dose is used. A future rule must calibrate these choices prospectively against new data and independently developed methods. The current negative results are part of the benchmark evidence.

'''
    text=once(text,'## 7. Coordination and common information: a conditional link',discussion+'## 7. Coordination and common information: a conditional link')
    old='Two learned checkpoints were used, one per architecture on different tasks, with sixteen source families per task. Both corpora are nonuniform; Sudoku clue counts and training relationships are incompletely controlled, and the word-copy sources are manually selected. The tasks are not a matched comparison, and half of the GPT-2 families fail the native-anchor gate.'
    new='Eight learned checkpoints are included: two pretrained NCM/GPT-2 systems and six small Transformers trained here. The first two studies use different tasks with sixteen source families each; the new common-task study uses forty evaluation orbits and three paired seeds. Sudoku clue counts and training relationships are incompletely controlled, word-copy sources are manually selected, and the rotation task is simple and shares its rule and alphabet with training. Half of the GPT-2 families fail the native gate; the common-task models instead reach native ceiling. Neither study establishes generalization to arbitrary tasks or architectures.'
    text=once(text,old,new)
    text=once(text,'No secret external-method evaluation, independently developed diagnostic or matched-task architecture comparison was performed. A second learned architecture is executed in this version, with an explicitly different access contract.',
              'No secret external-method evaluation or independently developed diagnostic was tested. A common-task learned comparison is executed in this version, while the AR teacher-forcing and parallel-query training objectives remain different.')
    text=once(text,'For a broader benchmark claim, the resource still requires independent methods, a matched-task learned comparison, explicit white-box access tracks, corruption calibration on separate development data and reserved evaluation families.',
              'For a broader benchmark claim, the resource still requires independent methods, larger matched tasks including a faithful NCM, external evaluation and successful corruption calibration. The new task supplies explicit access tracks and reserved families, but its development calibration fails the declared informative-damage criterion.')
    anchor='## 9. Conclusion'
    provenance='The new [matched-task protocol](PROTOCOLE.md), [report](RAPPORT.md), [summary](results/summary.json), [audit](results/verification.json) and [README](README.md) retain the six trainings and both access contracts. The internal audit passes 42,240 exact recompositions, 576 explicit regenerations, 19,200 scalar losses, 72 native replays and 3,840 blocked-path conditions; these are repeated software checks, not independent samples. Editorial assembly of this version is recorded in the [journal](JOURNAL.md).\n\n'
    text=once(text,anchor,provenance+anchor)
    text=once(text,'including cases where the experiment remains uninformative about repair or abstains because native anchors fail.',
              'including cases where corruptions fail their damage calibration, native anchors fail, strict labels abstain under correct anchors, or regenerated continuations worsen perturbed performance.')
    text+='\n- Geiger, A., Lu, H., Icard, T., and Potts, C. (2021). [Causal Abstractions of Neural Networks](https://papers.nips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html). NeurIPS 34.\n'
    text+='- Zhang, F., and Nanda, N. (2024). [Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://openreview.net/pdf?id=Hf17y6u9BC). ICLR.\n'
    (D/'PAPER_MATCHED_TASK.md').write_text(text)
    print('Version 3 assembled; no scientific outcomes or source manuscript modified.')

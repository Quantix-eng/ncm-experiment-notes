"""Post-experiment editorial integration. Does not alter historical manuscript."""
from pathlib import Path
import re

D=Path(__file__).resolve().parent
SOURCE=D.parent/'article_benchmark_coordination/PAPER.md'


def replace_once(text, old, new):
    assert text.count(old)==1, old[:100]
    return text.replace(old,new)


if __name__=='__main__':
    text=SOURCE.read_text()
    text=replace_once(text,
        '# Attributing Apparent Revision in Parallel Generators: A Calibrated Protocol and a Frozen-NCM Case Study',
        '# Attributing Apparent Revision in Neural Generators: A Calibrated Protocol with NCM and Autoregressive Case Studies')
    text=replace_once(text,
        'Working manuscript, 25 October 2026. Methodological study with an initial empirical case study. The historical experiments are retained without modification; this manuscript adds explicitly post-hoc analyses of saved traces. No publication, public release or external validation is implied by this working version.',
        'Working manuscript, version 2, 25 October 2026. This version adds a locally frozen GPT-2 experiment to the retained NCM study and analytical calibration. It preserves their original outcomes and distinguishes the new prespecified experiment from post-hoc descriptive exports. No public release, external validation or submission is implied.')
    start=text.index('A changing intermediate answer')
    end=text.index('\n## 1. Introduction',start)
    abstract='''A changing intermediate answer does not identify the computation responsible for that change. We develop a revision-attribution protocol with analytical controls, crossed continuations and explicit access contracts. Constructed systems share identical readout trajectories despite different continuation programs; a complete diagnostic recognizes all 344 announced library instances as expected by construction. We then instantiate the protocol on two frozen learned architectures. A Sudoku NCM solves all 48 counterfactual variants; its preference diagnostic is input-dominant in 15/16 source families and abstains in one. Local perturbations create no substantial repair opportunity, and regenerated state-swap continuations are matched in exactness by recipient-native frozen controls. For GPT-2 small on 48 controlled word-copy prompts, native accuracy is 33/48. Swapping the last-position state after block six while retaining recipient context yields input-dominant preference in 8/16 families; the other eight fail the native-anchor gate and remain non-identifiable. Regenerated and recipient-native frozen tails give 19/32 and 18/32 recipient-correct swapped outcomes. Mean NLL recovery under local perturbation coexists with two correctness losses. These studies establish executable interventions on real FP32 states while illustrating limits on attribution. They do not identify internal-mechanism ground truth, separate architectures independently of task and access, or measure a Wyner common-information rate. The resource is a calibrated protocol with two learned-model case studies, requiring independent methods and external evaluation before a reference-benchmark claim.'''
    text=text[:start]+abstract+'\n'+text[end:]
    text=replace_once(text,
        'The contribution is a protocol and case study; we do not claim a reference benchmark, a new common-information theorem, or independently validated mechanism classification.',
        'A second learned-model study applies the protocol to GPT-2, preserving a recipient-context channel while exchanging its last-position state between layers. This broadens the executed architectures and tasks, and tests whether functional signatures can be treated as architecture labels. The contribution is a protocol and two case studies; we do not claim a reference benchmark, a new common-information theorem, or independently validated mechanism classification.')
    methods='''### 4.4 Autoregressive extension and its access contract

We additionally use the pretrained [GPT-2 small checkpoint](https://huggingface.co/openai-community/gpt2), revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, with 124,439,808 parameters. Inference uses CPU FP32, one thread, evaluation mode and eager causal attention in Transformers 4.57.1. No fine-tuning is performed. Sixteen manually chosen word pairs define prompts `The code is WORD. The code is`; A/B differ in one assigned-word token and Aplus changes the initial `The` to `My` without changing the target. All 48 prompts have eight tokens and a single-token reference with initial space. Exactness is the argmax over the entire vocabulary. Tokenization is checked before model outputs; no source is removed or replaced after observing performance.

We cut after block six of twelve. The exchanged state is the final prompt position's 768-dimensional residual vector. Earlier-token representations at every remaining block are retained from the recipient's native computation. Each block receives these representations and the current intervened last-position vector. Causality keeps earlier positions invariant. A tail consists of the last position's separate attention and MLP contributions from blocks 7–12, added in native FP32 order; the final normalization/head supplies the reader. Obtaining the context representations costs a prior computation; no speedup is claimed.

The intervention concerns depth within one next-token computation, not already-emitted-token time. It is a partial-state exchange with recipient memory retained, not a swap of all prompt information. This contract is explicit because it determines what an input-effect diagnostic can mean. The thirteen O readouts use the final reader on embeddings and every block output, without assuming intermediate calibration.

C tests two Gaussian directions on disjoint halves of the residual vector, both signs, at norm 0.1 times the state norm, giving 128 trials. E tests the same two 12-condition factorials for A/B and A/Aplus, giving 384 outcomes. Family aggregation, 20,000 bootstrap draws, four adjusted C intervals, the 0.01-nat margin and the E gates/ratios are unchanged. The margin is inherited rather than independently validated on text. Native failures cause abstention and remain in aggregate effects. All scripts, thresholds, data, weights and imported sources were locally sealed before inference at 2026-10-25T15:09:14Z, covering 1,291 files. The [full AR protocol](PROTOCOLE.md) records the contract and scope.

'''
    text=replace_once(text,'## 5. Results',methods+'## 5. Results')
    results='''### 5.6 GPT-2: context dominance is compatible with autoregression

Official/native reconstruction and sham continuation agree exactly for all 48 prompts. Native accuracy is 12/16 on A, 8/16 on B and 13/16 on Aplus: 33/48 overall. Both A/B anchors are correct in eight families. Intermediate readings contain corrections and degradations: B is correct in twelve sources after block ten and eight at the final block. These are internal readouts, not revisions of emitted text.

For C, D=0.00181115 nat/token, adjusted interval [0.00141522, 0.00213673]; R=0.00781555 [0.00088758, 0.01519674]; I=0.00181959 [0.00113743, 0.00242062]; S=0.00168670 [0.00070116, 0.00268459]. Mean frozen damage falls within the fixed equivalence margin and the joint useful-and-specific recovery criterion is not met. Y00 and Y10 are correct in 80/128 trials; Y11, Y01 and Y1wrong in 78/128. A disclosed post-hoc paired exporter finds no argmax change under frozen corruption, but two correctness losses and no gains in each adapted-tail condition. Thus improved mean NLL under regeneration coexists with reduced exactness. The exporter uses saved outputs and changes no frozen diagnostic.

Mean E context effect u is 3.76857 nat, descriptive 95% interval [1.23271, 6.39620]; state effect v is 0.30567 [0.19423, 0.42402]; interaction w is 0.01608 [-0.21121, 0.22946]. The unchanged rule diagnoses input-dominant preference in **8/16 families**, while **8/16 remain non-identifiable due to native-anchor failure**. Every family passing the native gate meets the other input-dominance conditions; reporting only this eligible subset would hide half of the corpus. All sixteen effects are retained.

Across 32 off-diagonal A/B swaps, regenerated tails give 19 recipient-correct outputs, 10 donor-correct and three others. Recipient-native frozen tails give 18, 11 and three; donor-native frozen tails give nine, 20 and three. A paired descriptive export finds one gain and no loss for regeneration against the recipient-native control. This local difference does not establish a universal repair mechanism. In A/Aplus swaps, regenerated, recipient-native and donor-native tails give 25/32, 26/32 and 25/32 correct outputs. Identical references are not normalized into a response-preference ratio.

The AR [audit](results/verification.json) recomposes all 1,024 C/E decoder outcomes, recalculates 640 scalar NLLs and reproduces scores, grouped C intervals, labels, norms, diagonals and invariances. Explicit sublayer calls replay 16 C and 16 E continuations on the first two families, plus six official native forwards. States and logits match exactly. Historical manifests and frozen inputs remain unchanged. This is an internal software audit, not an independent empirical replication.

'''
    text=replace_once(text,'## 6. What the case study validates',results+'## 6. What the case studies validate')
    anchor='## 7. Coordination and common information: a conditional link'
    extension='''The GPT-2 experiment adds a second learned architecture and a distinct task, with successful parity and crossed-outcome audits. It demonstrates that input-dominant preference can occur in an AR next-token computation when recipient context remains accessible. The usual append-only output interface does not determine this internal dependency. Nor is `frozen_recipient_native` an architecture class: it is a continuation policy applied in both studies.

The 8/16 AR and 15/16 NCM diagnostic counts cannot rank architectures. Task, native correctness, intervention geometry, retained memory and the meaning of the continuation axis differ. A future diffusion study likewise cannot assume a mixed signature before defining conditioning, state and remaining noise. No diffusion result is claimed here.

'''
    text=replace_once(text,anchor,extension+anchor)
    text=replace_once(text,
        'Only one learned checkpoint and 16 nonuniform source families were used. Their clue counts and training relationship are incompletely controlled.',
        'Two learned checkpoints were used, one per architecture on different tasks, with sixteen source families per task. Both corpora are nonuniform; Sudoku clue counts and training relationships are incompletely controlled, and the word-copy sources are manually selected. The tasks are not a matched comparison, and half of the GPT-2 families fail the native-anchor gate.')
    text=replace_once(text,
        'No secret external-method evaluation or comparison to another learned architecture was performed.',
        'No secret external-method evaluation, independently developed diagnostic or matched-task architecture comparison was performed. A second learned architecture is executed in this version, with an explicitly different access contract.')
    text=replace_once(text,
        'For a broader benchmark claim, the resource still requires independent methods, an additional learned family, explicit white-box access tracks, corruption calibration on separate development data and reserved evaluation families.',
        'For a broader benchmark claim, the resource still requires independent methods, a matched-task learned comparison, explicit white-box access tracks, corruption calibration on separate development data and reserved evaluation families.')
    text=replace_once(text,
        'All raw outcomes and local freezes are preserved. The [evidence ledger](PREUVES.md) maps manuscript claims to files; auxiliary analyses and integrity checks are in this directory.',
        'All raw outcomes and local freezes are preserved. The [original evidence ledger](PREUVES.md) maps calibration/NCM claims to retained files; the new [AR report](RAPPORT.md), [protocol](PROTOCOLE.md), [summary](results/summary.json) and [reproduction instructions](README.md) document this extension. The [journal](JOURNAL.md) distinguishes the frozen analysis from the later paired descriptive export.')
    text=replace_once(text,
        'Our constructed calibration and frozen-NCM application show how this can be done reproducibly, including cases where the experiment remains uninformative about repair.',
        'Our constructed calibration, frozen-NCM application and autoregressive extension show how this can be done reproducibly, including cases where the experiment remains uninformative about repair or abstains because native anchors fail.')
    # Only the two links local to the original article directory require rebasing.
    text=text.replace('](THEORIE_WYNER.md)','](../article_benchmark_coordination/THEORIE_WYNER.md)')
    text=text.replace('](PREUVES.md)','](../article_benchmark_coordination/PREUVES.md)')
    text+='\n- GPT-2 checkpoint and model card. [openai-community/gpt2](https://huggingface.co/openai-community/gpt2), fixed revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.\n'
    assert text.count('## Abstract')==1 and text.count('### 5.6')==1
    (D/'PAPER_MULTIMODEL.md').write_text(text)
    print('Wrote consolidated manuscript; original retained unchanged.')

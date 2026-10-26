"""Build a bounded cross-domain article from verified experiment outputs."""
from base import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv,subprocess,shutil,zipfile,platform,importlib.metadata
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def read(name):return json.loads((D/'results'/name).read_text())
def pc(v):return f'{100*v:.2f}%'
def interval(v):return f'[{100*v[0]:.2f}, {100*v[1]:.2f}]'
def save(fig,name):
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','svg','pdf']:fig.savefig(D/f'figures/{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

def figures(text,audit,screen,test):
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    r=audit['hard']['generated_train'];axes[0].plot(range(1,13),[np.exp(v['marginal_entropy']) for v in r],'o-',label='Generated, train (1,000)')
    axes[0].plot(range(1,13),[np.exp(v['positional_entropy']) for v in r],'s--',label='Position-conditioned')
    axes[0].set(title='Sudoku hard, 16 codes/level')
    for ax,task in zip(axes[1:],['lm1b','tinystories']):
        for stratum,style in [('random','o-'),('long','s--')]:ax.plot(range(1,13),[v['occupancy']['perplexity'] for v in text[task][stratum]['levels']],style,label=f'{stratum} contexts')
        ax.set(title=task.upper()+f", {text[task]['random']['levels'][0]['vocab']:,} codes/level")
    for ax in axes:ax.set(xlabel='Hierarchy level (1-based)',ylabel='Empirical code perplexity',yscale='log',xticks=[1,2,5,8,12]);ax.axhline(1,color='gray',lw=.8);ax.legend(fontsize=7)
    fig.suptitle('Code diversity: three published tokenizers; text samples 256 random / 128 long contexts');save(fig,'01_cross_domain_diversity')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,task in zip(axes,['lm1b','tinystories']):
        r=read(task+'_controls.json');ax.plot(range(1,13),np.array(r['within_four']['per_level_code_disagreement'])*100,'o-',label='Within 4-token blocks')
        ax.plot(range(1,13),np.array(r['global_permutation']['per_level_code_disagreement'])*100,'s--',label='Global permutation')
        ax.set(title=task.upper(),xlabel='Hierarchy level (1-based)',ylabel='Native codes changed (%)',ylim=(0,105));ax.legend(fontsize=8)
    fig.suptitle('32 paired long contexts per task; permutations are synthetic diagnostic inputs');save(fig,'02_text_permutations')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,style in [('fixed','o-'),('zero','s--')]:
        axes[0].plot(range(1,13),[screen[f'{name}_{k}']['changed_grids']/96*100 for k in range(1,13)],style,label=name)
        axes[1].plot(range(1,13),[screen[f'{name}_{k}']['delta_success']['mean']*100 for k in range(1,13)],style,label=name)
    for ax in axes:ax.set(xlabel='Intervened hierarchy level (1-based)',xticks=range(1,13));ax.legend()
    axes[0].set(ylabel='Final grids changed (%)');axes[1].set(ylabel='Solved-rate difference (percentage points)');axes[1].axhline(0,color='gray',lw=.8)
    fig.suptitle('Sudoku hard: screening on 96 validation puzzles; fixed template vs zero contribution');save(fig,'03_all_levels_screen')
    fig,axes=plt.subplots(1,2,figsize=(11,4));names=list(test['test'])
    for ax,corpus in zip(axes,['test','external']):
        r=[test[corpus][n]['delta_success'] for n in names];m=np.array([v['mean'] for v in r])*100;ci=np.array([v['ci95'] for v in r])*100
        ax.errorbar(range(len(names)),m,yerr=np.stack([m-ci[:,0],ci[:,1]-m]),fmt='o',capsize=4)
        ax.set(xticks=range(len(names)),xticklabels=names,title=corpus+' corpus',ylabel='Solved-rate difference (percentage points)');ax.tick_params(axis='x',labelrotation=25);ax.axhline(0,color='gray',lw=.8)
    fig.suptitle('Validation-selected bypasses: test n=600, external n=256; paired 95% bootstrap intervals');save(fig,'04_bypass_test')

def main():
    text={task:read(task+'_audit.json') for task in ['lm1b','tinystories']};data={task:read(task+'_data.json') for task in text};arch={task:read(task+'_architecture.json') for task in text}
    audit=read('all_levels_audit.json');screen=read('sudoku_screen.json');test=read('sudoku_test.json');selection=read('sudoku_selection_frozen.json')
    old=json.loads((ROOT/'directions_mecanistes/results/pooling_algebra.json').read_text());old_diag=json.loads((ROOT/'directions_mecanistes/results/level5_diagnostic.json').read_text())
    figures(text,audit,screen,test);parts=[]
    def add(s):parts.append(s.strip()+'\n')
    jt=test['test']['joint_three'];je=test['external']['joint_three'];two=test['test']['joint_two']
    title='Architectural Invariance and Bypassable Levels in Next-Concept Models: A Cross-Domain Audit'
    add(rf'''---
title: "{title}"
date: "24 October 2026"
geometry: margin=0.85in
fontsize: 10pt
---

# Abstract

An empirically constant quantization level in a structured task does not by itself establish an architectural failure across domains. We audit published Next-Concept Model (NCM) checkpoints, connecting a Sudoku representation constraint to controlled tests of two text tokenizers. In an additive encoder, fixed token histograms within pooling windows yield identical pooled embeddings before normalization and residual subtraction. The Sudoku serialization aligns the nine-position scale with complete rows and their fixed digit histograms. This explains a structural source of low variation, but does not mathematically force residual codebook collapse. On LM1B and TinyStories, we audit all 12 levels using 256 randomly selected validation contexts and a second stratum of 128 long contexts per tokenizer. Neither exhibits an empirically constant level in either stratum. Block permutations preserve local raw embedding averages while changing native tokenizer codes, separating an isolated pooling invariance from end-to-end representational collapse. A validation-only screen of Sudoku bypasses selects levels {', '.join(map(str,selection['levels']))}. Retaining their modal code contributions reduces predicted levels from 12 to 9 and solves {jt['metrics']['solved']}/600 test puzzles versus 516/600 natively: {100*jt['delta_success']['mean']:.2f} percentage points, paired 95% CI {interval(jt['delta_success']['ci95'])}. These results support a data/architecture interaction and checkpoint-specific bypass opportunities; they do not establish a universal NCM defect, exact equivalence of the nine-pass variant, or a measured speedup.

# Introduction and scope

NCM uses a multiscale residual vector-quantized tokenizer and a generator that predicts one code level at a time [1]. Our starting observation is narrow: level 5 of the published hard-Sudoku tokenizer selects one code on the official solutions examined in our earlier campaign. We ask whether this is a Sudoku-specific interaction, whether comparable constant levels appear in text, and whether further generator predictions can be omitted without retraining.

Three distinct claims require different evidence. A pooling operator can lose order information by construction. A trained quantizer can select a constant code sequence on a particular corpus. A generator prediction can sometimes be replaced by a constant template with little or no change to final answers. None of these statements alone implies either of the others. We keep their measurements and interventions separate.

The contribution is a checkpoint audit, not a new training algorithm or a claim that all hierarchical representations fail. The average-pooling identity is elementary; the mechanistic contribution is its alignment with the exact Sudoku serialization, the residual and dropout controls, and the empirical limits on cross-domain generalization. Previous campaigns and their sources are preserved alongside this extension.

# A conditional invariance result

Consider an additive encoder before contextual processing and normalization,
$E(x)_i=e(x_i)+p_i$, where $p_i$ is fixed by position. Let $B_j$ be a pooling window, and let $n_{{jv}}(x)$ count token $v$ inside that window. Then

$$
P_jE(x)=\frac{{1}}{{|B_j|}}\sum_v n_{{jv}}(x)e(v)+\frac{{1}}{{|B_j|}}\sum_{{i\in B_j}}p_i.
$$

**Proposition.** If two sequences have the same token histogram in every pooling window, their pooled additive embeddings are identical in real arithmetic.

**Proof.** Every token-embedding sum depends only on the corresponding counts $n_{{jv}}$. Every positional-embedding sum is fixed. Both terms are therefore equal for each window. Finite-precision implementations can introduce rounding differences.

This result concerns the specified representation and operator. With positional embeddings, GroupNorm before pooling can reintroduce differences through its statistics. Learned, position-sensitive downsampling can break the averaging invariance. Most importantly, a residual quantizer receives $E_{{norm}}(x)-\hat z_{{<k}}(x)$; the previous contributions can vary even when a raw pooled embedding does not. The proposition therefore does not force a single residual code, imply a reconstruction failure, or identify the history of codebook optimization.

## Sudoku specialization and prior controls

The Sudoku sequence has 89 tokens: nine rows of nine digits and eight separators. The nine adaptive pooling windows use $\lfloor j89/9\rfloor:\lceil(j+1)89/9\rceil$. Each contains one complete row and one or two separators. Every valid row contains each digit once. Thus every window has an invariant token histogram across solved boards.
''')
    r=old['checkpoint_variances']['hard']
    add(f'''On 256 hard training solutions, raw pooled-embedding variance is {r['9']['raw_embedding_inter_puzzle_variance']:.3g} at nine positions, compared with {r['8']['raw_embedding_inter_puzzle_variance']:.3g} and {r['10']['raw_embedding_inter_puzzle_variance']:.3g} at eight and ten positions. The residual entering level 5 remains variable: its inter-puzzle variance is {old_diag['hard']['levels'][4]['input']['inter_puzzle_variance']:.4g}. Our earlier train-only 16-center k-means probe reduces held-out quantization MSE by {pc(old_diag['hard']['quantizer_refit']['relative_mse_reduction'])}; this is not a demonstrated improvement to Sudoku solving.

Previous frozen-weight controls also activate additional level-5 codes with encoder dropout, without EMA updates. Suppressing the residual contribution can change final grids, while forcing its native constant code and retaining that contribution reproduces answers exactly on 856 tested puzzles. These observations rule out the shortcut explanation that a constant level must be a zero or globally useless component. Full results are in the preserved mechanistic report.

# Experimental protocol

## Published assets and text data

We use the published LM1B tokenizer, its small-generator configuration, and an independently trained TinyStories tokenizer. LM1B small, medium, and large share one tokenizer, so they are not three independent tokenizer replications. Generator text weights are not loaded and text generation quality is not evaluated. Exact Hugging Face commits, URLs, sizes and SHA-256 hashes are recorded. Existing local TinyStories weights are copied into the study and checked against the official LFS digest.

For each task, the official tokenized validation rows provide 128-token contexts by truncating longer rows and right-padding shorter ones. LM1B row boundaries come from the published offsets; TinyStories rows come from the published plain-data pickle. We draw 256 rows without replacement and separately draw up to 128 with at least 96 non-PAD tokens, using seed 84117. The second stratum can overlap the first and is descriptive, not an independent replication. Reconstruction includes all non-PAD tokens, including other special tokens. Controls use the first 32 selected long contexts. Permuted sequences are diagnostic inputs, not evidence of natural-language quality.
''')
    for task in text:
        d=data[task];a=arch[task]
        add(f"**{task.upper()}.** Validation rows: {d['n_source_rows']:,}; random sample padding {pc(d['random_pad_fraction'])}; long-stratum padding {pc(d['long_pad_fraction'])}; overlap {d['stratum_overlap']} rows. The encoder has {a['encoder_blocks']} contextual blocks and {a['encoder_conv_blocks']} convolutional blocks; positional embeddings: {a['enc_positional']}; learned first-level downsampling: {a['learned_l1_sampling']}, cascaded: {a['learned_l1_cascade']}.")
    add('''Both corpora mostly provide full contexts, so the long stratum offers little length contrast; it is not a strong padding stress test. No selected random context is duplicated within either task's sample.

## Measurements and fidelity

For each level we measure empirical code counts, marginal entropy, entropy conditioned on position, perplexity, unique code sequences, and fixed-position inter-context variance before and after native downsampling. A level is empirically constant if its complete code sequence is identical across all sampled contexts; unused entries in a large codebook do not meet this definition. We also retain residual MSE before/after each contribution and nearest-code MSE.

Explicit residual accumulation is compared with the untouched official encode path on eight examples. On a small first-batch sample, direct FP64 differences to codebook vectors check nearest assignments independently of the expanded-distance formula. We hash model tensors before and after the audit. All inference is CPU FP32 with one compute thread; the LM1B run uses batch 1 after a documented memory amendment, while TinyStories uses batch 4. These controls establish the tested computations, not universal numerical stability.

## Sudoku screen and statistical units

For all twelve hard-Sudoku levels, templates are positional modes of generated codes from the 1,000 training puzzles. On the first 96 validation puzzles, we separately fix each level to its template while retaining its contribution and regenerate subsequent levels. We also suppress each residual contribution while retaining its prediction. Fixed-template candidates are ranked by changed success status, then changed final grids, then level index. The best three are frozen before the new test interventions.

The selected levels are then fixed separately, in a pair, and jointly on 600 test and 256 external puzzles. Every level and position retains its original identity; fewer Transformer predictions are executed. Paired percentile intervals use 2,000 puzzle-level resamples, seed 84117. Wilson intervals describe discordance rates. Bootstrap intervals are conditional on the templates and validation-selected intervention; zero observed discordances yield a degenerate bootstrap and do not prove population equivalence. No equivalence margin or speed benchmark is specified. Earlier studies already inspected the Sudoku test and external data, so this extension remains exploratory. Cross-difficulty families are shared and are not independent dataset replications.

# Cross-domain tokenizer results

![Empirical code perplexity across hierarchy levels. Text has 256 random and 128 long contexts per tokenizer. Codebook sizes and sampling ceilings differ between tasks; these curves are not comparable utilization percentages.](figures/01_cross_domain_diversity.pdf)
''')
    for task,d in text.items():
        for s in ['random','long']:
            lev=d[s]['levels'];p=[v['occupancy']['perplexity'] for v in lev];u=[v['occupancy']['used'] for v in lev];h=[v['occupancy']['positional_entropy'] for v in lev];dead=[v['level'] for v in lev if v['occupancy']['empirical_constant_level']]
            add(f"**{task.upper()}, {s} (n={d[s]['n']}).** Constant levels: {dead if dead else 'none'}. Across levels, marginal perplexity ranges from {min(p):.2f} to {max(p):.2f}; observed distinct codes from {min(u)} to {max(u)}; minimum position-conditioned entropy {min(h):.4f} nats. Non-PAD reconstruction accuracy is {pc(d[s]['reconstruction_nonpad_accuracy'])}, over {d[s]['n_nonpad_tokens']:,} tokens. These are frozen-tokenizer reconstruction results, not text-generation scores.")
    add('''The Sudoku constant-level observation is therefore not reproduced in these two text tokenizers. This is a counterexample to inevitability, not evidence that all text checkpoints are free of underutilization, redundancy, dimensional collapse or generation defects. A small sample cannot establish that an unobserved code is permanently dead, and varied codes need not carry unique causally useful information.

## Isolated pooling invariance versus native codes

Within each four-token block, we randomly permute tokens. At 32 positions, raw additive embedding averages should remain the same under the proposition. We compare this with the actual first learned downsampler and with every native code level. A global token permutation is a control that also changes local histograms.
''')
    for task in text:
        c=read(task+'_controls.json')['within_four']
        add(f"**{task.upper()}.** Relative L2 difference of raw 32-position averages: {c['raw_pool32_relative_l2']:.3g}; after pre-quant normalization: {c['norm_pool32_relative_l2']:.3g}; actual first-level quantizer input: {c['first_input_relative_l2']:.3g}. First-level native code disagreement: {pc(c['per_level_code_disagreement'][0])}. Across all levels, disagreement ranges from {pc(min(c['per_level_code_disagreement']))} to {pc(max(c['per_level_code_disagreement']))}.")
    add('''![Code disagreement after synthetic permutations on 32 paired long contexts per text tokenizer. Raw pooling can be invariant while the learned tokenizer remains order-sensitive.](figures/02_text_permutations.pdf)

The observed invariance of raw local averages does not survive as a constant full code representation. Learned first-level sampling and the resulting residual history supply a concrete route for order information to reach later levels. We do not identify their separate causal shares by retraining or ablating all alternatives.

# Auditing and bypassing other Sudoku levels

Only level 5 is empirically constant on hard training codes. Level 2 is highly concentrated but variable. Other levels can have diverse codes while their individual fixed-template interventions are partly compensated downstream. Code diversity and practical bypassability must therefore be measured separately.

![All-level validation screen. Fixed preserves each template's residual contribution; zero removes that contribution. The two interventions answer different questions.](figures/03_all_levels_screen.pdf)
''')
    for k in [2,4,5,6,11]:
        a=audit['hard']['generated_train'][k-1];f=screen[f'fixed_{k}'];z=screen[f'zero_{k}']
        add(f"**Level {k}.** Train modal-position agreement {pc(a['modal_position_accuracy'])}, conditional entropy {a['positional_entropy']:.4f} nats; on validation, fixing changes {f['changed_grids']}/96 final grids and {f['status_changes']} success statuses; suppression changes {z['changed_grids']}/96 grids and {z['status_changes']} statuses.")
    add(f"The frozen ranking selects levels {', '.join(map(str,selection['levels']))}. Selection is based on validation behavior rather than an assumption that the smallest entropy levels are always the most removable.")
    for corpus,rs in test.items():
        native=516 if corpus=='test' else 219;n=600 if corpus=='test' else 256
        add(f"## {corpus.capitalize()} evaluation (native {native}/{n})")
        for name,r in rs.items():
            add(f"**{name}**, fixed levels {r['fixed_levels']}, {r['generator_calls']} predictions: {r['metrics']['solved']}/{n} solved, {r['changed_grids']} final grids changed, {r['introduced']} errors introduced and {r['rescued']} rescued. Paired solved-rate difference {100*r['delta_success']['mean']:.2f} percentage points, 95% CI {interval(r['delta_success']['ci95'])}; upper Wilson bound on status discordance {pc(r['status_change_wilson95'][1])}.")
    add('''![Paired performance changes for validation-selected bypasses. Confidence intervals crossing zero do not establish equivalence.](figures/04_bypass_test.pdf)

Replacing level 5 alone is exactly faithful on the evaluated puzzles. The ten- and nine-prediction variants are practical tradeoffs rather than exact simplifications of the native function. The joint result need not equal the sum of individual effects. We make no runtime claim: levels have different lengths, and prefix encoding, cache management and decoding contribute to cost.

# Related work and interpretation

Vector-quantized latent representations originate in VQ-VAE [2]. Next-scale generation has also been studied in visual autoregressive models [3]. Representation collapse in vector quantization is an established research topic [4]; this note does not introduce codebook collapse as a new phenomenon. The NCM paper itself analyzes tokenizer reconstruction and utilization, perturbs coarse text codes, and discusses fixed hierarchy costs [1, Appendices G-I]. Our distinct focus is the exact structured-data/pooling alignment and the separation of occupancy, invariance and bypass effects on released checkpoints.

The mathematical risk is broader than Sudoku: additive representations followed by averaging cannot distinguish arbitrary within-window reorderings with equal histograms. Its harmful realization is conditional on data regularities, the encoder, normalization, learned samplers, residual paths and training. Sudoku guarantees equal row histograms for every valid solution; general text does not impose the same constraint across contexts. The learned text architectures examined here also differ at the first level. The domain comparison therefore changes both data and architecture and does not isolate a single causal factor between them.

# Limitations and falsifiable next steps

The evidence covers three Sudoku tokenizers and two text tokenizers, one released training run per tokenizer and one local backend. Our interventions are concentrated on the hard-Sudoku generator. Neither the absence of constant text levels nor accurate reconstruction evaluates text generator quality. We have not established a universal defect, nor demonstrated that the remaining text codes are semantically disentangled or necessary. Occupancy and entropy estimates depend on sample size, position and codebook size. The two text strata can share rows and mostly contain full contexts; more padding-rich data would be needed to study length dependence.

We lack training trajectories and matched architecture-by-data retraining. We cannot establish why optimization selected the Sudoku codebook configuration or isolate the causal effects of learned first-level sampling. Earlier dropout and k-means controls are post-hoc. The nine-prediction intervention was selected on validation, but this whole research program was motivated by previously inspected test behavior. Confidence intervals do not integrate variability over training seeds, template estimation or repeated intervention searches. Multiple exploratory comparisons are not multiplicity-corrected.

A stronger causal follow-up would retrain matched additive and contextual encoders with average and learned pooling on both structured token histograms and text, using multiple seeds and predeclared quality margins. A practical follow-up would evaluate adaptive bypasses on genuinely untouched puzzles and measure wall-clock savings with randomized, interleaved timing. Testing further published text tokenizers would broaden empirical scope without turning a finite audit into a universal architectural theorem.

# Reproducibility and evidence

The release includes the frozen protocol, amendments, asset commits and checksums, exact selected text tokens, per-level code arrays and histograms, validation screening predictions, frozen Sudoku selection, held-out intervention predictions, PNG/SVG/PDF figures, CSV summaries, scripts and copied model sources. Previous campaigns remain unchanged and supply the original Sudoku proof controls. The source archive excludes large weights and corpora, whose pinned URLs and local locations are recorded.

Run `./run.sh` from the study folder to reproduce the extension sequentially, or `../.venv/bin/python scripts/verify.py` with the cache variables in that runner to verify existing evidence. The verifier checks manifests, recomputes metrics and bootstrap intervals from saved predictions, validates selection, replays native text encoding and verifies the PDF. Numerical identity is not guaranteed across BLAS, PyTorch or hardware versions. The memory amendment and interrupted attempt are retained in the journal.

# References

1. Kumbong et al. (2026). *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*. [arXiv:2610.14162v1](https://arxiv.org/html/2610.14162v1). [Official model assets](https://huggingface.co/hazyresearch/ncm-checkpoints), [official tokenized data](https://huggingface.co/datasets/hazyresearch/ncm-datasets), [source repository](https://github.com/HazyResearch/ncm).
2. van den Oord, Vinyals and Kavukcuoglu (2017). *Neural Discrete Representation Learning*. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937).
3. Tian et al. (2024). *Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction*. [arXiv:2404.02905](https://arxiv.org/abs/2404.02905).
4. *Representation Collapsing Problems in Vector Quantization* (2024). [arXiv:2411.16550](https://arxiv.org/abs/2411.16550).

# Appendix: numerical and interpretive checks

The complete per-level statistics are in `results/text_levels.csv` and `results/sudoku_levels.csv`; exact histograms and code sequences are retained in JSON/NPZ. The first learned downsampler changes scale as well as resolution. In particular, TinyStories exhibits large first-input magnitudes; its input/residual variance ratio must not be interpreted as attenuation by an ordinary averaging operator. Direct nearest-assignment checks and native-encode parity pass on the examined samples, but these checks are not a broad numerical stability study. Raw first-level scales are retained in the JSON evidence rather than silently normalized away.

The earlier level-9 revision study found that a strong observable baseline predicts replacement values and that adding the examined hidden summaries gives no established improvement. That result concerns predictive monitoring and interventions on a different question. It does not establish the architectural invariance result or strengthen the universality of the present tokenizer comparison.
''')
    (D/'PAPER.md').write_text('\n'.join(parts))
    french=f'''# Généralisation architecturale NCM : résultats et portée

La campagne est terminée. L'[article complet en anglais](PAPER.md), sa [version PDF](PAPER.pdf) et sa source LaTeX `PAPER.tex` présentent méthode, preuve conditionnelle, résultats et limites. Le [protocole](OBJECTIFS_ET_PROTOCOLE.md), l'[amendement mémoire](AMENDEMENT_MEMOIRE.md) et le [journal](JOURNAL.md) rendent les décisions inspectables.

## Réponse à la question principale

**Les résultats ne démontrent pas un défaut général du NCM.** La propriété mathématique concerne l'agrégation de représentations additives par moyennes. Sudoku impose les histogrammes fixes qui la rendent particulièrement saillante à neuf positions. Le tokenizer complet utilise aussi normalisation, sous-échantillonnage appris et résidus : l'invariance d'une moyenne ne force donc pas celle des codes natifs.

**Les deux cas texte ne reproduisent pas le niveau constant.** Chaque tokenizer est testé sur 256 contextes validation tirés au hasard, puis 128 contextes longs. Les deux strates peuvent se recouvrir et ne représentent pas deux réplications indépendantes. LM1B small/medium/large partagent le même tokenizer. Aucun générateur texte n'a été évalué.
'''
    for task,d in text.items():
        p=[v['occupancy']['perplexity'] for v in d['random']['levels']]
        french+=f"\n- **{task.upper()}** : aucun niveau constant dans les deux strates ; perplexité marginale des niveaux entre {min(p):.2f} et {max(p):.2f} sur l'échantillon aléatoire ; reconstruction hors padding {pc(d['random']['reconstruction_nonpad_accuracy'])}.\n"
    french+=f'''
Les permutations dans des blocs de quatre tokens préservent les moyennes brutes à 32 positions, mais modifient les codes du tokenizer natif. Il existe donc bien une invariance locale de l'opérateur, sans effondrement correspondant de toute la représentation texte. Cela ne prouve pas que tous les niveaux texte sont utiles, ni que tous les autres checkpoints sont exempts de problèmes.

## Peut-on faire neuf prédictions au lieu de douze sur Sudoku ?

Oui, une variante calculable a été testée, avec une perte observée. Les templates viennent des 1 000 puzzles train. Les niveaux **{selection['levels']}** sont sélectionnés sur 96 puzzles validation, puis gelés avant les nouvelles interventions test. Leur contribution résiduelle est conservée ; leur prédiction Transformer est évitée.

- **Natif** : 516/600 succès sur test, 219/256 sur externe.
- **Dix prédictions**, niveaux {test['test']['joint_two']['fixed_levels']} fixés : {two['metrics']['solved']}/600 ; {two['introduced']} erreurs introduites, {two['rescued']} corrigées.
- **Neuf prédictions**, niveaux {jt['fixed_levels']} fixés : {jt['metrics']['solved']}/600 ; variation {100*jt['delta_success']['mean']:.2f} point(s), IC 95 % {interval(jt['delta_success']['ci95'])} points ; {jt['introduced']} erreurs introduites, {jt['rescued']} corrigées ; {jt['changed_grids']} grilles modifiées.
- **Neuf prédictions sur externe** : {je['metrics']['solved']}/256 ; variation {100*je['delta_success']['mean']:.2f} point(s), IC {interval(je['delta_success']['ci95'])} ; {je['introduced']} erreurs introduites, {je['rescued']} corrigées.

L'IC recouvrant zéro ne prouve pas l'équivalence. Le seul raccourci exactement fidèle sur ces corpus est le niveau 5 seul. Neuf prédictions ne signifient pas 25 % de temps gagné : aucun benchmark de vitesse n'est revendiqué. Les niveaux 4 et 6 ont été audités, mais n'ont pas été retenus par le critère de fidélité sur validation.

## Ce que l'article peut défendre

Une interaction précise entre données structurées et architecture est identifiée ; son analogue élémentaire de pooling existe sur le texte, mais pas le même niveau constant dans les deux cas testés. Des niveaux très diversifiés peuvent aussi être partiellement compensés après remplacement. La bonne distinction est entre **invariance de l'opérateur**, **faible diversité empirique** et **coût causal d'une prédiction**.

L'article ne reconstitue pas l'entraînement et n'établit pas une pathologie universelle. Les tests Sudoku restent exploratoires puisque ces corpus ont été consultés dans les campagnes précédentes. Une étude causale de généralité demanderait un plan croisant données et architectures avec plusieurs entraînements contrôlés.

## Preuves et reproduction

Les quatre figures sont exportées en PNG/SVG/PDF. Les résultats par contexte et puzzle, histogrammes, matrices de codes et prédictions sont dans `results/`. Les poids et corpus épinglés sont copiés dans `artifacts/` et `data/`. L'archive `sources_generalisation.zip` contient tous les scripts utilisés, les sources des modèles et les configurations ; elle exclut les gros poids. `results/verification.json` donne les contrôles effectivement passés et `results/final_manifest.json` les empreintes finales.

Depuis ce dossier : `./run.sh` pour tout reproduire ; `PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" ../.venv/bin/python scripts/verify.py` pour vérifier les artefacts. Les campagnes précédentes et leurs manifestes restent inchangés.
'''
    (D/'RAPPORT.md').write_text(french)
    with (D/'results/text_levels.csv').open('w') as out:
        writer=csv.writer(out);writer.writerow(['task','stratum','level','n','codes_used','perplexity','positional_entropy','modal_position_fraction','input_variance','residual_variance'])
        for task,d in text.items():
            for st in ['random','long']:
                for r in d[st]['levels']:
                    o=r['occupancy'];writer.writerow([task,st,r['level'],d[st]['n'],o['used'],o['perplexity'],o['positional_entropy'],o['modal_position_fraction'],r['geometry']['input']['inter_sequence_variance'],r['geometry']['residual']['inter_sequence_variance']])
    with (D/'results/sudoku_levels.csv').open('w') as out:
        writer=csv.writer(out);writer.writerow(['checkpoint','kind','level','modal_position_accuracy','positional_entropy','codes_used'])
        for task,d in audit.items():
            for kind,rows in d.items():
                for r in rows:writer.writerow([task,kind,r['level'],r['modal_position_accuracy'],r['positional_entropy'],r['codes_used']])
    env=os.environ.copy()
    for var,sub in [('TMPDIR','tmp'),('TEXMFVAR','tex-var'),('TEXMFCONFIG','tex-config')]:
        path=D/'.cache'/sub;path.mkdir(parents=True,exist_ok=True);env[var]=str(path)
    build=D/'.cache/tex-build';build.mkdir(exist_ok=True)
    with (D/'logs/paper_build.log').open('w') as out:
        subprocess.run(['pandoc','PAPER.md','--standalone','--number-sections','-V','colorlinks=true','-o','PAPER.tex'],cwd=D,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
        for _ in range(2):subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(build),'PAPER.tex'],cwd=D,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
    shutil.copyfile(build/'PAPER.pdf',D/'PAPER.pdf')
    sys.path.insert(0,str(D/'.deps'));pdf_version=importlib.metadata.version('pymupdf')
    (D/'sources/pdf-requirements.txt').write_text('pymupdf=='+pdf_version+'\n')
    write(D/'sources/environment.json',dict(python=platform.python_version(),platform=platform.platform(),packages={n:importlib.metadata.version(n) for n in ['torch','numpy','scikit-learn','matplotlib','omegaconf']},
       pdf_package=dict(pymupdf=pdf_version),document_tools={n:subprocess.check_output([n,'--version'],text=True).splitlines()[0] for n in ['pandoc','pdflatex']}))
    archive_sources();log('publication_complete',title=title,words=len((D/'PAPER.md').read_text().split()))

def archive_sources():
    with zipfile.ZipFile(D/'sources_generalisation.zip','w',zipfile.ZIP_DEFLATED) as z:
        files=list((D/'scripts').glob('*.py'))+list(D.glob('*.md'))+[D/'PAPER.tex',D/'protocol.json',D/'run.sh']
        files += [p for p in (D/'sources').rglob('*') if p.is_file() and '__pycache__' not in str(p)]
        for p in files:z.write(p,str(p.relative_to(ROOT)))
        for name in ['scripts/common.py','protocol.json','results/splits.json','requirements.txt','directions_mecanistes/scripts/diagnose_level5.py','directions_mecanistes/scripts/pooling_algebra.py']:
            z.write(ROOT/name,name)
        for p in (ROOT/'vendor/ncm_mac').rglob('*.py'):z.write(p,str(p.relative_to(ROOT)))
        for p in (ROOT/'artifacts/sudoku').glob('*/config.yaml'):z.write(p,str(p.relative_to(ROOT)))
        for p in (D/'artifacts').rglob('*.yaml'):z.write(p,str(p.relative_to(ROOT)))

if __name__=='__main__':main()

"""Narrative is generated from the frozen tests, including an unsupported hypothesis."""
from base import *
def pc(x):return f'{100*x:.2f}%' if x is not None else 'non estimable (denominator zero)'
def ci(x):return '['+', '.join(f'{v:.5f}' for v in x)+']' if x is not None else 'undefined: denominator zero'
def documents(c,t,o):
    f=c['fresh'];R=f['contrasts']['R'];I=f['contrasts']['I'];v=t['all_positions']['transitions'][7]['adoption_then_abandonment'];vc=t['final_content_positions']['transitions'][7]['adoption_then_abandonment'];ov=o['reference_nonpad']['transitions'][7]['adoption_then_abandonment'];positive=f['co_primary_positive']
    title='Apparent Answer Revision in NCM: Codec Dynamics and '+('Conditional Residual Compensation' if positive else 'a Test of Conditional Residual Compensation')
    verdict='Le critère co-primaire annoncé est satisfait pour cette intervention.' if positive else "Le critère co-primaire annoncé n'est pas satisfait : une compensation utile et spécifique à l'omission n'est pas établie."
    rows=[];english=[]
    for name,q in c.items():
        rows.append(f"### {name}, n={q['Y00']['metrics']['n']}")
        for k in ['Y00','Y10','Y11','Y01','random_tail_delta']:rows.append(f"- {k} : {q[k]['metrics']['solved']} grilles exactes ; NLL cases vides {q[k]['mean_nll_blank']:.6f} ; {q[k]['changed_grids_vs_native']} grilles modifiées vs natif.")
        for k in ['R','I']:rows.append(f"- {k} : {q['contrasts'][k]['mean']:.6f} nat ; IC 95 % {ci(q['contrasts'][k]['ci95'])} ; IC 97,5 % {ci(q['contrasts'][k]['ci975'])}.")
        arr=np.load(D/f'results/{name}_crossed.npz');mse=' / '.join(format(float(arr[k+'_latent_mse'].mean()),'.6f') for k in ['Y00','Y10','Y11','Y01'])
        rows.append(f"- Codes modifiés entre suites : {int((arr['native_codes']!=arr['adapted_codes']).sum())} ; MSE latente moyenne Y00/Y10/Y11/Y01 : {mse}.")
        was=np.all(arr['Y10_ids']==arr['reference'],1);now=np.all(arr['Y11_ids']==arr['reference'],1);d=q['contrasts']['delta_success_Y11_Y10']
        rows.append(f"- Y10 vers Y11 : {int((~was&now).sum())} grilles sauvées, {int((was&~now).sum())} échecs introduits ; gain de réussite {100*d['mean']:.2f} points, IC 95 % en proportions {ci(d['ci95'])}.")
        english.append(f"- {name}, n={q['Y00']['metrics']['n']}: Y00/Y10/Y11/Y01 solve {'/'.join(str(q[k]['metrics']['solved']) for k in ['Y00','Y10','Y11','Y01'])} grids. R={q['contrasts']['R']['mean']:.6f} nats, 97.5% CI {ci(q['contrasts']['R']['ci975'])}; I={q['contrasts']['I']['mean']:.6f}, CI {ci(q['contrasts']['I']['ci975'])}.")
    data=np.load(D/'results/lm1b_generation.npz');ids=data['decoded'];anchor=ids[:,-1];mask=(anchor>103)&(anchor<30522);changed=(ids[:,7]!=anchor)&(ids[:,8]==anchor)&((ids[:,9]!=anchor)|(ids[:,10]!=anchor))&mask;vocab=(D/'sources/bert_vocab.txt').read_text().splitlines() if (D/'sources/bert_vocab.txt').exists() else [];ex=[]
    for i,j in np.argwhere(changed)[:12]:
        vals=ids[i,7:,j];ex.append(dict(trajectory=int(i),seed=int(data['seeds'][i]),position=int(j),token_ids8_to12=vals,wordpieces8_to12=[vocab[int(k)] if int(k)<len(vocab) else str(k) for k in vals]))
    write(D/'results/text_counterexamples.json',ex);example=' → '.join(ex[0]['wordpieces8_to12']) if ex else 'aucun contre-exemple non spécial à cette transition'
    report=f'''# Codec, adaptation et test de compensation conditionnelle

Campagne expérimentale du 24 octobre 2026, CPU FP32, checkpoints gelés. [Article anglais](PAPER.md), [PDF](PAPER.pdf), [protocole gelé](OBJECTIFS_ET_PROTOCOLE.md). Les six campagnes précédentes restent intactes ; le patching et les puzzles à indice disambiguïsant sont hors périmètre.

## Résultats principaux

**Plan croisé sur 256 nouveaux puzzles : R={R['mean']:.6f} nat, IC 97,5 % {ci(R['ci975'])} ; I={I['mean']:.6f} nat, IC 97,5 % {ci(I['ci975'])}.** {verdict} Le récit « le générateur tente de compenser » ne peut pas être déduit du seul changement de ses codes.

**Le générateur LM1B small est exécuté, pas seulement son tokenizer.** Sur 128 trajectoires natives, {v['numerator']}/{v['denominator']} positions adoptant à 9 leur token raw final l'abandonnent à 10 ou 11 : **{pc(v['rate'])}**, IC 95 % {ci(v['ci95'])} en proportions. Sur le masque secondaire des tokens finaux non spéciaux : {vc['numerator']}/{vc['denominator']} = {pc(vc['rate'])}. L'ancre finale est rétrospective, pas une réponse correcte.

**Codec LM1B oracle :** {ov['numerator']}/{ov['denominator']} corrections à 9 sont ensuite dégradées, soit {pc(ov['rate'])}, sur 128 phrases validation réutilisées, padding exclu. Ce contrôle possède de vraies cibles de reconstruction et doit rester distinct des textes générés.

## 1. Ce que le récit peut affirmer

Le contrôle oracle Sudoku antérieur démontre que le codec suffit à produire une reconstruction non monotone, même à partir de la bonne solution. Il ne démontre pas que tous les revirements du générateur proviennent exclusivement du codec. L'attention du décodeur et celle du générateur ne sont pas la même composante. La grille intermédiaire n'est pas réinjectée dans le générateur pour lui montrer ses erreurs.

Le titre retenu est **{title}**. La mention « a Test of » est conservée si la compensation n'est pas établie, afin de distinguer la question expérimentale d'une conclusion positive. L'observation sur Sudoku et LM1B étend la portée du phénomène au-delà du Sudoku ; elle ne prouve ni universalité, ni défaut général de NCM, ni correction sémantique en texte.

## 2. Protocole, données et incertitude

Omission de la contribution 9, conditions, métriques et critère gelés avant calcul. Les 600 hard test et les 256 externes antérieurs sont des réplications exploratoires déjà examinées. Le test principal porte sur 256 nouvelles solutions générées après le gel par backtracking aléatoire, puis retrait glouton jusqu'à 30 indices. Unicité vérifiée indépendamment par MRV et Algorithm X. Les clés D4/normalisation des chiffres sont exclues contre les solutions officielles, externes antérieures et nouvelles déjà retenues ; toutes les isomorphies et une contamination d'entraînement ne sont pas exclues.

L'effectif est un budget de précision fixé, sans garantie de puissance ; le générateur de Sudoku n'est pas uniforme. Aucun seuil, niveau, dose ou sous-groupe n'est sélectionné sur les nouvelles réponses. Les jeux restent séparés, sans regroupement selon le signe des résultats.

La perte L est la NLL des chiffres de référence sur les cases initialement vides, moyennée par puzzle. R=L10-L11 mesure le bénéfice de régénérer la suite sous perturbation. I=L10+L01-L11-L00 retranche le bénéfice que la même suite adaptée apporte à l'état natif. Un R positif seul peut refléter une meilleure suite pour les deux états. Ces contrastes ne sont pas une décomposition causale unique ou une mesure d'intention de réparer.

Bootstrap percentile apparié par puzzle, 4 000 tirages ; IC descriptifs à 95 % et IC à 97,5 % pour les deux contrastes co-primaires avec ajustement Bonferroni. Le critère annoncé exige les deux bornes basses à 97,5 % positives sur les nouveaux puzzles. La couverture bootstrap est approximative et conditionnelle à ces poids.

## 3. Les quatre conditions et les contrôles

Après niveau 9, z=z9 et δ=-r9. T0 est la somme native des résidus 10..12 ; Tδ celle de la suite régénérée depuis z8, à puzzle constant.

- Y00 : état natif et suite native.
- Y10 : état sans r9 et contributions natives suivantes figées.
- Y11 : état sans r9 et suite suivante régénérée.
- Y01 : état natif et suite adaptée à l'omission.

Les ajouts suivent l'ordre natif. La branche perturbée part directement de z8 pour éviter un aller-retour de soustraction en FP32 ; les écarts de réassociation des sommes sont sauvegardés. Les croisements sont des interventions artificielles, possiblement hors distribution. Un contrôle gaussien ajoute à la fin de Y10 une direction indépendante de même norme que ΔT=Tδ-T0 ; cela apparie l'amplitude, pas toute la géométrie ou la distribution.

Parité du runner avec le chemin natif sur 32 puzzles : codes, logits et accumulations 8/9 identiques. Les 600 baselines sont comparées aux traces antérieures ; les 128 premières omissions recoupent l'étude temporelle. Les caches et états de branches sont contrôlés. Les logits et les deux suites de codes sont conservés par puzzle.

## 4. Résultats complets du plan croisé

{chr(10).join(rows)}

![NLL croisée](figures/01_crossed_nll.png)

![Contrastes annoncés](figures/02_contrasts.png)

Un R favorable sur l'ancien externe ne suffit pas si la suite adaptée bénéficie aussi à l'état natif : le quatrième terme est nécessaire pour la question de spécificité. La réussite exacte est secondaire et ne remplace pas la NLL annoncée. Aucun IC recouvrant zéro ne prouve une équivalence de performance.

Sur les nouveaux puzzles, cosinus moyen de ΔT avec -δ : {f['geometry']['cosine_to_negative_delta']['mean']:.4f}. Ratio moyen ||δ+ΔT||/||δ|| : {f['geometry']['remaining_delta_ratio']['mean']:.4f}. Une valeur supérieure à 1 ne correspond pas à une annulation euclidienne. La géométrie latente et le bénéfice de reconstruction peuvent diverger ; les pertes restent le test principal. Les MSE vers l'encodage de la solution sont des diagnostics utilisant la référence, pas des signaux de contrôle disponibles au modèle.

![Géométrie latente](figures/03_geometry.png)

La non-confirmation porte sur une omission complète, un checkpoint et ces données. Elle n'exclut pas d'autres formes de compensation à d'autres niveaux ou doses. La tolérance à l'omission et l'adaptation des codes sont compatibles avec les résultats, sans qu'une intention de réparer soit démontrée.

## 5. Générateur LM1B : exécution et observable

Checkpoint LM1B small publié, 97 065 600 paramètres, environ 390 Mo ; empreinte SHA-256 vérifiée contre l'entrée LFS officielle et révision épinglée identique à celle du tokenizer antérieur. Génération native eager, 128 trajectoires B=1 avec graines distinctes ; préfixe CLS=101, premier code échantillonné dans l'unigramme sauvegardé, température 1, top-k=0, top-p=1, CFG=0. Les douze longueurs sont 1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108 et 128 ; le niveau 9 n'est pas une résolution identique au Sudoku.

Chaque accumulation est reconstruite explicitement avant modification, pour éviter l'aliasing des historiques natifs. Le dernier état est égal au natif. Les deux premières graines et leurs douze lectures sont rejouées exactement. Le chargeur compare tous les tenseurs enregistrés au checkpoint et reconstruit les petites têtes partagées par copie ordinaire, afin que leurs vues utilisent les vrais biais chargés.

Incident documenté : le premier chargeur assignait les paramètres sans réactualiser les références des vues de biais partagés. Ses 46 trajectoires sont écartées et conservées sous data/invalid_loader_trajectories ; **aucune n'entre dans les résultats**. Le calcul corrigé reprend les mêmes graines et tous les paramètres de sampling. Un incident disque a motivé le chargement meta/mmap, la reconstruction exacte des buffers RoPE non persistants et le nettoyage de caches régénérables non manifestés. Les poids sources ne sont pas modifiés.

L'argmax brut du décodeur définit les lectures principales. Le post-traitement officiel fréquence/adjacence est reproduit uniquement pour la sortie finale d'inspection ; il diffère du raw sur {pc(t['raw_vs_official_final_disagreement'])} des positions. Ces heuristiques ne sont pas mélangées aux courbes de révision. Aucune perplexité GPT2 ni qualité sémantique n'est évaluée.

En génération inconditionnelle, le token final brut sert seulement d'ancre rétrospective. Une adoption à k signifie différent de l'ancre à k-1, égal à k ; un abandon signifie différent à un niveau ultérieur. Le dernier accord vaut 100 % par construction et les abandons après adoption à 11/12 sont structurellement nuls. Toutes les transitions sont rapportées, sans sélection d'un pic favorable.

![Révisions LM1B](figures/04_text_generation.png)

L'analyse principale utilise les 128 positions fixes. Le masque secondaire retient les identifiants finaux 104..30521 : il exclut tokens spéciaux, padding et lignes inutilisées, mais conditionne sur le futur. Les IC rééchantillonnent les trajectoires, pas les tokens supposés indépendants. Exemple WordPiece aux niveaux 8..12 : **{example}**. Identifiants, graines et positions des exemples sont dans results/text_counterexamples.json.

## 6. Codec texte oracle et portée de la généralisation

128 phrases validation déjà sélectionnées sont encodées puis décodées progressivement ; les références non-PAD permettent ici de parler de corrections et dégradations de reconstruction. L'exactitude finale vaut {pc(o['reference_nonpad']['agreement_by_level'][-1])}. Les lots de quatre sont vérifiés contre des lots unitaires sur les quatre premières phrases. Ces références sont distinctes des générations inconditionnelles et restent exploratoires, car déjà examinées.

![Codec oracle LM1B](figures/05_text_oracle.png)

Une révision du token généré n'est pas une amélioration de sens, et une ancre finale de faible qualité reste une ancre. La présence de non-monotonie sur un générateur texte et sur Sudoku établit des exemples hors Sudoku, pas une loi de toute la famille NCM. Une attribution à l'architecture exigerait des entraînements et architectures contrôlés.

## 7. Conclusion et reproduction

{verdict} Le résultat solide distingue suffisance du codec pour produire des revirements, adaptation effective des codes et compensation utile. Il ne démontre ni origine exclusivement codec de toutes les révisions, ni transition de phase, ni contrôle métacognitif natif. Patching d'attention et puzzles disambiguïsants restent réservés à une autre étude.

Reproduction : `./run.sh`. Les poids antérieurs sont lus dans le parent ; seul le nouveau générateur est téléchargé dans ce dossier. Les NPZ contiennent logits et codes croisés, graines et lectures texte ; results/verification.json annonce le rejeu effectivement passé, results/final_manifest.json fige les artefacts. sources_compensation.zip archive les scripts, modèles, configurations et protocoles, sans gros poids. Les six manifestes antérieurs sont contrôlés. Les incidents et leurs corrections sont conservés dans JOURNAL.md et logs/.
'''
    (D/'RAPPORT.md').write_text(report)
    english_paper(title,verdict,c,t,o,english)

def english_paper(title,verdict,c,t,o,rows):
    f=c['fresh'];R=f['contrasts']['R'];I=f['contrasts']['I'];v=t['all_positions']['transitions'][7]['adoption_then_abandonment'];vc=t['final_content_positions']['transitions'][7]['adoption_then_abandonment'];ov=o['reference_nonpad']['transitions'][7]['adoption_then_abandonment'];conclusion='The joint criterion supports compensation under this intervention.' if f['co_primary_positive'] else 'The prespecified joint criterion for useful, state-specific compensation is not met.'
    paper=f'''---
title: "{title}"
date: "24 October 2026"
geometry: margin=22mm
fontsize: 10pt
header-includes:
  - \\usepackage{{float}}
  - \\floatplacement{{figure}}{{H}}
---

# Abstract

We separate non-monotone progressive-codec readouts from conditional generator compensation in Next-Concept Models. Earlier Sudoku controls reproduce decoded reversions while encoding known correct solutions without generator execution. Here a crossed state/tail intervention removes residual 9, and a separate experiment executes the released LM1B small generator. On 256 newly generated unique Sudoku puzzles, recovery R is {R['mean']:.6f} nats, with 97.5% paired-bootstrap interval {ci(R['ci975'])}; specificity I is {I['mean']:.6f}, interval {ci(I['ci975'])}. {conclusion} In 128 native LM1B trajectories, {v['numerator']}/{v['denominator']} positions adopting their own final raw token at level 9 later abandon it ({pc(v['rate'])}). This retrospective stability metric does not measure semantic correctness. Results extend the existence of non-monotone readouts beyond Sudoku while distinguishing code adaptation from demonstrated compensation.

# Mechanistic question and oracle evidence

The Sudoku generator predicts codes from cumulative latent representations and puzzle clues. It does not consult its decoded intermediate grid. Earlier codes remain fixed, while each new residual can alter any decoded digit. Append-only codes therefore do not imply locked decoded answers.

On the earlier 600 hard test puzzles, a codec encoding known correct solutions produces 3,176 reversions among 12,717 cells corrected at level 9 (24.97%). Free generation produces 3,088 among 12,333 (25.04%). The oracle reconstructs all 600 final grids exactly, but none at levels 1-11. This establishes codec sufficiency to produce the phenomenon, not exclusive attribution of generator reversions: the selected correction sets differ. The codec decoder itself includes attention. Nested dropout in the tokenizer recipe also rules out simply assuming that only complete representations were supervised; configurations do not recover the full historical training exposure.

# Frozen crossed experiment

After native level 9, let z=z9 and delta=-r9. T0 is the sum of native contributions 10-12; Tdelta is the tail regenerated from z8 with the same puzzle. We evaluate Y00 (native state/native tail), Y10 (perturbed state/frozen native tail), Y11 (perturbed state/regenerated tail), and Y01 (native state/tail adapted to the perturbation).

We directly use z8 rather than subtracting and readding residuals in FP32. Contributions retain their native order; reassociation deviations are recorded. An explicit runner matches native codes, final logits and states 8/9 on 32 puzzles. All 600 official baselines match prior codes and outputs; the first 128 omitted branches match the prior temporal study. Cache cleanup and branch isolation are asserted. Parameters remain frozen.

For per-puzzle mean NLL L on initially empty cells, R=L10-L11 measures recovery from recomputing the tail. I=L10+L01-L11-L00 subtracts the benefit the same adapted tail has on the native state. A better Y11 than Y10 alone need not imply state-specific repair. These are explicit controlled contrasts, not a unique causal mediation decomposition or a measure of intent.

Both contrasts are prespecified co-primary. Four thousand paired puzzle-bootstrap draws provide 95% descriptive intervals and 97.5% intervals with a two-contrast Bonferroni adjustment. The declared compensation criterion requires both adjusted lower bounds positive on fresh puzzles. Coverage is approximate and conditional on frozen weights. A Gaussian readout control matches the norm of adapted-minus-native tail change, but not its geometry or distribution.

# Data and independence

The 600 official hard test and 256 previous external puzzles were examined previously and remain exploratory replications. After protocol freezing, we generate 256 new solutions by randomized backtracking, then remove clues greedily while retaining uniqueness until 30 clues remain. MRV and independent Algorithm X verify each puzzle. Solution keys under D4 and digit normalization are excluded against official, prior external and other new solutions. The audit does not cover every isomorphism or unknown training contamination.

The new corpus is a fixed precision budget, without guaranteed power, and its generator is not uniform. No level, dose, subset or decision threshold is changed based on its model results. Populations are kept separate rather than pooled after examining signs.

# Crossed results

{chr(10).join(rows)}

![Crossed negative log likelihood on initially empty cells, averaged by puzzle.](figures/01_crossed_nll.pdf)

![Recovery and specificity with 97.5% paired-bootstrap intervals. Fresh puzzles are the prespecified primary population.](figures/02_contrasts.pdf)

On fresh puzzles the four NLL values Y00/Y10/Y11/Y01 are {f['Y00']['mean_nll_blank']:.6f}/{f['Y10']['mean_nll_blank']:.6f}/{f['Y11']['mean_nll_blank']:.6f}/{f['Y01']['mean_nll_blank']:.6f}. {conclusion} A favorable recovery estimate on the previous external set does not demonstrate specificity when the adapted tail also benefits the native state. Exact-grid success is secondary and does not replace the announced loss observable. Overlap with zero does not establish equivalence.

The mean cosine of tail change with the removed residual is {f['geometry']['cosine_to_negative_delta']['mean']:.4f}; mean norm(delta+tail change)/norm(delta) is {f['geometry']['remaining_delta_ratio']['mean']:.4f}. These assess Euclidean cancellation, not necessarily task-level compensation. Different downstream codes do not alone prove useful repair. We retain branch codes, logits and latent MSE to the encoded reference solution.

![Latent response to omission on fresh puzzles.](figures/03_geometry.pdf)

This tests one complete residual omission, not all possible compensation mechanisms. Crossed states may be out of distribution; a random norm-matched direction is not a fully distribution-matched control. Insensitivity of a frozen tail and adaptation without benefit remain distinct possibilities.

# Native LM1B generation

We execute the published LM1B small generator, with 97,065,600 parameters, strict loading and SHA-256 verification against the official LFS digest at the pinned revision. The already pinned tokenizer is reused. Native eager generation uses 128 independently seeded B=1 trajectories, CLS prefix 101, the checkpoint's saved level-0 unigram, temperature 1, top-k 0, top-p 1 and CFG 0. The level lengths are 1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108 and 128; level 9 is not the same spatial resolution as Sudoku level 9.

We explicitly decode each cumulative representation before mutation, avoiding aliasing of the original history list, and verify the final accumulation against the native method. The first two seeds and all their readouts are replayed exactly. Parameters are memory mapped and deterministic nonpersistent RoPE buffers are rebuilt as in the constructor. Shared-head views are bound to loaded biases using ordinary copying for that component; every registered tensor is checked against the checkpoint. An initial incorrect loader retained stale shared-bias references: its 46 trajectories are excluded and archived separately, and all reported sequences are recomputed. Disk and loading incidents are recorded in the journal, with no changed seeds or sampling parameters.

The main readout is raw decoder argmax at all 128 positions. The official final frequency/adjacency postprocessing is reproduced separately for inspection and disagrees with raw final tokens at {pc(t['raw_vs_official_final_disagreement'])} of positions. We do not mix this heuristic with intermediate curves or evaluate GPT2 perplexity or semantic quality.

Since unconditional generation has no unique target sentence, its own final raw token is a retrospective anchor. Adoption at k means disagreement at k-1 followed by agreement at k; abandonment means later intermediate disagreement. Final agreement is 100% by construction, and abandonment after arrival at levels 11/12 is structurally zero. All transitions are reported. A secondary mask retains final IDs 104-30521, excluding specials and unused vocabulary rows but conditioning on the future. Intervals resample trajectories rather than tokens assumed independent.

![Native text trajectories and abandonment of adopted final tokens.](figures/04_text_generation.pdf)

At arrival 9 the count is {v['numerator']}/{v['denominator']} ({pc(v['rate'])}; 95% trajectory-bootstrap interval in proportions {ci(v['ci95'])}). The final non-special mask gives {vc['numerator']}/{vc['denominator']} ({pc(vc['rate'])}). Counterexamples retain seed, position and token identifiers. This measures non-monotone readout relative to an eventual output, not semantic error correction.

# LM1B oracle codec and scope

As a distinct control, we encode 128 previously selected validation rows and decode their partial accumulations. Non-PAD reference tokens give actual reconstruction targets. The number of corrections at 9 later degraded is {ov['numerator']}/{ov['denominator']} ({pc(ov['rate'])}); final reconstruction accuracy is {pc(o['reference_nonpad']['agreement_by_level'][-1])}. These rows are not paired references for unconditional generation and remain exploratory. Four-row oracle batches are checked against singleton batches on the first four rows.

![Text oracle reconstruction on reused validation rows.](figures/05_text_oracle.pdf)

The existence of these trajectories outside Sudoku is not a universal architectural law or defect. A final generated token can itself be poor and does not become correct by being final. Architecturally causal generalization would require controlled model families and training comparisons.

# Conclusion and reproducibility

{conclusion} The defensible result separates codec sufficiency, effective code adaptation and useful compensation. No phase transition, exclusive codec origin, metacognitive monitoring or intent to repair is inferred. Attention patching and disambiguating-puzzle interventions are deliberately deferred.

Limitations include one Sudoku checkpoint for interventions, one text generator size, one omission, one text temperature, nonuniform puzzles and artificial crossed states. Frozen weights and uncertainty intervals do not remove these limits. Run `./run.sh` to reproduce the campaign. Saved NPZ files contain per-case logits, codes, seeds and all readouts; verification states its exact replay scope. The source archive contains scripts, model sources, configurations and protocol, excluding large weights. Six earlier campaign manifests remain unchanged.

Primary sources: [NCM article](https://arxiv.org/abs/2610.14162), [official repository](https://github.com/HazyResearch/ncm), [published checkpoints](https://huggingface.co/hazyresearch/ncm-checkpoints). Numerical results are local measurements.
'''
    (D/'PAPER.md').write_text(paper)

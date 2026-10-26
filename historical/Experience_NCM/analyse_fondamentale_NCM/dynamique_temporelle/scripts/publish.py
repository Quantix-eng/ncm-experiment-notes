"""Render the temporal study from saved per-puzzle measurements."""
from base import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv,zipfile,subprocess,shutil

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
def read(n):return json.loads((D/'results'/n).read_text())
def pct(x):return f'{100*x:.2f} %'
def interval(x):return '['+', '.join(f'{100*v:.2f}' for v in x)+']'
def save(fig,name):
    fig.tight_layout(rect=(0,0,1,.93))
    for ext in ['png','svg','pdf']:fig.savefig(D/f'figures/{name}.{ext}',bbox_inches='tight',dpi=160)
    plt.close(fig)

def figures(o,a,r,s):
    t=o['hard_test'];x=np.arange(1,13)
    fig,ax=plt.subplots(1,2,figsize=(10,3.2))
    for name,label in [('hard_test','Hard test (600)'),('external','External (256)')]:
        v=o[name]['profiles'];ax[0].plot(x,[100*q['cell_accuracy'] for q in v],'o-',label=label);ax[1].plot(x,[100*q['solved_rate'] for q in v],'o-',label=label)
    for z in ax:z.set(xlabel='Native generation level',xticks=x,ylim=(-2,102));z.axvline(9,color='.65',ls=':',lw=1);z.legend(fontsize=8)
    ax[0].set(ylabel='Correct Sudoku cells (%)');ax[1].set(ylabel='Exact solved grids (%)');fig.suptitle('Stopping is costly at 8, 9, 10 and 11; level 9 does not finish the solution');save(fig,'01_native_stops')
    fig,ax=plt.subplots(1,2,figsize=(10,3.2));tr=t['transitions'];xx=np.arange(2,13)
    co=np.array([q['all_cells']['corrected'] for q in tr]);de=np.array([q['all_cells']['degraded'] for q in tr]);ax[0].bar(xx-.17,co,width=.34,label='Wrong to correct',color='#326c98');ax[0].bar(xx+.17,de,width=.34,label='Correct to wrong',color='#bd7040');ax[0].plot(xx,co-de,'k.-',label='Net corrected');ax[0].set(xlabel='Arrival level',ylabel='Cell events across 600 test puzzles',xticks=xx);ax[0].legend(fontsize=8)
    ex=t['counterexamples'][0];ax[1].plot(np.arange(8,13),ex['values8_to12'],'o-',color='#326c98');ax[1].axhline(ex['reference'],color='#bd7040',ls='--',label='Reference digit');ax[1].set(xlabel='Native generation level',ylabel='Decoded digit',xticks=range(8,13),yticks=range(1,10),title=f"Puzzle {ex['source_index']}, row {ex['row']}, column {ex['column']}");ax[1].legend(fontsize=8)
    fig.suptitle('Correcting at 9 does not lock the decoded value');save(fig,'02_corrections_reversals')
    fig,ax=plt.subplots(1,2,figsize=(10,3.2));g=np.array([float(k) for k in r]);v=list(r.values());ax[0].plot(g,[100*q['cell_accuracy'] for q in v],'o-');ax[0].set(ylabel='Correct cells (%)');ax[1].plot(g,[q['nll_cells'] for q in v],'o-',label='Gold-digit NLL (nats)');ax[1].set(ylabel='Gold-digit negative log probability (nats)')
    for z in ax:z.set(xlabel='Fixed level-9 contribution amplitude gamma')
    fig.suptitle('Readout interpolation: decode z8 + gamma R9; 128 test puzzles, no continuation');save(fig,'03_reader_interpolation')
    fig,ax=plt.subplots(1,2,figsize=(10,3.2));names=['native']+['omit_'+str(k) for k in Q['omit_levels']];values=[100*a[n]['metrics']['solved_rate'] for n in names];ax[0].bar(np.arange(len(names)),values,color=['.4']+['#326c98']*5);ax[0].set(xticks=range(len(names)),xticklabels=['Native']+[f'Omit {k}' for k in Q['omit_levels']],ylabel='Final exact solved grids (%)',ylim=(0,102));ax[0].tick_params(axis='x',labelrotation=25)
    gs=Q['continuation_gamma'];ax[1].plot(gs,[100*a['gamma_'+str(g)]['metrics']['solved_rate'] for g in gs],'o-');ax[1].set(xlabel='Level-9 contribution amplitude gamma',ylabel='Final exact solved grids (%)',ylim=(0,102));fig.suptitle('Residual omission with tail regeneration; 128 paired hard test puzzles');save(fig,'04_temporal_interventions')
    fig,ax=plt.subplots(1,2,figsize=(10,3.2));xx=np.arange(len(s));width=.32
    for i,timing in enumerate(['before','after']):ax[0].bar(xx+(i-.5)*width,[100*v[timing]['change_rate'] for v in s.values()],width,label=f'Perturb {timing} level 9')
    ax[0].set(xticks=xx,xticklabels=[pct(float(k)) for k in s],xlabel='Perturbation norm / norm(z8)',ylabel='Final grids changed vs native (%)');ax[0].legend(fontsize=8)
    for i,v in enumerate(s.values()):
        m=v['after_minus_before_change']['mean']*100;lo,hi=np.array(v['after_minus_before_change']['ci95'])*100;ax[1].errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color='#326c98')
    ax[1].axhline(0,color='.6',lw=1);ax[1].set(xticks=xx,xticklabels=[pct(float(k)) for k in s],xlabel='Perturbation norm / norm(z8)',ylabel='After minus before (percentage points)');fig.suptitle('Matched absolute perturbations: 64 puzzles, 3 seeds; 95% CI by puzzle');save(fig,'05_susceptibility')
    fig,ax=plt.subplots(1,2,figsize=(10,3.2))
    for z,key in zip(ax,['level','cumulative_codes']):
        f=t['descriptive_fits'][key];xx=x if key=='level' else np.cumsum([q['length'] for q in t['profiles']]);z.plot(xx,[100*q['cell_accuracy'] for q in t['profiles']],'o',label='Observed');z.plot(xx,100*np.array(f['best']['prediction']),label=f"Selected knot: level {f['best']['knot']}");z.plot(xx,100*np.array(f['linear_prediction']),':',label='Single line');z.set(xlabel='Level' if key=='level' else 'Cumulative generated code count',ylabel='Correct cells (%)');z.legend(fontsize=8)
    fig.suptitle('A descriptive fitted bend depends on the horizontal parameterization');save(fig,'06_descriptive_bends')

def documents(o,a,r,s):
    t=o['hard_test'];e=o['external'];v=t['reversals']['corrected9_then_wrong_any'];va=o['hard_all']['reversals']['corrected9_then_wrong_any'];vf=t['reversals']['corrected9_then_wrong_final'];acc=t['secondary_acceleration'];native=a['native']['metrics'];om=a['omit_9'];ex=t['counterexamples'][0]
    profile='\n'.join(f"- Niveau {p['level']} : {p['solved']}/600 grilles ; {pct(p['cell_accuracy'])} de cases exactes ; médiane {p['cell_error_quantiles'][3]:.0f} erreurs." for p in t['profiles'])
    omission='\n'.join(f"- Contribution {k} omise : **{a['omit_'+str(k)]['metrics']['solved']}/128** ; écart {100*a['omit_'+str(k)]['delta_success']['mean']:.2f} points, IC 95 % {interval(a['omit_'+str(k)]['delta_success']['ci95'])} ; {a['omit_'+str(k)]['introduced']} échecs introduits, {a['omit_'+str(k)]['rescued']} corrigés ; {a['omit_'+str(k)]['changed_grids']} grilles modifiées ; norme relative moyenne retirée {a['omit_'+str(k)]['mean_relative_removed_norm']:.3f}." for k in Q['omit_levels'])
    gammas='\n'.join(f"- Gamma {g} : {a['gamma_'+str(g)]['metrics']['solved']}/128 grilles exactes ; {a['gamma_'+str(g)]['changed_grids']} réponses modifiées." for g in Q['continuation_gamma'])
    noise='\n'.join(f"- Dose {pct(float(k))} de norme(z8) : réponses modifiées avant/après {pct(q['before']['change_rate'])}/{pct(q['after']['change_rate'])} ; différence après moins avant {100*q['after_minus_before_change']['mean']:.2f} points, IC {interval(q['after_minus_before_change']['ci95'])}. Réussite avant/après {pct(q['before']['success_rate'])}/{pct(q['after']['success_rate'])} ; contraste de réussite {100*q['after_minus_before_success']['mean']:.2f} points, IC {interval(q['after_minus_before_success']['ci95'])}." for k,q in s.items())
    replications='\n'.join(f"- {name}, n={q['n']} : {q['profiles'][-1]['solved']} grilles exactes finales ; {q['reversals']['corrected9_then_wrong_any']['numerator']}/{q['reversals']['corrected9_then_wrong_any']['denominator']} corrections à 9 redeviennent incorrectes, soit {pct(q['reversals']['corrected9_then_wrong_any']['rate'])}, IC {interval(q['reversals']['corrected9_then_wrong_any']['ci95'])}." for name,q in o.items())
    french=f'''# Dynamique temporelle NCM : corrections utiles, sans verrouillage au niveau 9

Étude exploratoire, 24 octobre 2026. Checkpoints Sudoku publiés, gelés, CPU FP32. La campagne de correction du pooling est conservée intacte. [Note anglaise](PAPER.md), [PDF](PAPER.pdf), [protocole gelé](OBJECTIFS_ET_PROTOCOLE.md).

## Résultat principal

**L'hypothèse d'un verrouillage strict des valeurs décodées au niveau 9 est réfutée par les trajectoires natives.** Sur les 600 puzzles hard test, {v['numerator']} des {v['denominator']} cases incorrectes à 8 puis corrigées à 9 redeviennent incorrectes à 10, 11 ou 12 : **{pct(v['rate'])}**, IC 95 % par puzzle **{interval(v['ci95'])}**. Sur les 2 000 hard, le même constat concerne {va['numerator']}/{va['denominator']} corrections. Cela ne réfute pas toute hypothèse d'engagement dans un état interne non observé ; cela réfute le verrouillage proposé des valeurs lues dans la grille.

L'arrêt à 8 donne 0/600 grilles exactes, contre 516/600 à 12. Mais les arrêts à 9, 10 et 11 donnent eux aussi zéro. Le contraste arrêt-8 moins final vaut -86,00 points ; IC {interval(t['stop_minus_final']['8']['ci95'])}. Ce test montre l'insuffisance du préfixe, sans isoler le rôle de 9. L'omission du seul résidu 9 avec régénération de la suite donne {om['metrics']['solved']}/128, contre {native['solved']}/128 nativement : 16 réponses et 1 962 codes sont modifiés, mais la baisse de réussite de 1,56 point a un IC [-3,91 ; 0,00]. L'intervention établit une influence causale sur certaines réponses, sans démontrer un avantage populationnel de performance ni une nécessité générale du niveau 9.

## 1. Questions et critères de réfutation

Nous séparons cinq questions : continuer est-il nécessaire ; le résidu 9 contribue-t-il à la réponse ; le débit de correction s'accélère-t-il ; les valeurs se verrouillent-elles ; la réponse devient-elle moins sensible aux perturbations ? Aucune réponse isolée n'établit une transition de phase.

Une trajectoire déterministe à douze niveaux, dont les longueurs et les poids diffèrent, ne fournit pas une famille de tailles et un paramètre de contrôle permettant une analyse de taille finie. Une cassure ajustée à douze points est donc décrite comme une courbure, pas comme une transition fondamentale. Plus de puzzles augmente la précision sur la population d'entrées ; cela n'augmente pas la taille du système.

Le pic de changements argmax peut combiner ajout d'information et franchissement de seuils du décodeur. « Décision fonctionnelle » et « artefact du décodeur » ne forment pas une alternative exhaustive. Nous n'attribuons ni intention ni métacognition aux changements de grille.

## 2. Mécanisme observé et données

Le runner explicite suit la récurrence `c_k = argmax G_k(pool_k(z_(k-1)), puzzle)`, puis `z_k = z_(k-1) + R_k(c_k)` et `grille_k = argmax D(z_k)` ; le premier niveau utilise une requête initiale dédiée. `R_k` inclut l'embedding, le rééchantillonnage et la transformation résiduelle du niveau. Les prédictions de codes anciens ne sont jamais révisées sur la trajectoire native : leur fixation est une propriété du protocole à **tous** les niveaux, pas une découverte au seul niveau 9. En revanche, chaque ajout peut modifier la lecture de toute la grille. Il faut donc distinguer fixation des codes déjà émis et verrouillage des valeurs Sudoku décodées.

L'observable principal est le ratio `nombre(c8 faux, c9 correct, au moins un c10..12 faux) / nombre(c8 faux, c9 correct)`, où c désigne ici la valeur de case décodée, distincte des codes VQ de la récurrence. Ce ratio mesure une propriété falsifiable des trajectoires, sans supposer une décision latente.

### Données, exécution et incertitude

Réutilisation explicite des traces natives vérifiées de la campagne initiale : hard/easy/medium, 2 000 puzzles chacun, test hard fixé de 600, et 256 puzzles externes. Ce ne sont pas de nouvelles données aveugles : ces jeux ont déjà été examinés. Les difficultés officielles partagent leurs familles de solutions ; les réplications ne sont pas indépendantes à ce niveau. Les interventions utilisent les 128 premiers puzzles du test hard fixé ; les bruits les 64 premiers, sans sélection selon le résultat.

À l'arrêt k, nous décodons l'accumulation des contributions 1..k, à 89 positions. Il ne s'agit ni du seul bloc k ni d'un remplissage des niveaux manquants par le code d'indice zéro, dont l'embedding n'est pas nul. Le runner explicite est identique au chemin natif sur 32 puzzles pour les codes, logits finaux et accumulations 8/9 ; les 128 trajectoires causales sont aussi comparées aux traces natives sauvegardées. Le décodage au niveau 8 est contrôlé directement.

Les taux de revirement portent sur les 81 cases, indices compris ; les transitions sur cases initialement vides sont également conservées dans le JSON. La réussite exacte porte sur les 89 tokens et est doublée d'un contrôle sur les 81 cases. À tous les arrêts 1..11, **aucun** des 600 puzzles n'a ses 81 cases entièrement exactes : l'échec précoce ne provient pas des séparateurs. Les sorties intermédiaires sont celles du décodeur de solutions complètes, pas celles d'un lecteur optimal entraîné pour chaque niveau ; leur pauvreté ne prouve pas l'absence d'une solution décodable autrement.

Bootstrap percentile apparié, 2 000 tirages par puzzle, poids et modèles fixes. Pour un ratio de corrections, numérateur et dénominateur sont rééchantillonnés ensemble. Pour le bruit, les trois graines sont moyennées dans chaque puzzle avant le bootstrap. Les contrastes sont exploratoires, sans correction de multiplicité. Le protocole est gelé avant cette campagne, après les observations des campagnes antérieures : pas de revendication confirmatoire indépendante.

## 3. Arrêts temporels et trajectoires natives

{profile}

![Arrêts natifs](figures/01_native_stops.png)

Une métrique exigeant 81 cases correctes est conjonctive : une amélioration continue des cases peut produire une augmentation tardive abrupte des grilles exactes. Le saut final ne localise donc pas à lui seul une décision discrète.

Exemple vérifiable : puzzle source **{ex['source_index']}**, ligne **{ex['row']}**, colonne **{ex['column']}**, référence **{ex['reference']}**. Les valeurs aux niveaux 8..12 sont **{' → '.join(map(str,ex['values8_to12']))}**. La case est corrigée à 9, dégradée à 11, puis récupérée à 12. Le puzzle et onze autres contre-exemples figurent dans `results/observations.json`.

Parmi les corrections de 9 du test, {vf['numerator']} restent incorrectes au dernier niveau, soit {pct(vf['rate'])}, IC {interval(vf['ci95'])}. Les revirements transitoires sont donc beaucoup plus fréquents que les erreurs persistantes. {pct(t['reversals']['fraction_values9_changed_later'])} de toutes les valeurs lues à 9 changent encore au moins une fois ; aucune des 600 grilles lues à 9 n'est identique à sa sortie finale. Le nombre de grilles déjà résolues à 9 puis perdues est zéro, mais ce contrôle est vide puisque aucune grille n'était résolue à 9.

{replications}

## 4. Un pic de corrections réel, sans preuve de transition de phase

Le pic historique de 41 620 événements sur hard est un nombre brut de passages incorrect vers correct entre 8 et 9, et non 41 620 décisions irréversibles ni 41 620 puzzles. Le nombre net doit soustraire les dégradations simultanées. Les comptes bruts dépendent également du nombre de cases, de puzzles et du budget de codes par niveau ; une version par code ajouté est conservée.

Le gain net de cases exactes entre 8 et 9 dépasse la moyenne des gains 7→8 et 9→10 de **{100*acc['mean']:.2f} points**, IC {interval(acc['ci95'])}, sur test. Cette accélération locale ne s'accompagne pas du verrouillage proposé.

![Corrections et contre-exemple](figures/02_corrections_reversals.png)

Une droite segmentée continue, sélectionnée parmi huit nœuds possibles sur les mêmes douze points, place sa meilleure cassure au niveau **{t['descriptive_fits']['level']['best']['knot']}** en fonction du niveau, et **{t['descriptive_fits']['cumulative_codes']['best']['knot']}** en fonction du nombre cumulé de codes. Les préférences bootstrap décrivent la stabilité de cet ajustement conditionnel ; elles ne constituent pas un test de transition. La comparaison favorise par construction un modèle plus flexible et ne valide pas une singularité.

![Courbures descriptives](figures/06_descriptive_bends.png)

## 5. Ablations temporelles : contribution utile, intervention imparfaite

Nous calculons normalement la prédiction du niveau ciblé, omettons uniquement sa contribution résiduelle, puis régénérons les niveaux suivants à partir de l'accumulation modifiée. L'horaire, les autres poids et le nombre de têtes exécutées sont conservés. L'omission n'est donc pas une mesure de gain de temps d'un modèle compressé. Baseline : **{native['solved']}/128** grilles exactes.

{omission}

L'omission de 9 conserve l'essentiel de la réussite ; les niveaux suivants peuvent récupérer malgré le résidu absent. La contribution finale domine ce test d'omission, mais son rôle ne prouve pas non plus une transition à 12. Les normes retirées diffèrent entre niveaux. Les écarts ne sont pas un classement parfaitement contrôlé de l'importance spécifique des têtes : supprimer un résidu de grande amplitude peut être plus perturbant. Les interventions peuvent produire des états hors distribution. Elles identifient l'effet de cette suppression précise sur ces puzzles, pas une nécessité universelle ni un mécanisme de décision.

Reprise des niveaux 10..12 depuis z8 + gamma R9, où R9 est la contribution issue des codes natifs de 9 :

{gammas}

Gamma 1 reproduit exactement les codes et sorties natifs ; gamma 0 reproduit exactement l'omission du résidu 9. Gamma 1,5 extrapole hors de l'intervalle natif et reste diagnostique.

![Interventions temporelles](figures/04_temporal_interventions.png)

## 6. Contrôle de lecture fractionnaire

Sur les mêmes 128 puzzles, nous conservons les codes natifs de 9 et décodons 21 points z8 + gamma R9, sans continuation. De gamma 0 à 1, l'exactitude des cases passe de {pct(r['0.0']['cell_accuracy'])} à {pct(r['1.0']['cell_accuracy'])} ; la NLL moyenne du chiffre de référence passe de {r['0.0']['nll_cells']:.4f} à {r['1.0']['nll_cells']:.4f} nat. Les logits, argmax et marges sont sauvegardés pour chaque point.

![Interpolation de lecture](figures/03_reader_interpolation.png)

Cette interpolation examine comment le lecteur transforme une contribution fixe en prédictions discrètes. Les changements argmax sont des franchissements de frontières de lecture. L'interpolation ne remplace pas une trajectoire native avec des codes recalculés, ne teste pas toutes les discontinuités des têtes argmax et ne prouve pas que le niveau 9 serait sans fonction. Le test causal précédent répond séparément à cette dernière question.

## 7. Susceptibilité avant/après 9

Une perturbation gaussienne isotrope normalisée est ajoutée juste avant ou après la contribution 9, puis la suite est régénérée. Pour un puzzle, une dose et une graine, la direction et la norme absolue sont exactement appariées ; la norme est dose × norme(z8) dans les deux cas. Le ratio à norme(z9) n'est donc pas apparié. Avant 9, la tête 9 est recalculée ; après 9, ses codes sont conservés. C'est une comparaison de sensibilité du calcul restant, pas une recherche exhaustive d'attracteurs.

{noise}

![Susceptibilité appariée](figures/05_susceptibility.png)

Les IC [0,0] issus d'aucun changement de statut sont dégénérés et ne prouvent pas une équivalence populationnelle. Le taux de réussite natif des 64 puzzles est 54/64 ; aucun bruit testé ne change ce statut. Deux doses, trois graines et 64 puzzles ne suffisent pas à caractériser toutes les perturbations ou un bassin global. Une invariance à ce bruit ne prouverait pas l'irréversibilité ; une réponse modifiée ne serait pas à elle seule une inversion de décision. Les revirements natifs constituent le test direct du verrouillage des valeurs.

## 8. Conclusion et suite justifiée

Le résultat défendable est une **correction distribuée, non monotone, avec accélération locale autour de 9 et corrections substantielles ultérieures**. Le niveau 9 contribue au calcul, mais les données réfutent son interprétation comme verrouillage strict de la grille. Aucun résultat de cette campagne n'établit une transition de phase architecturale, ni une décision interne consciente ou métacognitive. L'étude ne généralise pas au texte, où ces interventions temporelles n'ont pas été exécutées.

Une suite pertinente serait de comparer des modèles réentraînés à plusieurs tailles et budgets de niveaux, avec un paramètre de contrôle annoncé, puis de chercher un ordre macroscopique et une mise à l'échelle reproductible. Sans réentraînement, on peut chercher un engagement latent plus faible par patching de candidats cohérents et tests de récupération, avec une définition préalable distincte des argmax de grille. Ces pistes restent des propositions, pas des résultats acquis.

## 9. Preuves et reproduction

Depuis ce dossier, `./run.sh` rejoue les observations, les interventions, la publication et la vérification. Les poids, sources et jeux de données antérieurs sont lus dans le dossier parent autorisé ; aucune écriture ne les modifie. Le runner et le code de campagne sont dans `scripts/`. Les traces sources sont les NPZ du parent, explicitement réutilisées ; les nouvelles réponses par intervention et les logits de lecture sont dans `results/interventions.npz` et `results/reader.npz`.

`results/verification.json` précise les contrôles réellement exécutés ; `results/final_manifest.json` fige les artefacts livrés. `sources/previous_manifests.json` lie cette campagne aux quatre états antérieurs vérifiés. `sources_temporelles.zip` archive les scripts utilisés, le code des modèles, les configurations, les protocoles et les archives de sources des quatre campagnes antérieures ; les gros poids et traces restent dans le parent avec leurs empreintes. Le rapport ne remplace pas les données par des exemples sélectionnés : tous les comptes et toutes les réponses des sous-échantillons fixés sont conservés.
'''
    (D/'RAPPORT.md').write_text(french)
    eng_om='\n'.join(f"- Omit residual {k}: {a['omit_'+str(k)]['metrics']['solved']}/128 exact grids, paired difference {100*a['omit_'+str(k)]['delta_success']['mean']:.2f} percentage points, 95% CI {interval(a['omit_'+str(k)]['delta_success']['ci95'])}." for k in Q['omit_levels'])
    eng_noise='\n'.join(f"- Norm dose {100*float(k):.0f}%: changed final grids before/after {100*q['before']['change_rate']:.2f}%/{100*q['after']['change_rate']:.2f}%; after-minus-before difference {100*q['after_minus_before_change']['mean']:.2f} percentage points, 95% CI {interval(q['after_minus_before_change']['ci95'])}." for k,q in s.items())
    paper=f'''---
title: "Distributed Revision Without Level-9 Locking in Sudoku Next-Concept Models"
date: "24 October 2026"
geometry: margin=21mm
fontsize: 10pt
header-includes:
  - \\usepackage{{float}}
  - \\floatplacement{{figure}}{{H}}
---

## Abstract

A peak of 41,620 incorrect-to-correct cell revisions at generation level 9 in a published Sudoku Next-Concept Model motivates an irreversible-decision hypothesis. We distinguish continuation necessity, residual contribution, local acceleration, decoded-value locking and perturbation sensitivity. Native trajectories refute strict locking: on 600 fixed hard test puzzles, {v['numerator']} of {v['denominator']} cells corrected at level 9 become incorrect again at levels 10-12 ({100*v['rate']:.2f}%; puzzle-bootstrap 95% CI {interval(v['ci95'])}). Stopping at levels 8, 9, 10 or 11 solves zero grids, versus 516/600 at level 12. Omitting only residual 9 while regenerating the tail solves {om['metrics']['solved']}/128 grids, versus {native['solved']}/128 natively. We provide matched perturbations, fractional readout controls and explicit counterexamples. The evidence supports distributed non-monotone refinement with local acceleration, rather than irreversible level-9 locking or an established architectural phase transition.

## Scope and falsifiable hypotheses

We study published, frozen Sudoku NCM checkpoints using local CPU FP32 inference. This is a new temporal analysis of previously examined datasets and native traces, not an independent blind replication. The pooling-repair study is preserved separately. The methodological distinction is between a necessary continuation, a useful individual contribution, a locally faster improvement, a locked decoded value and an underlying phase transition. None implies all the others.

Our primary observable is the conditional rate of native reversions: a cell is incorrect at 8, correct at 9 and incorrect at least once at 10, 11 or 12. A single such native counterexample contradicts strict decoded-value locking. It need not exclude an unspecified internal commitment whose readout remains imperfect; that weaker claim requires its own observable definition. We do not attribute intention or metacognition to a deterministic change in a decoded digit.

Twelve levels with different lengths and parameters do not supply a system-size family and control parameter for finite-size scaling. A fitted bend at this resolution cannot establish a phase transition. Adding puzzles improves sampling of inputs, not the size of the computational system. We therefore report local acceleration and descriptive segmented fits without identifying them with a fundamental transition.

## Computational mechanism and measurements

The explicit runner implements the recurrence

$$c_k = \\operatorname{{argmax}} G_k(P_k(z_{{k-1}}),p), \\qquad z_k = z_{{k-1}} + R_k(c_k), \\qquad y_k=\\operatorname{{argmax}} D(z_k).$$

Here p is the puzzle prefix, P is level-specific pooling, and R includes embedding, upsampling and the residual transform. The first level uses its dedicated initial query. Earlier emitted VQ codes are never revised on the native path: this commitment holds at every level by construction. In contrast, each new residual can change the decoded digits anywhere in the grid. Append-only code commitment and decoded Sudoku-value locking are distinct observables.

Our primary ratio is the number of cells wrong at 8, correct at 9 and wrong at least once at 10-12, divided by the number wrong at 8 and correct at 9. Its bootstrap unit is the source puzzle. A last observed change within 12 levels describes a finite horizon, not irreversible convergence beyond that horizon.

### Data, runner and uncertainty

We reuse previously verified cumulative decodings for all 2,000 puzzles of each official difficulty and 256 independently generated external hard puzzles. The original hard split reserves 600 test puzzles. The causal subset is the first fixed 128 test puzzles; susceptibility uses its first 64. Easy, medium and hard share solution families, so they are correlated replications. All datasets had been inspected previously, and comparisons remain exploratory.

At stop k, the decoder receives the sum of contributions 1 through k at 89 positions, including the separators. We neither decode only block k nor fill absent fine blocks with the valid nonzero embedding indexed by code 0. We report exact 89-token success alongside the correctness of the 81 Sudoku cells. All 600 test puzzles have at least one wrong Sudoku cell at every stop before 12, excluding separator-only failure as an explanation.

An explicit cached runner reproduces native codes, final logits and accumulated states 8/9 exactly on 32 puzzles. All 128 intervention baselines additionally match stored native codes and outputs; their level-8 readouts match the original traces. Cache clearing and branch state preservation are checked. No weights, codebooks or decoder parameters are trained or modified.

Uncertainty uses 2,000 paired percentile bootstrap draws by puzzle, conditional on these frozen models. Reversion ratios resample numerator and denominator together. Noise seeds are averaged within each puzzle before bootstrap. No multiplicity correction or confirmatory significance claim is made. The protocol was frozen before this campaign but after earlier exploration.

## Native trajectories: improvement without locking

Correct-cell accuracy on hard test rises from {100*t['profiles'][7]['cell_accuracy']:.2f}% at 8 to {100*t['profiles'][8]['cell_accuracy']:.2f}% at 9, {100*t['profiles'][10]['cell_accuracy']:.2f}% at 11 and {100*t['profiles'][11]['cell_accuracy']:.2f}% at 12. The respective median numbers of wrong cells are 44, 31, 14 and 0. Exact-grid success remains zero through level 11 and reaches 86.00% at 12. Exactness is a conjunction across 81 cells: a late jump in exact-grid accuracy alone does not localize a discrete decision. Early readouts also use a decoder trained for complete representations, so failure does not establish the absence of information accessible to a different trained reader.

![Native stopping curves on fixed test and external puzzles. Exact-grid accuracy alone hides substantial late cell correction.](figures/01_native_stops.pdf)

Of {v['denominator']} test cells corrected at 9, {v['numerator']} become incorrect later ({100*v['rate']:.2f}%, 95% CI {interval(v['ci95'])}). Only {vf['numerator']} are incorrect at the final level ({100*vf['rate']:.2f}%, CI {interval(vf['ci95'])}); transient reversions dominate persistent failures. Across all hard puzzles, the corresponding any-later count is {va['numerator']}/{va['denominator']}. External puzzles show {e['reversals']['corrected9_then_wrong_any']['numerator']}/{e['reversals']['corrected9_then_wrong_any']['denominator']} reversions ({100*e['reversals']['corrected9_then_wrong_any']['rate']:.2f}%). Full easy/medium profiles and reversion counts are retained in the French report and JSON.

For source puzzle {ex['source_index']}, row {ex['row']}, column {ex['column']}, the reference digit is {ex['reference']}; the native values at levels 8-12 are {', '.join(map(str,ex['values8_to12']))}. A correction at 9 is lost at 11 and recovered at 12. In total, {100*t['reversals']['fraction_values9_changed_later']:.2f}% of all test cell values read at 9 change at least once later. None of the 600 grids at 9 equals its final output. Zero solved grids lost after 9 is vacuous here: zero were solved at 9.

![Gross corrections, degradations and net changes, alongside one native counterexample to decoded-value locking.](figures/02_corrections_reversals.pdf)

The historical count 41,620 measures cell events, not puzzles or irreversible decisions. Degradations must be subtracted to obtain net improvement. The net accuracy gain from 8 to 9 exceeds the mean of the adjacent gains 7 to 8 and 9 to 10 by {100*acc['mean']:.2f} percentage points (95% CI {interval(acc['ci95'])}). This supports local acceleration under this observable, not irreversibility. Counts normalized by the added number of codes are also saved.

## Temporal residual interventions

We omit the selected residual contribution while executing its prediction head and regenerating later levels normally. The schedule and other parameters remain fixed. Thus these interventions measure a specific residual removal, not runtime savings from a compressed architecture. The native baseline solves {native['solved']}/128 puzzles.

{eng_om}

Omitting residual 9 changes 16 final grids and 1,962 codes, establishing causal influence on these outputs. Its success difference is -1.56 percentage points (95% CI [-3.91, 0.00]), without a demonstrated population-level performance advantage. Most performance is retained, showing substantial recovery by the remaining computation. Residual 12 dominates this omission test, without establishing a phase transition at 12 either. Residual norms differ across levels and the altered states can be out of distribution. Effect sizes therefore do not establish a perfectly amplitude-controlled ranking of head importance. A large loss from omission can show a functional contribution without identifying an irreversible decision. Continuing from z8 + gamma R9, with native R9 fixed, separately yields {', '.join(f"gamma {g}: {a['gamma_'+str(g)]['metrics']['solved']}/128" for g in Q['continuation_gamma'])}. Gamma 1 exactly matches native codes and outputs; gamma 0 exactly matches omission 9. Gamma 1.5 is a diagnostic extrapolation.

![Final success after residual omission and fractional level-9 amplitude, with the tail regenerated.](figures/04_temporal_interventions.pdf)

## Fractional readout and matched susceptibility

With native level-9 codes fixed, we decode 21 states z8 + gamma R9 without generating further levels. Cell accuracy changes from {100*r['0.0']['cell_accuracy']:.2f}% to {100*r['1.0']['cell_accuracy']:.2f}% and gold-digit negative log likelihood from {r['0.0']['nll_cells']:.4f} to {r['1.0']['nll_cells']:.4f} nats. Saved logits and margins allow direct inspection of threshold crossings in the argmax reader. This path is not a native computation with recomputed discrete codes. Smooth readout interpolation does not imply that every generator-head path is smooth, nor that level 9 is functionless.

![Fixed-code interpolation of the decoder input: argmax quality and gold-digit negative log probability.](figures/03_reader_interpolation.pdf)

We also add the identical normalized Gaussian direction with identical absolute norm immediately before or after level 9, then regenerate the remaining tail. Perturbation norm is dose times norm(z8) in both conditions; relative norm against z9 is not matched. The before condition recomputes level 9 while the after condition preserves its native codes. We use two doses, three seeds and 64 puzzles.

{eng_noise}

![Susceptibility of the remaining computation to matched absolute perturbations. Intervals resample puzzles after averaging seeds.](figures/05_susceptibility.pdf)

Native success on these 64 puzzles is 54/64; none of the tested perturbations changes success status. Degenerate bootstrap intervals [0,0] are not evidence of population equivalence. This limited isotropic stress test cannot certify a global attractor basin. Reduced sensitivity would not prove irreversibility, while a changed final output would not alone establish reversal of an internal decision. Native reversions are the direct test of strict decoded-value locking.

## Descriptive bends and limits of the phase-transition claim

A continuous segmented least-squares model selects a knot among levels 3-10 using the same 12 points on which it is fitted. On test, the selected knot is level {t['descriptive_fits']['level']['best']['knot']} when the abscissa is level number, and level {t['descriptive_fits']['cumulative_codes']['best']['knot']} when it is cumulative generated-code count. Bootstrap preferences describe this conditional fit, not a significance test of a singularity. The flexible model necessarily has an advantage over a single line and is not independently validated.

![Selected descriptive bends depend on the horizontal parameterization of the same accuracy observations.](figures/06_descriptive_bends.pdf)

This campaign supports distributed, non-monotone correction with a local acceleration around level 9. It refutes strict locking of decoded Sudoku values at that level. It neither proves nor universally disproves an internal phase transition under a different operational definition. It does not generalize these temporal interventions to text. Stronger claims would require defined order parameters, controlled model families, repeated training and a reproducible scaling analysis; a weaker latent-commitment hypothesis would require candidate-specific patching and recovery tests with a prespecified criterion.

## Reproducibility

The accompanying French report, frozen protocol, journal and per-puzzle NPZ files document all measurements. Run `./run.sh` in this study directory to reproduce the observations, interventions, figures and verification. The four earlier campaigns are read-only inputs whose manifests are checked again. `sources_temporelles.zip` contains the executed scripts, local model source, configurations and earlier source archives. Large published weights and original traces remain in the parent analysis directory, linked by hashes. Verification results state their actual replay scope; hashes establish artifact identity, not correctness of interpretation.

Primary model source: [NCM article](https://arxiv.org/abs/2610.14162), [official repository](https://github.com/HazyResearch/ncm). Numerical results here are local measurements, not claims copied from these sources.
'''
    (D/'PAPER.md').write_text(paper)

def archive():
    paths=[p for p in (D/'scripts').glob('*.py')]+[D/n for n in ['protocol.json','OBJECTIFS_ET_PROTOCOLE.md','JOURNAL.md','run.sh']]+list((D/'sources').glob('*'))
    with zipfile.ZipFile(D/'sources_temporelles.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            if p.is_file():z.write(p,str(p.relative_to(D)))
        for p in (ROOT/'vendor/ncm_mac').rglob('*'):
            if p.is_file() and p.suffix in ['.py','.yaml','.yml','.json','.toml','.txt'] and '__pycache__' not in str(p):z.write(p,'parent/'+str(p.relative_to(ROOT)))
        for p in [ROOT/'scripts/common.py',ROOT/'protocol.json',ROOT/'results/splits.json',ROOT/'requirements.txt',ROOT/'sources_utilisees.zip',ROOT/'directions_mecanistes/sources_directions_mecanistes.zip',ROOT/'generalisation_architecturale/sources_generalisation.zip',ROOT/'correction_pooling/sources_correction.zip']:
            z.write(p,'parent/'+str(p.relative_to(ROOT)))

def pdf():
    build=D/'.cache/tex-build';build.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['TEXMFVAR']=str(D/'.cache/tex-var');env['TEXMFCONFIG']=str(D/'.cache/tex-config');env['TMPDIR']=str(D/'.cache/tmp');Path(env['TMPDIR']).mkdir(parents=True,exist_ok=True)
    subprocess.run(['/opt/homebrew/bin/pandoc','PAPER.md','-s','--number-sections','--shift-heading-level-by=-1','-V','colorlinks=true','-o','PAPER.tex'],cwd=D,env=env,check=True)
    with (D/'logs/pdf.log').open('w') as f:
        for _ in range(2):subprocess.run(['/Library/TeX/texbin/pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(build),'PAPER.tex'],cwd=D,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    shutil.copy2(build/'PAPER.pdf',D/'PAPER.pdf')

def main():
    o=read('observations.json');a=read('interventions.json');r=read('reader.json');s=read('susceptibility.json');figures(o,a,r,s);documents(o,a,r,s)
    with (D/'results/profiles.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['dataset','level','n','solved','cell_accuracy','blank_cell_accuracy','median_wrong_cells'])
        for name,v in o.items():
            for p in v['profiles']:writer.writerow([name,p['level'],v['n'],p['solved'],p['cell_accuracy'],p['cell_accuracy_blank'],p['cell_error_quantiles'][3]])
    pdf();archive();log('temporal_published')

if __name__=='__main__':main()

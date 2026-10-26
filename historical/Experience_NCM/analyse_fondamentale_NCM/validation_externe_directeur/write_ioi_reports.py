"""Editorial export from conserved summaries; never modifies scientific artifacts."""
from pathlib import Path
import json

D=Path(__file__).resolve().parent/'01_ioi'
def read(p):return json.loads((D/p).read_text())
def put(p,s):
    p=D/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s.rstrip()+'\n')
def f(x):return f'{x:.7g}'
def measure(summary,name):
    v=summary['contrasts'][name];lo,hi=v['ci9875']
    return f"{f(v['mean'])}, IC descriptif 98,75 % [{f(lo)} ; {f(hi)}]"
def listed(s):
    return '\n'.join(f"- `{name}` : {measure(s,name)}." for name in ['D','R','I','S','B',
        'block_backups_minus_adapted','block_controls_minus_adapted','self_repair','delta_DE','ordinary_ablation_damage'])
def scores(s):
    return '\n'.join(f"- `{name}` : deux noms {v['correct_candidate']}/{s['n_resamples']} ; vocabulaire entier {v['correct_vocab']}/{s['n_resamples']}."
        for name,v in s['accuracy'].items())

dev=read('results/dev_summary.json')
test_path=D/'results/test_summary.json'
test=read('results/test_summary.json') if test_path.exists() else None
subject=test or dev; phase='réservée' if test else 'de développement'
prefix='test' if test else 'dev'
verification=read(f'results/{prefix}_verification.json') if (D/f'results/{prefix}_verification.json').exists() else None
secondary=[]
for name in ['_L9H6','_L10H0']:
    p=D/f'results/test{name}_summary.json'
    if p.exists():secondary.append(read(f'results/test{name}_summary.json'))
secondary_text='\n\n'.join('### '+s['head'].lstrip('_')+' — secondaire descriptif\n\n'+listed(s)+'\n\n'+scores(s) for s in secondary)
if len(secondary)<2:secondary_text+='\n\nLes réplications secondaires non encore présentes ne sont pas annoncées comme exécutées.'

scope='''Les intervalles rééchantillonnent les familles en conservant leurs deux donneurs. Les sources sont aussi réemployées comme donneurs à d'autres familles : cette dépendance n'est pas entièrement traitée par ce bootstrap. Ces intervalles sont des approximations conditionnelles aux poids, aux sources et au plan de donneurs ; ils ne certifient ni une population de toutes les formulations IOI ni la variation entre réentraînements. Les deux réplications secondaires ne participent pas à la famille confirmatoire primaire. Les quatre intervalles principaux D,R,S,B utilisent le niveau98,75% annoncé ; ceux des autres quantités sont descriptifs, sans nouvelle revendication de contrôle de multiplicité global.

Les seuils de couverture locale sont des seuils d'effet descriptifs appliqués aux observations, pas un taux d'attributions mécanistes correctes. Les mesures DLA, self-repair centré et NLL n'ont pas le même estimand. Un désaccord ne prouve pas qu'une méthode antérieure se trompe.'''

main=f'''# GPT-2 small × MIB–IOI — rapport d'application

25 octobre 2026. [Feuille de route](FEUILLE_DE_ROUTE.md), [protocole dev](PROTOCOLE_DEV.md), [protocole réservé](PROTOCOLE_TEST.md), [amendements conservés](AMENDEMENTS.md). Rapports exclusivement Markdown, sans entraînement supplémentaire.

## État réellement atteint

{'L’évaluation primaire sur512familles réservées est exécutée.' if test else 'Le pilote128familles est achevé ; les sorties réservées ne sont pas encore disponibles dans ce rapport.'} Le dommage matériel est désormais présent sur IOI, contrairement aux anciens cas peu dommageables. Cela rend la récupération testable dans ce contrat ; cela ne transforme pas une description fonctionnelle en vérité complète du mécanisme de GPT-2.

Le développement fixé avant inférence donne D={f(dev['gate']['mean_damage'])}nat et {f(dev['gate']['fraction_damage_over_001']*100)}% des familles au-dessus de0,01nat. La porte pratique ≥0,05nat et ≥25%familles est franchie. Le sham est exact et huit interventions L9H9 ont été comparées bit à bit à la passe complète officielle avec hook avant ouverture du test. Aucun cas n'a été conservé ou exclu selon son dommage, sa justesse native ou sa récupération.

## Données, indépendance et modèle

MIB–IOI provient des [données publiques externes](https://huggingface.co/datasets/mib-bench/ioi), révision épinglée dans [la provenance](data/provenance.json). Les partitions officielles sont respectées :128sources dev et512test ; variantes A, s2_io_flip B et s1_io_flip Aplus groupées. Les étiquettes et la tokenisation ont été contrôlées avant sorties ; les20000? exemples ne constituent pas notre effectif : les corpus officiels lus comportent10000dev et1000test, notre sélection est128/512. Aucun des10000dev ou1000test n'est exclu par les critères techniques. L'audit relève neuf templates distincts dans chaque sélection et aucun template littéral commun aux deux. Les variantes d'une même famille ne sont pas de nouvelles unités indépendantes.

GPT-2 small local, révision607a30d783dfa663caf39e06633721c8d4cfcd7e. CPU FP32, un thread, eager attention, aucun cache KV mutable ; BOS50256 explicite et noms chacun à un token avec espace initial. Il ne s'agit pas d'une reproduction bit à bit de tout le pipeline officiel MIB. Les données, le choix des composants et les antécédents sont extérieurs ; l'implémentation, les métriques et les audits restent internes.

## Contrat causal et contrôles

La tête primaire est L9H9, dernière position. Sa sortie64dimensions est remplacée par la valeur native d'une autre source. La projection pleine officielle reste FP32. La frontière est après attention totale et avant le MLP de la même couche. La continuation recalcule ce MLP puis les blocs restants, avec les représentations des tokens antérieurs du receveur fixées à leurs valeurs natives. Elle a donc encore accès à un canal contextuel du receveur. L9H6 et L10H0 ont chacune leur frontière et restent secondaires.

Deux donneurs sont fixés par décalages37/73 à site identique. Y00,Y10,Y11,Y01,Y1wrong croisent état natif/corrompu et suites natives/adaptées/mal appariées. Les deux derniers bras de confirmation bloquent l'adaptation des backups publiés strictement aval, ou d'un témoin égal en nombre et stratifié par couche. Les têtes simultanées L9H0/H7 ne sont pas appelées « aval ». Les conditions E croisent A/B et A/Aplus avec état et suite, sans attribuer une classe d'architecture à partir d'une suite figée.

## Résultats de l'évaluation {phase} primaire

{subject['n_families']}familles et{subject['n_resamples']}remplacements. D,R,I,S,B sont en nats ; self-repair et delta_DE sont des effets de logit centré, à ne pas additionner aux NLL.

{listed(subject)}

Le dommage conventionnel est L11−L00 : il conserve le dommage résiduel après toute adaptation. D−R lui est égal par définition. Une récupération positive ne signifie donc ni retour exact au natif ni amélioration de l'entrée non perturbée.

## Justesse : deux noms et vocabulaire entier

{scores(subject)}

Y00 est répété pour les deux donneurs :{subject['n_resamples']}lignes Y00 ne sont pas autant de sources natives différentes. La réussite parmi deux candidats ne suffit pas à garantir le bon token dans tout le vocabulaire. Les sorties Y01 ou mal appariées sont conservées même lorsqu'elles améliorent un score.

## Couverture, abstention et confirmation

La règle d'effets locaux D>0,05/R>0,01 conserve{subject['claim_coverage']['without_specificity']}/{subject['n_families']}familles ; ajouter S>0,01 en conserve{subject['claim_coverage']['complete']}/{subject['n_families']}. Ces comptes ne sont pas une précision de classification : ils décrivent combien de cas satisfont les contrastes observés. La version sans contrôle figé ne peut établir la nécessité de l'adaptation. La version sans appariement ne peut établir sa spécificité.

Le blocage doit être lu avec ses deux effets absolus, pas seulement B. Une différence backups−témoin positive ne suffit pas si les deux blocages améliorent la perte. Toute attribution à ces backups requiert aussi une perte augmentée par rapport à Y11. Le rôle éventuel des MLP, d'autres têtes et du lecteur reste ouvert ; aucune exclusivité du groupe n'est démontrée. Les [résultats par template](results/{prefix}_summary.json) restent visibles et ne servent pas à choisir après coup une formulation favorable.

{scope}

## Réplications secondaires

{secondary_text}

## Vérification, incidents et coûts

Le premier passage s'arrête à l'export non fini, puis A1 localise un NaN dans la projection de mesure DE. Un produit vectoriel FP32 a produit181NaN avec entrées finies ; ce comportement n'est pas reproduit dans toutes les dispositions mémoire, notamment après copie des poids. La cause exacte n'est pas établie. A2 passe cette mesure seulement enFP64, vérifiée contre NumPy et sommation scalaire ; le réseau et toutes ses interventions restent FP32. Les archives initiale/A1 et les marges inchangées sont conservées. Le [diagnostic isolé](results/projection_incident.json) ne doit pas être présenté comme une reproduction universelle du NaN.

L'audit primaire est {'passé' if verification and verification.get('passed') else 'à finaliser'} : les pertes sont recalculées par sommation scalaire et les sorties officielles sont rejouées sur quatre sources fixées, deux donneurs chacune. Les shams et recompositions sont contrôlés à chaque source par le runner. Le script initial ne couvre pas les secondaires ; un audit supplémentaire distinct peut les couvrir, ainsi que les blocages et E. Ces vérifications logicielles restent internes.

Temps du runnerdev valide :{f(dev['elapsed_seconds'])}s. Temps du runner{phase} décrit ici :{f(subject['elapsed_seconds'])}s, chargement et calcul jusqu'à la sauvegarde des états inclus ; les audits sont séparés. Les incidents et installations ont aussi un coût qui n'est pas contenu dans ce seul temps valide. La machine8Go a connu une pression mémoire et des travaux concurrents : ces durées ne sont pas un benchmark matériel ni une preuve d'efficacité relative d'une méthode.

## Portée et arrêt

Les contrôles de contributions et de normalisation figées sont des antécédents directs de [Rushing et Nanda](https://arxiv.org/html/2402.15390v2). Les backups ont été décrits par [Wang et al.](https://arxiv.org/html/2211.00593v1), notamment sous knockout conjoint des trois Name Movers, différent de notre intervention unitaire. Cette campagne apporte une épreuve tenue à l'écart du développement, avec un dommage utile et des contrôles de donneurs et de blocage. Elle ne découvre pas la compensation de GPT-2 et ne démontre pas une supériorité prédictive générale face aux mesures publiées.

L'utilité supplémentaire doit porter sur une conclusion précise corroborée, et non sur un simple désaccord d'unités ou un taux d'abstention plus élevé. Aucun score de vérité mécaniste générale ni statut de benchmark de référence n'est assigné. Une implémentation par une personne extérieure et une comparaison de méthodes indépendamment développées restent nécessaires pour ces revendications plus larges.
'''
# Remove an editorial scratch expression rather than leaving it in the report.
main=main.replace('les20000? exemples ne constituent pas notre effectif : ','')
put('RAPPORT.md',main)

put('02_comparaison/RAPPORT.md',f'''# D02 — Rapport de comparaison des estimands

## Calcul effectivement réalisé

Évaluation {phase} primaire :{subject['n_families']}familles,{subject['n_resamples']}remplacements. Tous les comparateurs utilisent les mêmes sources, donneurs et poids. Les résultats complets et leurs intervalles sont dans [le rapport IOI](../RAPPORT.md).

- Effet total d'ablation régénérée, L11−L00 :{measure(subject,'ordinary_ablation_damage')}nat.
- Variation d'effet direct de la tête, facteur LN original :{measure(subject,'delta_DE')}logit centré.
- Self-repair publié, Δlogit centré−ΔDE :{measure(subject,'self_repair')}logit centré.
- Récupération NLL sous adaptation, L10−L11 :{measure(subject,'R')}nat.
- Appariement spécifique de la suite, L1wrong−L11 :{measure(subject,'S')}nat.

## Ce que la comparaison permet

L'effet total conserve toutes les réponses du système après perturbation ; DLA isole la projection directe selon un lecteur annoncé ; le self-repair soustrait cette variation directe au logit centré ; R mesure un changement de perte sous deux continuations. Ces objets ne sont pas des estimateurs concurrents d'une unique vérité. Les mesures restent dans leurs unités, sans classement de « classification correcte » inventé pour les auteurs antérieurs.

SansY10, la nécessité d'une continuation adaptée n'est pas testée. SansY1wrong, sa spécificité n'est pas testée. La règle complète conserve{subject['claim_coverage']['complete']}familles à ses seuils locaux, contre{subject['claim_coverage']['without_specificity']}sans spécificité. Il ne s'agit pas d'une courbe de précision validée par un oracle interne de GPT-2. Le blocage annoncé fournit une corroboration fonctionnelle supplémentaire, pas des labels de mécanismes universels.

## Valeur ajoutée et travail restant

Le calcul joint rend les hypothèses de lecture plus explicites et permet de conserver les cas négatifs. Les versions amputées sont des ablations de notre protocole ; elles ne comptent pas comme autant de méthodes développées indépendamment. EAP-IG, méthode de localisation, n'est pas ajouté à un classement de compensation sans sortie commune. Cette évaluation ne démontre donc pas une supériorité générale sur plusieurs méthodes indépendantes. Le [rapport méthodologique](../../00_positionnement/02_methodes/RAPPORT.md) précise les estimands et critères.

{scope}
''')

put('03_specificite/RAPPORT.md',f'''# D05 — Rapport de spécificité à site constant

## Mise en œuvre

Chaque source possède deux donneurs distincts fixés avant effets. Pour une tête donnée, les corruptions ont le même support — dernière position, même tête entière. Le suffixe mal apparié a été adapté au second donneur, puis appliqué à la première corruption, et réciproquement. Les normes ne sont pas automatiquement égales ; le contrôle réduit le changement de support sans l'éliminer comme explication géométrique générale.

## Résultat {phase}

{subject['n_families']}familles,{subject['n_resamples']}échanges. S={measure(subject,'S')}nat. La récupérationR={measure(subject,'R')}nat reste une mesure distincte. Ajouter le seuilS>0,01 àD>0,05/R>0,01 fait passer le nombre de familles retenues de{subject['claim_coverage']['without_specificity']}à{subject['claim_coverage']['complete']}. Aucun donneur n'a été choisi pour accroîtreS. Les familles avecS nul ou négatif sont conservées.

L'argmax vocabulaire est correct pour{subject['accuracy']['Y11']['correct_vocab']}/{subject['n_resamples']}suites appariées et{subject['accuracy']['Y1wrong']['correct_vocab']}/{subject['n_resamples']}mal appariées. Une différence de confiance peut subsister sans changer cet argmax. Une suite mal appariée nuisible peut aussi augmenterS ;S ne prouve ni une représentation symbolique de l'erreur ni une réparation unique.

## Preuves et limites

Les identités de donneurs, mesures, états et suites sont conservées dans `../results/{prefix}_observations.json` et `../results/{prefix}_states.npz`. Le runner réservé ajoute normes effectives au point de coupure et normes projetées idéales. Le pilote initial ne les exportait pas séparément ; aucune collection absente n'est présentée comme conservée. L'ensemble de sources sert aussi de banque donneuse, ce qui limite l'interprétation des intervalles. L'évaluation sur des donneurs futurs globalement disjoints reste à réaliser pour une précision de prédiction de spécificité.

{scope}
''')

abs_effect=subject['contrasts']['block_backups_minus_adapted']
diff=subject['contrasts']['B']
corroborated=(subject['contrasts']['D']['ci9875'][0]>.05 and subject['contrasts']['R']['ci9875'][0]>.01 and
              diff['ci9875'][0]>.01 and abs_effect['ci9875'][0]>0)
put('04_blocage/RAPPORT.md',f'''# D06 — Rapport de blocage des backups publiés

## Intervention indépendante des contrastes de proposition

Le groupe est déterminé par la publication IOI, avant les résultats : quatre backups couche10 et deux couche11, strictement aval de L9H9. Le témoin a la même répartition4+2. On conserve leurs valeurs natives pendant la reprise perturbée ; les autres composants s'adaptent. Les têtes simultanées de couche9 sont exclues de l'interprétation aval. Le témoin n'est pas supposé inerte.

## Résultat {phase} primaire

- Blocage backups moins continuation adaptée :{measure(subject,'block_backups_minus_adapted')}nat.
- Blocage témoin moins continuation adaptée :{measure(subject,'block_controls_minus_adapted')}nat.
- Effet différentielB :{measure(subject,'B')}nat.
- Justesse vocabulaire :Y11{subject['accuracy']['Y11']['correct_vocab']}/{subject['n_resamples']}, backups figés{subject['accuracy']['block_backups']['correct_vocab']}/{subject['n_resamples']}, témoins figés{subject['accuracy']['block_controls']['correct_vocab']}/{subject['n_resamples']}.

La condition de corroboration restreinte (dommage et récupération utiles, blocage absolument défavorable, effet différentiel au-delà de la marge annoncée) est {'satisfaite dans cette évaluation' if corroborated else 'non satisfaite dans cette évaluation'}. Cette formulation ne transforme pasB seul en nécessité. Les résultats sont conservés quel que soit leur signe, sans recherche d'une autre liste de composants sur ces mêmes sorties.

## Interprétation délimitée

Cette expérience éprouve si empêcher l'adaptation de ce groupe publié dégrade la récupération davantage qu'un groupe témoin. Elle ne prouve pas que toutes ces têtes contribuent individuellement, que leurs effets sont additifs, ni que toute récupération passe par elles. L'inhibition négative, les MLP et la normalisation finale restent accessibles. La comparaison publie les effets absolus et le témoin afin d'éviter la conclusion trompeuseB>0⇒backups nécessaires.

Le résultat ne constitue pas une nouvelle découverte générale du self-repair. Il corrobore ou contredit une attribution fonctionnelle annoncée sous une intervention précise ; le circuit IOI publié reste une explication partielle. Les [résultats par template](../results/{prefix}_summary.json), les suites et les cas défavorables sont conservés.
''')
print('Wrote four IOI reports; reserved primary present:',bool(test),'secondary present:',len(secondary))

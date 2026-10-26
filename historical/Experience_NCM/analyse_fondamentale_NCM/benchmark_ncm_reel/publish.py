"""Render the retained pilot in Markdown; no graphics or model calls."""
import json
from pathlib import Path

D = Path(__file__).resolve().parent


def main():
    q = json.loads((D/'results/summary.json').read_text())
    audit = json.loads((D/'results/verification.json').read_text())
    execution = json.loads((D/'results/execution.json').read_text())
    frozen = json.loads((D/'results/protocol_frozen.json').read_text())
    assert audit['status'] == 'passed'
    def fmt(x): return f'{x:.6f}'.replace('.',',')
    def effect(item,adjusted=False):
        ci = item['ci_adjusted'] if adjusted else item['ci95']
        return f"{fmt(item['mean'])} nat, IC {'98,75' if adjusted else '95'} % [{fmt(ci[0])} ; {fmt(ci[1])}]"
    ctext = '\n'.join(f"- **{name}** : {effect(q['C'][name],True)}." for name in ('D','R','I','S'))
    native = '\n'.join(f"- **{name}** : {item['final']['solved']}/16 grilles exactes ; "
                       f"exactitude de case {100*item['final']['cell_accuracy']:.2f} % ; "
                       f"{item['final']['malformed']} grilles avec chiffres hors 1..9."
                       for name,item in q['O'].items())
    elabels = '\n'.join(f'- `{name}` : **{count}/16** familles.' for name,count in q['E']['label_counts'].items())
    families = '\n'.join(f"- `{r['source_id']}` : natif A/B exact={r['native_exact_A']}/{r['native_exact_B']}, "
                         f"g={fmt(r['native_span'])}, u={fmt(r['input_effect'])}, v={fmt(r['state_effect'])}, "
                         f"w={fmt(r['interaction'])} nat ; `{r['functional_label']}`."
                         for r in q['E']['per_family'])
    invariant = '\n'.join(f"- `{name}` : {effect(item) if not name.endswith('TV') else fmt(item['mean'])+' (variation totale moyenne ; IC 95 % ['+fmt(item['ci95'][0])+' ; '+fmt(item['ci95'][1])+'])'}."
                          for name,item in q['E']['invariant'].items())
    text = f"""# Étape 3a — test du protocole sur les vrais états NCM

25 octobre 2026. **Les régimes O, C et E ont été exécutés sur les 16 familles contre-factuelles, avec le checkpoint hard gelé et une inférence CPU FP32.** L'adaptateur reproduit le natif bit à bit sur les 48 entrées et l'audit séparé passe. Cela établit la faisabilité technique des interventions testées, pas une identification certaine du mécanisme interne.

Décision C préfixée : **`{q['C']['decision']}`**. Les diagnostics E sont détaillés ci-dessous. Les seuils, niveaux et doses n'ont pas été modifiés après les sorties. Les résultats insuffisants restent conservés ; « non identifiable » ne signifie ni preuve de mélange ni preuve d'une complexité intrinsèque supérieure aux témoins.

## 1. Protocole et statut

[Protocole complet](PROTOCOLE.md), [paramètres](protocol.json). Gel local à **{frozen['utc']}**, avant les nouvelles inférences, incluant scripts scientifiques, données, poids/configuration et sources importées. Ce n'est pas une prérégistration indépendante. Les puzzles et réponses étaient connus par construction ; leurs sorties NCM n'avaient pas servi à régler cette campagne.

Un seul checkpoint hard, déterministe, sans mise à jour des poids. Lots unitaires, CPU FP32, un thread ; statistiques calculées en FP64 sur les logits conservés. A, B et Aplus représentent 48 entrées issues de 16 bases à deux solutions. Leurs unicités et comptes de solutions sont revérifiés par deux solveurs avec limite de trois. A/B diffèrent par un indice qui change la réponse ; Aplus ajoute un indice vrai en conservant la solution A et augmente donc le nombre d'indices.

Le masque primaire garde les cases vides dans les trois variantes. La préférence A/B ne porte que sur les réponses différentes de ce masque : ni l'indice discriminant ni l'indice supplémentaire ne sont comptés comme transport de réponse. Les références ne sont jamais données au générateur ni utilisées pour choisir une perturbation favorable.

## 2. O — performances et trajectoires natives

{native}

Les douze lectures natives et les comptes de corrections/dégradations à chaque transition sont conservés dans `native.npz` et `summary.json`. Le décodage au niveau 9 constitue également l'ablation de toute la suite 10..12. Aucun niveau spectaculaire n'est sélectionné pour remplacer le bilan complet. Une trajectoire non monotone est une observation de reconstruction, pas une preuve que le modèle relit ses erreurs ou raisonne à nouveau.

Les taux 15 % et 100 % de la calibration analytique concernaient ses règles et sa bibliothèque de programmes. Ils ne sont ni une limite universelle des observations, ni une garantie de précision d'attribution sur ce réseau. Ici O reçoit explicitement une conclusion mécaniste non identifiable.

## 3. C — récupération sous perturbations contrôlées

Point de coupure fixé : après le niveau 9. Deux supports distincts d'une position latente 89×256, quatre perturbations gaussiennes signées par ancrage, sur A et B : **128 essais**, conservés sans sélection sur dommage. Norme locale `0.5*||z9||/sqrt(89)`. Le rapport de norme global moyen est **{100*q['diagnostics']['perturbation_relative_norm_mean']:.3f} %**. Ce sont des interventions continues potentiellement hors distribution, pas des écritures symboliques de chiffres.

Les cinq lectures correspondent au natif Y00, à la perturbation avec suite figée Y10, à la reprise Y11, au retour de la suite adaptée sur l'état natif Y01 et à une suite adaptée à l'autre support Y1wrong. Normes, signes, codes et contributions sont conservés. Les trois résidus sont ajoutés séquentiellement en FP32, sans réassociation de somme.

Les NLL sont moyennées sur les cases vides communes, puis sur les quatre essais et les deux ancrages de chaque famille. Les IC portent sur **16 familles**, pas sur 128 essais indépendants. Bootstrap percentile de 20 000 tirages, IC ajustés 98,75 % pour D/R/I/S, marge fixée de 0,01 nat par case vide.

{ctext}

D mesure le dommage avec suite figée ; R la récupération par reprise ; I l'interaction croisée ; S l'avantage sur une suite adaptée à une autre corruption de même norme. La règle exige les quatre bornes inférieures ajustées au-dessus de 0,01 pour soutenir une récupération fonctionnelle utile et spécifique dans ce domaine. Une équivalence du dommage moyen n'est annoncée que si son IC ajusté entier tient dans [-0,01 ; 0,01]. Un IC incluant zéro ne prouve pas une équivalence.

Décision exécutée : **`{q['C']['decision']}`**. Le seuil est un choix exploratoire préfixé, sans validation indépendante d'utilité. La fraction de codes tardifs modifiés vaut **{100*q['C']['tail_code_changed_fraction']:.2f} %** ; elle ne remplace pas la mesure d'utilité. Les résultats par signe sont descriptifs et ne changent pas la décision principale. Aucune classe analytique n'est assignée au générateur.

## 4. E — entrée contre état, puis contrôle invariant

Deux plans à 12 conditions par famille : A/B et A/Aplus, soit **384 sorties**. L'entrée réceptrice fournit seule le cache de préfixe ; l'état donneur est z9, sans cache donneur ni réalignement ajusté. Les deux suites natives sont figées séparément, ou la suite est recalculée depuis la combinaison entrée/état. Les diagonales reproduisent le natif ; avec état et suite figés, le préfixe receveur ne change pas la sortie du décodeur.

Sur A/B, la préférence moyenne `m=log p(S_B)-log p(S_A)` donne les effets suivants, sur toutes les familles :

- Effet d'entrée u : **{effect(q['E']['input_effect'])}**.
- Effet d'état v : **{effect(q['E']['state_effect'])}**.
- Interaction w : **{effect(q['E']['interaction'])}**.
- Étendue native g : **{effect(q['E']['native_span'])}**.

L'identité **u+v=g** vaut par construction algébrique : leurs ratios ne sont pas des fractions causales d'un raisonnement. Le diagnostic individuel exige deux grilles natives exactes, g>0,1 nat et |w/g|<=0,2. Les ratios d'entrée >=0,8 ou <=0,2 indiquent une domination descriptive ; les ratios intermédiaires restent « mélange ou mécanisme non modélisé ». Les interactions ou ratios non monotones ne sont pas forcés dans cette classification. Toutes les familles restent dans les effets globaux.

{elabels}

Une préférence dominée par l'entrée ne démontre pas un recalcul intégral depuis zéro ; une préférence dominée par l'état ne démontre pas l'absence d'accès aux indices. Ces conclusions portent sur la réponse A/B, ce point de coupure et ces interventions, sans localisation de circuit interne.

Sur le contrôle A/Aplus, la solution étant identique, aucune normalisation par une différence de réponse n'est appliquée :

{invariant}

Les changements de NLL sont en nats par case vide ; les distances TV sont sans unité. L'information supplémentaire peut améliorer ou modifier le calcul sans changer la référence. Il serait incorrect de compter tout changement comme une attribution fausse. L'ajout d'un indice conserve le facteur de confusion de cardinalité signalé au protocole.

## 5. Diagnostics par famille

Les valeurs u/v/w/g ci-dessous sont en nats, calculées sur les cases de solution différentes restées vides. Les libellés sont des diagnostics conditionnels, sans vérité mécaniste attribuée au réseau.

{families}

## 6. Décalage de distribution et validité

Les coordonnées perturbées sortant de l'enveloppe min/max coordonnée par coordonnée des 48 états natifs z9 représentent en moyenne **{100*q['diagnostics']['envelope_coordinate_fraction_mean']:.4f} %**, maximum **{100*q['diagnostics']['envelope_coordinate_fraction_max']:.4f} %**. Cette mesure d'amplitude n'est ni une densité estimée ni une preuve que les autres coordonnées seraient sur distribution. Les états natifs échangés peuvent eux-mêmes être hors distribution conditionnellement à leur nouveau préfixe.

Une absence de récupération confirmée peut refléter la dose, le point de coupure, la précision statistique ou une continuation peu utile. Une sensibilité aux deux sources peut refléter interactions, redondance ou mécanisme non modélisé ; elle ne suffit pas à prouver un mélange structurel. Seize familles et un checkpoint n'autorisent aucune généralisation architecturale. Les IC bootstrap à si petit effectif restent approximatifs et ne prennent pas en compte un réentraînement.

## 7. Audit, artefacts et arrêt

Les 48 entrées passent la parité native bit à bit, y compris les 12 accumulations, et la reprise sans perturbation. L'audit reconstruit **{audit['counts']['decoder_reconstructions']} lectures** à partir des contributions sauvegardées et rejoue **{audit['counts']['neural_tail_replays']} suites neurales** des deux premières familles avec une boucle explicite distincte. Il recalcule NLL et préférences indépendamment, les intervalles C, les diagnostics E, et les comptes de transitions. Aucun de ces contrôles logiciels ne représente une réplication scientifique indépendante ; ils sont écrits et exécutés par le même agent.

Les empreintes des entrées, poids et sources gelés restent inchangées. Durée du runner : **{execution['elapsed_seconds']:.1f} s**, contrôles et sauvegardes compris ; ce n'est pas un benchmark de vitesse.

- [Protocole](PROTOCOLE.md), [gel](results/protocol_frozen.json), [journal](JOURNAL.md).
- [Résumé et diagnostics](results/summary.json), [audit](results/verification.json).
- `results/native.npz` : codes, états, résidus et logits des 12 niveaux pour les 48 entrées.
- `results/crossed.npz` : deltas, suites et logits C ; `results/exchanges.npz` : les 24 conditions E par famille.
- [Reproduction](README.md). Les archives Numpy se chargent sans pickle.

**Arrêt après l'étape 3a.** Aucun changement du manuscrit ou des anciennes campagnes ; aucune expérience Wyner, aucun changement opportuniste de seuil ou de dose et aucun reclassement certain du NCM.
"""
    (D/'RAPPORT.md').write_text(text)
    print(D/'RAPPORT.md')


if __name__ == '__main__':
    main()

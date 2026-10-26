# Étape 3a — test du protocole sur les vrais états NCM

<!-- lecture-sorties -->
**Contrôle essentiel :** sur les échanges A↔B, la suite recalculée et la suite native du receveur figée donnent toutes deux **32/32 solutions du receveur**. L'adaptation de la suite n'est donc pas nécessaire à la justesse finale dans ces cas. En C, les 128 perturbations conservent déjà les solutions avec suite figée. Voir le [complément de lecture et de format](COMPLEMENT_SORTIES.md), qui précise le statut post-traitement de ces comptes sans modifier les décisions gelées.

25 octobre 2026. **Les régimes O, C et E ont été exécutés sur les 16 familles contre-factuelles, avec le checkpoint hard gelé et une inférence CPU FP32.** L'adaptateur reproduit le natif bit à bit sur les 48 entrées et l'audit séparé passe. Cela établit la faisabilité technique des interventions testées, pas une identification certaine du mécanisme interne.

Décision C préfixée : **`mean_frozen_damage_equivalent_within_prespecified_margin`**. Les diagnostics E sont détaillés ci-dessous. Les seuils, niveaux et doses n'ont pas été modifiés après les sorties. Les résultats insuffisants restent conservés ; « non identifiable » ne signifie ni preuve de mélange ni preuve d'une complexité intrinsèque supérieure aux témoins.

## 1. Protocole et statut

[Protocole complet](PROTOCOLE.md), [paramètres](protocol.json). Gel local à **2026-10-25T13:57:39Z**, avant les nouvelles inférences, incluant scripts scientifiques, données, poids/configuration et sources importées. Ce n'est pas une prérégistration indépendante. Les puzzles et réponses étaient connus par construction ; leurs sorties NCM n'avaient pas servi à régler cette campagne.

Un seul checkpoint hard, déterministe, sans mise à jour des poids. Lots unitaires, CPU FP32, un thread ; statistiques calculées en FP64 sur les logits conservés. A, B et Aplus représentent 48 entrées issues de 16 bases à deux solutions. Leurs unicités et comptes de solutions sont revérifiés par deux solveurs avec limite de trois. A/B diffèrent par un indice qui change la réponse ; Aplus ajoute un indice vrai en conservant la solution A et augmente donc le nombre d'indices.

Le masque primaire garde les cases vides dans les trois variantes. La préférence A/B ne porte que sur les réponses différentes de ce masque : ni l'indice discriminant ni l'indice supplémentaire ne sont comptés comme transport de réponse. Les références ne sont jamais données au générateur ni utilisées pour choisir une perturbation favorable.

## 2. O — performances et trajectoires natives

- **A** : 16/16 grilles exactes ; exactitude de case 100.00 % ; 0 grilles avec chiffres hors 1..9.
- **B** : 16/16 grilles exactes ; exactitude de case 100.00 % ; 0 grilles avec chiffres hors 1..9.
- **Aplus** : 16/16 grilles exactes ; exactitude de case 100.00 % ; 0 grilles avec chiffres hors 1..9.

Les douze lectures natives et les comptes de corrections/dégradations à chaque transition sont conservés dans `native.npz` et `summary.json`. Le décodage au niveau 9 constitue également l'ablation de toute la suite 10..12. Aucun niveau spectaculaire n'est sélectionné pour remplacer le bilan complet. Une trajectoire non monotone est une observation de reconstruction, pas une preuve que le modèle relit ses erreurs ou raisonne à nouveau.

Les taux 15 % et 100 % de la calibration analytique concernaient ses règles et sa bibliothèque de programmes. Ils ne sont ni une limite universelle des observations, ni une garantie de précision d'attribution sur ce réseau. Ici O reçoit explicitement une conclusion mécaniste non identifiable.

## 3. C — récupération sous perturbations contrôlées

Point de coupure fixé : après le niveau 9. Deux supports distincts d'une position latente 89×256, quatre perturbations gaussiennes signées par ancrage, sur A et B : **128 essais**, conservés sans sélection sur dommage. Norme locale `0.5*||z9||/sqrt(89)`. Le rapport de norme global moyen est **5.300 %**. Ce sont des interventions continues potentiellement hors distribution, pas des écritures symboliques de chiffres.

Les cinq lectures correspondent au natif Y00, à la perturbation avec suite figée Y10, à la reprise Y11, au retour de la suite adaptée sur l'état natif Y01 et à une suite adaptée à l'autre support Y1wrong. Normes, signes, codes et contributions sont conservés. Les trois résidus sont ajoutés séquentiellement en FP32, sans réassociation de somme.

Les NLL sont moyennées sur les cases vides communes, puis sur les quatre essais et les deux ancrages de chaque famille. Les IC portent sur **16 familles**, pas sur 128 essais indépendants. Bootstrap percentile de 20 000 tirages, IC ajustés 98,75 % pour D/R/I/S, marge fixée de 0,01 nat par case vide.

- **D** : 0,000002 nat, IC 98,75 % [-0,000001 ; 0,000006].
- **R** : -0,000007 nat, IC 98,75 % [-0,000030 ; 0,000000].
- **I** : 0,000001 nat, IC 98,75 % [-0,000000 ; 0,000004].
- **S** : 0,000001 nat, IC 98,75 % [-0,000000 ; 0,000004].

D mesure le dommage avec suite figée ; R la récupération par reprise ; I l'interaction croisée ; S l'avantage sur une suite adaptée à une autre corruption de même norme. La règle exige les quatre bornes inférieures ajustées au-dessus de 0,01 pour soutenir une récupération fonctionnelle utile et spécifique dans ce domaine. Une équivalence du dommage moyen n'est annoncée que si son IC ajusté entier tient dans [-0,01 ; 0,01]. Un IC incluant zéro ne prouve pas une équivalence.

Décision exécutée : **`mean_frozen_damage_equivalent_within_prespecified_margin`**. Le seuil est un choix exploratoire préfixé, sans validation indépendante d'utilité. La fraction de codes tardifs modifiés vaut **0.45 %** ; elle ne remplace pas la mesure d'utilité. Les résultats par signe sont descriptifs et ne changent pas la décision principale. Aucune classe analytique n'est assignée au générateur.

## 4. E — entrée contre état, puis contrôle invariant

Deux plans à 12 conditions par famille : A/B et A/Aplus, soit **384 sorties**. L'entrée réceptrice fournit seule le cache de préfixe ; l'état donneur est z9, sans cache donneur ni réalignement ajusté. Les deux suites natives sont figées séparément, ou la suite est recalculée depuis la combinaison entrée/état. Les diagonales reproduisent le natif ; avec état et suite figés, le préfixe receveur ne change pas la sortie du décodeur.

Sur A/B, la préférence moyenne `m=log p(S_B)-log p(S_A)` donne les effets suivants, sur toutes les familles :

- Effet d'entrée u : **29,467573 nat, IC 95 % [28,158560 ; 30,757245]**.
- Effet d'état v : **3,179795 nat, IC 95 % [2,181040 ; 4,194019]**.
- Interaction w : **-0,731137 nat, IC 95 % [-1,402055 ; 0,009171]**.
- Étendue native g : **32,647368 nat, IC 95 % [31,621394 ; 33,579859]**.

L'identité **u+v=g** vaut par construction algébrique : leurs ratios ne sont pas des fractions causales d'un raisonnement. Le diagnostic individuel exige deux grilles natives exactes, g>0,1 nat et |w/g|<=0,2. Les ratios d'entrée >=0,8 ou <=0,2 indiquent une domination descriptive ; les ratios intermédiaires restent « mélange ou mécanisme non modélisé ». Les interactions ou ratios non monotones ne sont pas forcés dans cette classification. Toutes les familles restent dans les effets globaux.

- `input_dominant_response_preference_in_this_test` : **15/16** familles.
- `non_monotone_non_identifiable` : **1/16** familles.

Une préférence dominée par l'entrée ne démontre pas un recalcul intégral depuis zéro ; une préférence dominée par l'état ne démontre pas l'absence d'accès aux indices. Ces conclusions portent sur la réponse A/B, ce point de coupure et ces interventions, sans localisation de circuit interne.

Sur le contrôle A/Aplus, la solution étant identique, aucune normalisation par une différence de réponse n'est appliquée :

- `native_plus_minus_A_nll` : -0,000044 nat, IC 95 % [-0,000130 ; 0,000002].
- `input_plus_state_A_minus_native_plus_nll` : 0,000002 nat, IC 95 % [-0,000003 ; 0,000008].
- `input_A_state_plus_minus_native_A_nll` : -0,000002 nat, IC 95 % [-0,000008 ; 0,000002].
- `input_plus_state_A_probability_TV` : 0,000008 (variation totale moyenne ; IC 95 % [0,000002 ; 0,000014]).
- `input_A_state_plus_probability_TV` : 0,000007 (variation totale moyenne ; IC 95 % [0,000002 ; 0,000013]).

Les changements de NLL sont en nats par case vide ; les distances TV sont sans unité. L'information supplémentaire peut améliorer ou modifier le calcul sans changer la référence. Il serait incorrect de compter tout changement comme une attribution fausse. L'ajout d'un indice conserve le facteur de confusion de cardinalité signalé au protocole.

## 5. Diagnostics par famille

Les valeurs u/v/w/g ci-dessous sont en nats, calculées sur les cases de solution différentes restées vides. Les libellés sont des diagnostics conditionnels, sans vérité mécaniste attribuée au réseau.

- `sudoku-000` : natif A/B exact=True/True, g=36,178981, u=32,731150, v=3,447831, w=-0,350106 nat ; `input_dominant_response_preference_in_this_test`.
- `sudoku-001` : natif A/B exact=True/True, g=30,282344, u=28,842938, v=1,439406, w=-1,006233 nat ; `input_dominant_response_preference_in_this_test`.
- `sudoku-002` : natif A/B exact=True/True, g=32,776440, u=32,234188, v=0,542252, w=-2,537108 nat ; `input_dominant_response_preference_in_this_test`.
- `sudoku-003` : natif A/B exact=True/True, g=33,501398, u=26,883798, v=6,617600, w=-1,126384 nat ; `input_dominant_response_preference_in_this_test`.
- `sudoku-004` : natif A/B exact=True/True, g=31,530698, u=26,904243, v=4,626455, w=-2,526007 nat ; `input_dominant_response_preference_in_this_test`.
- `sudoku-005` : natif A/B exact=True/True, g=32,723545, u=31,037070, v=1,686475, w=-1,853791 nat ; `input_dominant_response_preference_in_this_test`.
- `sudoku-006` : natif A/B exact=True/True, g=32,049211, u=26,334802, v=5,714409, w=-0,184039 nat ; `input_dominant_response_preference_in_this_test`.
- `sudoku-007` : natif A/B exact=True/True, g=32,478812, u=29,407370, v=3,071443, w=-1,217041 nat ; `input_dominant_response_preference_in_this_test`.
- `sudoku-008` : natif A/B exact=True/True, g=34,120904, u=28,411833, v=5,709071, w=-2,096490 nat ; `input_dominant_response_preference_in_this_test`.
- `sudoku-009` : natif A/B exact=True/True, g=27,133768, u=23,847618, v=3,286150, w=-0,044056 nat ; `input_dominant_response_preference_in_this_test`.
- `sudoku-010` : natif A/B exact=True/True, g=31,645948, u=30,413894, v=1,232055, w=-1,191952 nat ; `input_dominant_response_preference_in_this_test`.
- `sudoku-011` : natif A/B exact=True/True, g=32,864112, u=29,734381, v=3,129731, w=-1,997573 nat ; `input_dominant_response_preference_in_this_test`.
- `sudoku-012` : natif A/B exact=True/True, g=34,395910, u=34,636506, v=-0,240596, w=-1,108654 nat ; `non_monotone_non_identifiable`.
- `sudoku-013` : natif A/B exact=True/True, g=32,595144, u=28,177804, v=4,417340, w=1,659906 nat ; `input_dominant_response_preference_in_this_test`.
- `sudoku-014` : natif A/B exact=True/True, g=35,457383, u=29,834521, v=5,622862, w=2,271699 nat ; `input_dominant_response_preference_in_this_test`.
- `sudoku-015` : natif A/B exact=True/True, g=32,623292, u=32,049057, v=0,574235, w=1,609635 nat ; `input_dominant_response_preference_in_this_test`.

## 6. Décalage de distribution et validité

Les coordonnées perturbées sortant de l'enveloppe min/max coordonnée par coordonnée des 48 états natifs z9 représentent en moyenne **0.2484 %**, maximum **0.4213 %**. Cette mesure d'amplitude n'est ni une densité estimée ni une preuve que les autres coordonnées seraient sur distribution. Les états natifs échangés peuvent eux-mêmes être hors distribution conditionnellement à leur nouveau préfixe.

Une absence de récupération confirmée peut refléter la dose, le point de coupure, la précision statistique ou une continuation peu utile. Une sensibilité aux deux sources peut refléter interactions, redondance ou mécanisme non modélisé ; elle ne suffit pas à prouver un mélange structurel. Seize familles et un checkpoint n'autorisent aucune généralisation architecturale. Les IC bootstrap à si petit effectif restent approximatifs et ne prennent pas en compte un réentraînement.

## 7. Audit, artefacts et arrêt

Les 48 entrées passent la parité native bit à bit, y compris les 12 accumulations, et la reprise sans perturbation. L'audit reconstruit **1024 lectures** à partir des contributions sauvegardées et rejoue **32 suites neurales** des deux premières familles avec une boucle explicite distincte. Il recalcule NLL et préférences indépendamment, les intervalles C, les diagnostics E, et les comptes de transitions. Aucun de ces contrôles logiciels ne représente une réplication scientifique indépendante ; ils sont écrits et exécutés par le même agent.

Les empreintes des entrées, poids et sources gelés restent inchangées. Durée du runner : **48.3 s**, contrôles et sauvegardes compris ; ce n'est pas un benchmark de vitesse.

- [Protocole](PROTOCOLE.md), [gel](results/protocol_frozen.json), [journal](JOURNAL.md).
- [Résumé et diagnostics](results/summary.json), [audit](results/verification.json).
- `results/native.npz` : codes, états, résidus et logits des 12 niveaux pour les 48 entrées.
- `results/crossed.npz` : deltas, suites et logits C ; `results/exchanges.npz` : les 24 conditions E par famille.
- [Reproduction](README.md). Les archives Numpy se chargent sans pickle.

**Arrêt après l'étape 3a.** Aucun changement du manuscrit ou des anciennes campagnes ; aucune expérience Wyner, aucun changement opportuniste de seuil ou de dose et aucun reclassement certain du NCM.

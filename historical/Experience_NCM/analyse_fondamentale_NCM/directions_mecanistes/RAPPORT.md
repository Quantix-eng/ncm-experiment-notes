# NCM : niveau 5 peu diversifié et révisions au niveau 9

Rapport de la seconde campagne — 24 octobre 2026. Analyse exploratoire des checkpoints Sudoku publiés, principalement hard. Le [rapport précédent](../RAPPORT.md) et ses résultats restent inchangés.

## Résultats principaux

**Niveau 5 : une contrainte structurelle de représentation est identifiée.** À cette échelle, le pooling du tokenizer regroupe chaque ligne complète du Sudoku. Toutes les solutions ont les mêmes histogrammes de chiffres par ligne ; avant GroupNorm et soustraction des contributions précédentes, les moyennes d'embeddings sont donc invariantes, à l'arrondi près. Cela explique un déficit de variation lié au couple données/architecture, sans démontrer à lui seul la cause historique du codebook presque constant.

**Une prédiction peut être évitée sur les données testées.** Forcer le code VQ 11 au niveau 5 hard, tout en conservant sa contribution, reproduit les codes et les réponses natives sur les 600 puzzles test et les 256 externes : 11 passes de prédiction des niveaux au lieu de 12. Il s'agit d'une équivalence sur ces corpus ; aucun gain de vitesse universel n'est revendiqué.

**La valeur de remplacement est partiellement prévisible depuis les sorties observables.** La sonde conditionnelle combinée atteint 55.97 %, contre 55.65 % pour la baseline observable ajustée. Gain 0.33 points, IC 95 % [-0.10 %, 0.76 %]. Sur le corpus externe : gain -0.07 points, IC [-0.66 %, 0.51 %]. L'ajout des états internes n'apporte donc pas de gain établi dans ce protocole conditionnel.

**Les interventions ne démontrent pas une direction de révision spécifique utilisée par le générateur.** Les contrastes principaux via les codes 9 recouvrent zéro dans les deux séries. Les différences de lecture directe ne doivent pas être prises pour un mécanisme causal de génération.

## 1. Objectifs, protocole et portée

L'étude suit [OBJECTIFS_ET_PROTOCOLE.md](OBJECTIFS_ET_PROTOCOLE.md), les paramètres [protocol.json](protocol.json) et le gel [results/protocol_frozen.json](results/protocol_frozen.json). Les amendements [conditionnel](AMENDEMENT_CONDITIONNEL.md) et [dropout](AMENDEMENT_DROPOUT.md) sont explicitement post-hoc ; ils ne remplacent pas les résultats v1. Le [journal](JOURNAL.md) indique leur motivation et l'ordre des analyses.

Les partitions de la campagne précédente sont conservées : 1 000 puzzles train, 400 validation, 600 test, groupés par familles de solutions. Les trois difficultés partagent les mêmes familles ; leurs diagnostics ne sont pas trois réplications indépendantes des données. Le corpus externe contient 256 puzzles uniques de la campagne précédente et ne sert à aucun ajustement. Il avait déjà été examiné pour d'autres questions ; après consultation de v1, il ne constitue plus une confirmation intacte pour les compléments.

Les poids NCM sont gelés, CPU FP32, un thread par processus, batch 16, graine 73129. Les intervalles prédictifs rééchantillonnent les puzzles et les comptes de cellules par puzzle, jamais des cellules supposées indépendantes. Les intervalles causaux sont appariés par puzzle (2 000 tirages). Les sondes ne sont pas réajustées dans le bootstrap ; les IC sont conditionnels à ce découpage et à ces ajustements. Les comparaisons exploratoires multiples ne font pas l'objet d'une correction de multiplicité.

Le runner explicite conserve les masques, le cache de préfixe, le calcul des logits et l'accumulation du port CPU. Avant toute interprétation, il reproduit bit à bit les codes, les logits finaux et les états z8/z9 sur 32 puzzles, deux lots distincts. À l'extraction, les codes et décodages 8/9 sont comparés aux traces précédentes sur la totalité des 2 256 puzzles hard et externes.

## 2. Ce que signifie ici un niveau « mort »

Le niveau 5 hard encode et génère un seul identifiant VQ sur les solutions/puzzles officiels examinés. Cette constance ne signifie ni activations neuronales nulles, ni contribution exactement nulle, ni inutilité dans tout régime d'entrée. Les codebooks easy et medium sont également peu diversifiés à ce niveau, avec une diversité différente de hard.

Le diagnostic utilise les 256 premières familles de train et 256 de test, sans entraîner NCM : variances à position fixée puis moyennées sur canaux/positions, distances aux codes, marges du plus proche voisin, MSE des résidus et statistiques EMA sauvegardées. L'accumulation explicite doit donner les mêmes codes que l'encodeur. Un calcul direct des distances en double précision est comparé au calcul rapide sur 256 vecteurs par niveau et checkpoint.

- **hard** : 1/16 codes observés au niveau 5 sur l'échantillon train ; variance inter-puzzles de son entrée 0.00182832, soit 0.19 % de la variance du résidu pleine résolution ; 0 désaccord de sélection entre distances rapides et directes sur les échantillons vérifiés. EMA positives pour 16/16 codes.

- **easy** : 2/16 codes observés au niveau 5 sur l'échantillon train ; variance inter-puzzles de son entrée 0.00175288, soit 0.18 % de la variance du résidu pleine résolution ; 0 désaccord de sélection entre distances rapides et directes sur les échantillons vérifiés. EMA positives pour 16/16 codes.

- **medium** : 2/16 codes observés au niveau 5 sur l'échantillon train ; variance inter-puzzles de son entrée 0.0019845, soit 0.21 % de la variance du résidu pleine résolution ; 0 désaccord de sélection entre distances rapides et directes sur les échantillons vérifiés. EMA positives pour 16/16 codes.

Les EMA sont des moyennes mobiles sauvegardées, pas des comptages de ce corpus ni un journal temporel. Leur positivité est incompatible avec l'interprétation simpliste « ces codes n'ont jamais été utilisés », mais elle ne permet pas de reconstruire quand et pourquoi ils l'ont été. Les tests de distances ne révèlent pas de bug de sélection dans les cas vérifiés ; un problème d'entraînement original reste une possibilité non démontrée.

## 3. Pourquoi neuf positions constituent une échelle particulière

Les solutions sont sérialisées en 89 tokens : neuf chiffres par ligne et huit séparateurs. Le pooling adaptatif à neuf positions utilise les fenêtres floor(j × 89/9) : ceil((j+1) × 89/9). Chacune contient les neuf chiffres d'une ligne, plus un séparateur aux extrémités ou deux pour les lignes intérieures.

L'encodeur configuré ici est une somme d'embeddings de tokens et de positions, sans blocs contextuels. Chaque ligne contient une fois chacun des chiffres 1 à 9 ; les embeddings positionnels sont fixes. La moyenne d'embeddings de chaque fenêtre avant normalisation est ainsi indépendante de l'ordre des chiffres dans la ligne. Les histogrammes de tokens sont exactement identiques entre les 256 solutions vérifiées ; la propriété découle également de leur validité Sudoku.

- **hard**, variance des embeddings moyennés avant GroupNorm : pooling 8 = 0.046544, pooling 9 = 3.3022e-16, pooling 10 = 0.059304. Après GroupNorm, pooling 9 = 1.582e-08.

- **easy**, variance des embeddings moyennés avant GroupNorm : pooling 8 = 0.044323, pooling 9 = 2.9221e-16, pooling 10 = 0.056378. Après GroupNorm, pooling 9 = 1.7011e-08.

- **medium**, variance des embeddings moyennés avant GroupNorm : pooling 8 = 0.046428, pooling 9 = 3.0444e-16, pooling 10 = 0.05916. Après GroupNorm, pooling 9 = 1.2513e-08.

![Pooling et variance](figures/01_pooling.png)

La GroupNorm peut réintroduire une petite variation. Surtout, le niveau 5 quantifie un résidu après les quatre contributions précédentes : l'invariance des embeddings d'origine n'impose donc pas mathématiquement un code résiduel unique. Le constat établit un mécanisme d'atténuation de la variation, pas un théorème de collapse du codebook ni une propriété générale du coarse-to-fine.

Les contrôles hard conservent les mêmes 64 solutions sources. La permutation des chiffres est une symétrie valide ; les permutations internes à chaque ligne, le mélange global des cases et les chiffres i.i.d. sont des entrées diagnostiques généralement invalides, et ne servent pas à mesurer une performance Sudoku.

- **original** : 1/16 codes utilisés ; variance de l'entrée résiduelle du niveau 5 0.00177717.

- **valid_digit_permutation** : 1/16 codes utilisés ; variance de l'entrée résiduelle du niveau 5 0.00181293.

- **within_rows_invalid** : 1/16 codes utilisés ; variance de l'entrée résiduelle du niveau 5 0.00187726.

- **global_shuffle_invalid** : 15/16 codes utilisés ; variance de l'entrée résiduelle du niveau 5 0.0680416.

- **iid_digits_invalid** : 16/16 codes utilisés ; variance de l'entrée résiduelle du niveau 5 0.0675562.

![Contrôles du niveau 5](figures/02_controls.png)

Le maintien d'un seul code après permutation interne aux lignes et l'activation de nombreux codes après mélange global soutiennent une sensibilité à la structure des moyennes par ligne. Le codebook n'est pas incapable d'activer d'autres entrées ; celles-ci peuvent toutefois correspondre à des entrées hors distribution.

## 4. Quantification restante et régime de dropout

Un k-means à 16 centres est ajusté aux seules entrées résiduelles du niveau 5 des 256 solutions de train. Les distances sont évaluées sur les 256 familles test. Le décodeur et le générateur restent inchangés ; il ne s'agit pas de remplacer le tokenizer puis de revendiquer une amélioration de résolution.

- **hard** : MSE test du codebook original 0.00259667, k-means 0.0011727, réduction relative 54.84 %, 16/16 centres utilisés sur test.

- **easy** : MSE test du codebook original 0.00258573, k-means 0.00112428, réduction relative 56.52 %, 16/16 centres utilisés sur test.

- **medium** : MSE test du codebook original 0.00273385, k-means 0.00124356, réduction relative 54.51 %, 16/16 centres utilisés sur test.

Il reste donc de la variation géométriquement exploitable après pooling, malgré la faible diversité du codebook original en évaluation. Sa meilleure quantification n'est pas une preuve d'information utile au Sudoku : elle peut notamment refléter les contributions résiduelles précédentes.

Le contrôle de dropout intervient uniquement sur encoder.drop, avec sa probabilité configurée. Les autres modules restent en évaluation et l'encodeur quantifie par le chemin sans mise à jour EMA. Quatre graines sont appliquées aux mêmes 64 solutions de train ; les state_dict sont comparés avant/après. Ce contrôle teste un régime d'activation sur poids gelés, sans reproduire l'histoire d'entraînement.

- **hard**, dropout p=0.30 : évaluation 1/16 codes, perplexité 1.000 ; dropout actif, codes cumulés 14/16, perplexité moyenne par répétition 1.125 (min–max 1.091–1.161). Poids et EMA identiques : True.

- **easy**, dropout p=0.30 : évaluation 2/16 codes, perplexité 1.717 ; dropout actif, codes cumulés 15/16, perplexité moyenne par répétition 2.419 (min–max 2.313–2.533). Poids et EMA identiques : True.

- **medium**, dropout p=0.30 : évaluation 1/16 codes, perplexité 1.000 ; dropout actif, codes cumulés 14/16, perplexité moyenne par répétition 1.291 (min–max 1.275–1.307). Poids et EMA identiques : True.

![Dropout et diversité](figures/06_dropout.png)

L'interprétation doit distinguer invariance des entrées propres, usage du codebook sous bruit et causalité historique de l'optimisation. Le dropout augmente effectivement la diversité sur les trois checkpoints, tout en conservant une forte dominance des codes natifs. La désactivation du dropout restaure les régularités des moyennes de lignes ; son activation les perturbe. Cela ne prouve pas que le dropout est l'unique cause de l'apprentissage observé. Les trajectoires et graines d'entraînement originales restent indisponibles.

## 5. Contribution du niveau 5 et prédiction évitable

Deux interventions sont comparées aux prédictions natives sauvegardées : (a) produire le code 11 déterminé sur train, sans exécuter la prédiction Transformer du niveau 5, puis conserver sa contribution ; (b) effectuer la prédiction mais annuler la contribution résiduelle du niveau 5, puis régénérer les niveaux suivants. Les positions et les autres niveaux conservent leur configuration.

- **official_test, fixed_code11_keep_residual** : 516/600 grilles exactes ; 0 grilles et 0 codes différents du natif ; 0 erreurs introduites, 0 corrigées ; variation de réussite 0.00 points, IC [0.00 %, 0.00 %] ; [11] prédictions de niveaux par batch. Borne Wilson supérieure du taux de changement de statut succès/échec : 0.64 %.

- **official_test, zero_residual_regenerate** : 516/600 grilles exactes ; 13 grilles et 364 codes différents du natif ; 0 erreurs introduites, 0 corrigées ; variation de réussite 0.00 points, IC [0.00 %, 0.00 %] ; [12] prédictions de niveaux par batch. Borne Wilson supérieure du taux de changement de statut succès/échec : 0.64 %.

- **external, fixed_code11_keep_residual** : 219/256 grilles exactes ; 0 grilles et 0 codes différents du natif ; 0 erreurs introduites, 0 corrigées ; variation de réussite 0.00 points, IC [0.00 %, 0.00 %] ; [11] prédictions de niveaux par batch. Borne Wilson supérieure du taux de changement de statut succès/échec : 1.48 %.

- **external, zero_residual_regenerate** : 219/256 grilles exactes ; 3 grilles et 129 codes différents du natif ; 0 erreurs introduites, 0 corrigées ; variation de réussite 0.00 points, IC [0.00 %, 0.00 %] ; [12] prédictions de niveaux par batch. Borne Wilson supérieure du taux de changement de statut succès/échec : 1.48 %.

Le remplacement par le code fixe est exactement équivalent sur les corpus testés. La suppression du résidu modifie certaines réponses tout en laissant les succès/échecs inchangés. L'IC bootstrap [0,0] dans ce cas est dégénéré : il ne démontre ni absence d'effet populationnel ni équivalence à toute précision. Les bornes Wilson donnent un repère sur les discordances non observées.

Onze passes au lieu de douze ne représentent pas automatiquement 8,3 % de temps gagné : les niveaux ont des tailles différentes et le préfixe, le cache et le décodeur ont un coût. Les durées brutes sont sauvegardées à titre descriptif, avec conditions non randomisées et calculs parfois concurrents ; aucun benchmark de vitesse n'est revendiqué.

## 6. Prévoir la révision sans lire le futur

Les observations sont l'état cumulatif z8 (89 positions × 256 canaux), le résumé de la dernière couche au niveau 8 (1 024 valeurs), les probabilités du décodeur au niveau 8, les indices et des statistiques observables. La projection gaussienne des résumés globaux vers 64 dimensions est fixée par la graine. Les positions de z8 correspondent aux tokens ; les cases sont repérées par positions fixes, séparateurs exclus. Les 26 positions de génération du niveau 8 ne sont pas assimilées directement à 81 cellules.

La baseline contient 71 valeurs locales/globales : 12 probabilités, 12 indicatrices du token actuel, 9 candidats autorisés par les indices, 18 indicatrices ligne/colonne, 20 statistiques globales sans référence. La combinaison y ajoute 256 canaux de z8 local et 64 projections globales. Les états seuls incluent aussi les coordonnées. Le contrôle permute les états de train entre puzzles, en conservant les sorties observables et les cibles initiales.

Les sondes sont des régressions ridge multi-sorties, standardisées sur train seulement. Alpha dans {10,100,1000} et température softmax dans {0.1,0.25,0.5,1} sont choisis par log-loss validation. Les cibles sont (i) le token prédit au niveau 9, avec 12 classes pour conserver les tokens spéciaux éventuels ; (ii) inchangé, corrigé, dégradé ou erreur remplacée par une autre erreur. Les références ne servent qu'à cette seconde cible et aux métriques. Aucune observation du niveau 9 n'entre dans les prédicteurs.

Les métriques portent sur les cases initialement vides. Le sous-groupe des cases effectivement changées est défini avec le futur pour l'évaluation ; il ne constitue jamais un oracle disponible à la sonde.

## 7. Résultat v1 : gain réel contre une baseline apprise, mais objectif dominé par la persistance

Le pic initial correspond précisément à **41,620 cases incorrectes devenant correctes** entre les niveaux 8 et 9 sur les 2 000 puzzles hard (toutes les cases, indices compris). Sur le même passage, 14,770 cases correctes deviennent incorrectes et 17,993 erreurs changent de valeur sans devenir correctes. Ces comptes décrivent les décodages du tokenizer ; ils ne localisent pas un instant où le modèle déciderait consciemment ni ne démontrent une révision explicite de raisonnement. Les évaluations de sondes ci-dessous concernent uniquement les cases initialement vides du test réservé.

- **baseline** : exactitude sur toutes les cases vides 54.78 %, sur les cases effectivement changées 2.98 %, log-loss globale 1.4623.

- **state** : exactitude sur toutes les cases vides 34.60 %, sur les cases effectivement changées 16.36 %, log-loss globale 1.8843.

- **combined** : exactitude sur toutes les cases vides 55.22 %, sur les cases effectivement changées 4.50 %, log-loss globale 1.4566.

- **shuffled_states** : exactitude sur toutes les cases vides 54.95 %, sur les cases effectivement changées 3.59 %, log-loss globale 1.4652.

Le contraste principal v1 combinaison–baseline sur changements vaut 1.52 points, IC [1.26 %, 1.79 %]. Le deuxième choix du décodeur 8 prédit pourtant 49.40 % des remplacements, bien au-delà des sondes entraînées sur toutes les cases. Le gain v1 ne suffit donc pas à établir une anticipation substantielle des remplacements. Les résultats initiaux sont conservés, sans les remplacer par le complément suivant.

- **Qualité du changement, baseline** : F1 macro 0.1847, AUROC « case corrigée » 0.5804, log-loss 1.1565.

- **Qualité du changement, state** : F1 macro 0.1949, AUROC « case corrigée » 0.5917, log-loss 1.1389.

- **Qualité du changement, combined** : F1 macro 0.2136, AUROC « case corrigée » 0.6139, log-loss 1.1411.

- **Qualité du changement, shuffled_states** : F1 macro 0.2105, AUROC « case corrigée » 0.6055, log-loss 1.1456.

La classification de la qualité reste limitée et ne démontre pas une direction précise de correction. La F1 macro du prochain token est calculée sur les 12 classes, y compris celles absentes des cibles, et doit être interprétée avec cette convention.

![Prédiction du prochain token](figures/03_next_token.png)

## 8. Complément conditionnel : remplacement et porte de changement

Après v1, un amendement exploratoire ajoute six statistiques locales accessibles au niveau 8 : entropie, maximum, deuxième probabilité, marge, probabilité du token actuel et nombre de candidats normalisé. La sonde de remplacement est ajustée uniquement sur les changements de train, et sélectionnée sur les changements de validation. Le token actuel est exclu de la distribution alternative. Une porte binaire prédit séparément le changement sur toutes les cases de train ; sa probabilité combine persistance et distribution alternative pour produire une prévision jointe sans oracle.

- **baseline**, test officiel : remplacement conditionnel correct 55.65 %, prévision jointe toutes cases 55.34 %, AUROC de la porte 0.6851.

- **state**, test officiel : remplacement conditionnel correct 25.80 %, prévision jointe toutes cases 53.88 %, AUROC de la porte 0.5886.

- **combined**, test officiel : remplacement conditionnel correct 55.97 %, prévision jointe toutes cases 55.10 %, AUROC de la porte 0.6862.

- **shuffled_states**, test officiel : remplacement conditionnel correct 55.27 %, prévision jointe toutes cases 54.95 %, AUROC de la porte 0.6842.

- **baseline**, externe : remplacement conditionnel correct 56.44 %, prévision jointe toutes cases 55.59 %, AUROC de la porte 0.6857.

- **state**, externe : remplacement conditionnel correct 25.30 %, prévision jointe toutes cases 53.79 %, AUROC de la porte 0.5913.

- **combined**, externe : remplacement conditionnel correct 56.38 %, prévision jointe toutes cases 54.99 %, AUROC de la porte 0.6879.

- **shuffled_states**, externe : remplacement conditionnel correct 55.91 %, prévision jointe toutes cases 55.28 %, AUROC de la porte 0.6884.

**Officiel** : combinaison–baseline ajustée 0.33 points, IC [-0.10 %, 0.76 %] ; combinaison–deuxième choix brut 6.58 points, IC [5.83 %, 7.33 %].

**Externe** : combinaison–baseline ajustée -0.07 points, IC [-0.66 %, 0.51 %] ; combinaison–deuxième choix brut 6.15 points, IC [5.04 %, 7.21 %].

![Évaluation externe](figures/04_external.png)

La valeur de remplacement est partiellement prévisible et une baseline observable apprise dépasse l'heuristique du deuxième choix brut. En revanche, l'ajout de z8 et des résumés internes n'apporte pas de gain établi face à cette baseline. L'exactitude jointe et l'AUROC de la porte rappellent que connaître le bon remplacement conditionnellement au changement ne revient pas à savoir quelle case changera.

Il s'agit d'un complément décidé après observation de v1. La baseline a aussi été enrichie : la différence entre les scores v1 et conditionnels ne peut donc pas être attribuée au seul changement du masque d'entraînement. L'évaluation externe n'a servi à aucun ajustement, mais elle avait déjà été consultée et n'est pas une nouvelle confirmation demeurée intacte.

## 9. Intervention locale avant le niveau 9

Sur les 96 premiers puzzles du test fixé, une case initialement vide et un chiffre alternatif sont choisis depuis les seuls prédicteurs pré-9 et la sonde gelée. Pour v1, le score est probabilité du chiffre alternatif moins celle du chiffre actuel. Pour la série conditionnelle, le score est probabilité prédite de changement × probabilité conditionnelle du chiffre alternatif. Ni référence ni transition réelle ne participent à ce choix.

La direction est la différence des poids ridge du chiffre demandé et du chiffre actuel, restreinte aux 256 canaux z8 locaux et ramenée aux coordonnées non standardisées. Il s'agit d'une direction partielle de la sonde : les autres variables, notamment les probabilités de lecture, sont tenues fixes dans cette construction. La perturbation ajoutée a une norme égale à 0,25 puis 0,5 fois la norme de z8 à la case ciblée ; le complément conditionnel utilise uniquement 0,25. Les directions négatives et trois directions aléatoires par puzzle ont exactement la même norme. Le reste de la génération est poursuivi normalement.

Pour séparer les chemins, trois lectures du niveau 9 sont comparées :

1. **Complète** : décodeur(z8 + delta + contribution des nouveaux codes 9).
2. **Codes seuls** : décodeur(z8 + contribution des nouveaux codes 9), sans delta au décodeur.
3. **Directe** : décodeur(z8 + delta + contribution des codes 9 natifs), sans régénération de ces codes.

La deuxième lecture est reconstruite explicitement pour éviter les erreurs d'arrondi d'une soustraction de delta. Elle est un contrôle d'intervention, pas l'identification d'une médiation naturelle unique dans un réseau non linéaire. Les niveaux suivants de la vraie branche conservent la perturbation.

**Direction v1** : contraste principal des log-rapports demandé/actuel via les codes seuls, positif–moyenne des aléatoires = 0.00394 nat, IC [-0.0527, 0.0600]. Positif–négatif = 0.02207, IC [-0.0727, 0.1349].

- sham : adoption demandée complète 55.21 %, codes seuls 55.21 %, directe 55.21 % ; 0 codes 9 modifiés sur 3 552 ; 0.000 autres cases changées en moyenne ; réussite finale 82/96, erreurs introduites 0, corrigées 0.

- 0.25_positive : adoption demandée complète 58.33 %, codes seuls 56.25 %, directe 57.29 % ; 6 codes 9 modifiés sur 3 552 ; 0.583 autres cases changées en moyenne ; réussite finale 82/96, erreurs introduites 0, corrigées 0.

- 0.25_negative : adoption demandée complète 50.00 %, codes seuls 55.21 %, directe 50.00 % ; 6 codes 9 modifiés sur 3 552 ; 0.594 autres cases changées en moyenne ; réussite finale 82/96, erreurs introduites 0, corrigées 0.

- 0.25_random0 : adoption demandée complète 57.29 %, codes seuls 55.21 %, directe 56.25 % ; 4 codes 9 modifiés sur 3 552 ; 0.562 autres cases changées en moyenne ; réussite finale 82/96, erreurs introduites 0, corrigées 0.

- 0.25_random1 : adoption demandée complète 46.88 %, codes seuls 55.21 %, directe 46.88 % ; 4 codes 9 modifiés sur 3 552 ; 0.573 autres cases changées en moyenne ; réussite finale 82/96, erreurs introduites 0, corrigées 0.

- 0.25_random2 : adoption demandée complète 51.04 %, codes seuls 55.21 %, directe 51.04 % ; 7 codes 9 modifiés sur 3 552 ; 0.562 autres cases changées en moyenne ; réussite finale 82/96, erreurs introduites 0, corrigées 0.

- 0.5_positive : adoption demandée complète 43.75 %, codes seuls 56.25 %, directe 43.75 % ; 13 codes 9 modifiés sur 3 552 ; 0.979 autres cases changées en moyenne ; réussite finale 82/96, erreurs introduites 0, corrigées 0.

- 0.5_negative : adoption demandée complète 42.71 %, codes seuls 55.21 %, directe 41.67 % ; 13 codes 9 modifiés sur 3 552 ; 0.906 autres cases changées en moyenne ; réussite finale 82/96, erreurs introduites 0, corrigées 0.

- 0.5_random0 : adoption demandée complète 47.92 %, codes seuls 55.21 %, directe 48.96 % ; 14 codes 9 modifiés sur 3 552 ; 1.073 autres cases changées en moyenne ; réussite finale 82/96, erreurs introduites 0, corrigées 0.

- 0.5_random1 : adoption demandée complète 39.58 %, codes seuls 55.21 %, directe 39.58 % ; 9 codes 9 modifiés sur 3 552 ; 1.010 autres cases changées en moyenne ; réussite finale 82/96, erreurs introduites 0, corrigées 0.

- 0.5_random2 : adoption demandée complète 38.54 %, codes seuls 55.21 %, directe 38.54 % ; 13 codes 9 modifiés sur 3 552 ; 1.021 autres cases changées en moyenne ; réussite finale 82/96, erreurs introduites 0, corrigées 0.

**Direction conditionnelle** : contraste principal des log-rapports demandé/actuel via les codes seuls, positif–moyenne des aléatoires = 0.02433 nat, IC [-0.0298, 0.0901]. Positif–négatif = -0.00113, IC [-0.0073, 0.0053].

- sham : adoption demandée complète 44.79 %, codes seuls 44.79 %, directe 44.79 % ; 0 codes 9 modifiés sur 3 552 ; 0.000 autres cases changées en moyenne ; réussite finale 82/96, erreurs introduites 0, corrigées 0.

- 0.25_positive : adoption demandée complète 42.71 %, codes seuls 43.75 %, directe 42.71 % ; 4 codes 9 modifiés sur 3 552 ; 0.458 autres cases changées en moyenne ; réussite finale 82/96, erreurs introduites 0, corrigées 0.

- 0.25_negative : adoption demandée complète 41.67 %, codes seuls 44.79 %, directe 41.67 % ; 6 codes 9 modifiés sur 3 552 ; 0.583 autres cases changées en moyenne ; réussite finale 82/96, erreurs introduites 0, corrigées 0.

- 0.25_random0 : adoption demandée complète 40.62 %, codes seuls 44.79 %, directe 39.58 % ; 6 codes 9 modifiés sur 3 552 ; 0.552 autres cases changées en moyenne ; réussite finale 82/96, erreurs introduites 0, corrigées 0.

- 0.25_random1 : adoption demandée complète 41.67 %, codes seuls 43.75 %, directe 41.67 % ; 6 codes 9 modifiés sur 3 552 ; 0.573 autres cases changées en moyenne ; réussite finale 82/96, erreurs introduites 0, corrigées 0.

- 0.25_random2 : adoption demandée complète 42.71 %, codes seuls 44.79 %, directe 42.71 % ; 8 codes 9 modifiés sur 3 552 ; 0.552 autres cases changées en moyenne ; réussite finale 82/96, erreurs introduites 0, corrigées 0.

![Contrôles d'intervention](figures/05_steering.png)

Les intervalles des contrastes principaux recouvrent zéro. Dans la série conditionnelle, l'effet ponctuel de la direction positive sur le log-rapport complet est très proche de celui de la lecture directe, tandis que le gain via les nouveaux codes seuls reste proche de zéro. Cela ne fournit pas de preuve d'une direction de révision spécifique exploitée par le générateur. L'adoption d'un chiffre et l'amplitude de son log-rapport ne sont pas des métriques identiques : l'une est discrète, l'autre peut varier sans franchir l'argmax.

Le nombre de changements des codes 9 est faible ; les quantificateurs peuvent être insensibles aux petites perturbations. Les succès finaux restent inchangés sur ces 96 puzzles, mais cela ne démontre ni absence de rôle causal de toute représentation de révision ni équivalence universelle. Les effets peuvent être distribués, non linéaires ou mal alignés avec la direction linéaire choisie. La dose 0,5 et les comparaisons secondaires ne constituent pas des recherches corrigées d'une dose optimale.

## 10. Ce que nous avons établi et ce qui reste ouvert

- Une invariance des moyennes d'embeddings à neuf positions est expliquée par la sérialisation des lignes complètes de Sudoku. Elle se retrouve avec les trois tokenizers examinés et ne constitue pas une propriété universelle de NCM.
- Le résidu avant quantification du niveau 5 reste variable, et un k-means local en exploite davantage. Ce gain géométrique n'établit pas un gain de résolution après modification des codebooks.
- L'usage du niveau doit être distingué entre entrées propres et régime de dropout, sans assimiler EMA positive et fréquence actuelle ni en déduire une trajectoire d'apprentissage.
- Sur les 856 puzzles hard testés, la prédiction du niveau 5 peut être remplacée par un code constant en conservant exactement les réponses. Cela n'autorise pas à supprimer sans effet sa contribution ni à annoncer un gain de vitesse proportionnel au nombre de niveaux.
- Les probabilités déjà lisibles au niveau 8 prédisent une partie des valeurs de remplacement au niveau 9. Le bénéfice supplémentaire des états examinés n'est pas établi contre la baseline ajustée.
- Aucune direction causale spécifique de révision ni contrôle métacognitif natif n'est démontré. Les résultats négatifs portent sur les sondes, positions, doses et contrôles effectivement testés.

Pour expliquer historiquement pourquoi les codebooks ont convergé ainsi, il manque des checkpoints d'entraînement, les logs et une réplication fidèle de la recette complète. Pour renforcer l'étude des révisions, il faudrait des données réellement nouvelles préservées avant choix de la méthode, davantage de graines, des lectures plus structurées ou non linéaires et des interventions localisant le chemin de génération. Aucun petit entraînement substitutif ni benchmark GPU n'a été réalisé ici.

## 11. Reproduction et preuves

Depuis ce sous-dossier, reproduction complète (longue) :

```sh
./run.sh
```

Vérification des artefacts sauvegardés :

```sh
PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" ../.venv/bin/python scripts/verify.py
```

L'environnement Python et les poids sont ceux de la campagne précédente ; [requirements.txt](../requirements.txt) et [verrou complet](../sources/requirements-lock.txt) documentent les versions. La reproduction suppose de conserver l'arborescence avec ../artifacts, ../data, ../vendor, ../scripts et ../results. Les caches sont confinés dans ce sous-dossier. Les calculs concurrents initiaux ne servent pas à mesurer la vitesse.

**Sources** : scripts/base.py ; runtime.py ; diagnose_level5.py ; pooling_algebra.py ; dropout_control.py ; archive_centers.py ; level5_interventions.py ; extract_revision.py ; probes.py ; conditional.py ; steer_revision.py ; report.py/reporting.py ; verify.py ; continue.py ; run.sh. Le runner d'origine importé est ../vendor/ncm_mac et les utilitaires communs sont ../scripts/common.py. Tous les scripts de cette campagne et les dépendances de modèle copiées sont inclus dans sources_directions_mecanistes.zip ; les gros poids et traces restent dans leurs dossiers.

**Artefacts inspectables** : results/level5_diagnostic.json et level5_geometry.npz ; pooling_algebra.json ; dropout_control.json/.npz ; kmeans_centers.npz ; level5_interventions.json/.npz ; revision_official.npz et revision_external.npz ; modèles et prédictions de probes et conditional ; steering.json/.npz et steering_conditional.json/.npz. Les CSV d'inspection et les figures PNG/SVG sont générés avec le rapport. Les références utilisées comme cibles sont conservées séparément des variables pré-9.

**Traçabilité** : results/protocol_frozen.json ; final_manifest.json et verification.json ; JOURNAL.md et logs/. L'assemblage automatique initial a précédé la création du script de rendu et a échoué ; le rendu final est exécuté après tous les compléments, sans perdre ni réajuster les résultats v1.

Référence scientifique : [article NCM](https://arxiv.org/html/2610.14162v1), §3 pour le tokenizer résiduel et le générateur ; les résultats et contrôles rapportés ici sont calculés localement. Les observations de cette étude concernent les poids publiés effectivement copiés, pas toutes les variantes NCM.

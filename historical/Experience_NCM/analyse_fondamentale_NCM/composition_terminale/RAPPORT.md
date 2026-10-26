# Composition de deux écritures terminales — rapport du 25 octobre 2026

Le pilote franchit au moins un seuil local de dégradation de la composition : la règle testée ne justifie pas d'investissement architectural supplémentaire. **Ce résultat ne démontre ni une interface symbolique fiable pour des demandes arbitraires, ni un avantage architectural.** Les échecs absolus et ceux des éditions isolées sont conservés ci-dessous.

## Expérience et statut

32 puzzles à 30 indices, indices 512 à 543 du corpus historique `recherche_positive/data/fresh.json` ; ces sources ne sont pas celles des 128 puzzles de l'ancien test causal. Ce corpus a déjà servi à analyser les sorties natives : il ne s'agit pas d'un nouvel échantillon réservé. Un seul checkpoint hard, CPU FP32, un thread ; aucun apprentissage, solveur, recherche de code par exemple ou changement de table.

Le gel local date de **2026-10-25T12:35:27Z**. Il inclut le protocole, les scripts d'expérience et de vérification, les données, les tables, les poids et les sources importées. Le journal ne constitue pas une prérégistration indépendante. Les empreintes sont inchangées avant et après les tests. Les états natifs régénérés correspondent bit à bit aux codes et latents des 32 sources conservées ; les 32 shams reproduisent leurs sorties.

Une paire de cases initialement vides est tirée, à partir des seuls indices, dans chacune de cinq configurations par puzzle. Les huit chiffres différents de la sortie native de chaque case donnent 64 cibles doubles par paire. Les cibles identiques entre cases sont incluses. Total : **160 paires, 10 240 doubles éditions, 2 560 éditions isolées**. Les positions peuvent se répéter entre configurations ; les essais ne sont pas indépendants.

L'intervention est `z_final + (R12(c12 édité) - R12(c12 natif))`. Toutes les autres entrées discrètes restent fixes. Le résidu commun est recalculé pour les deux codes simultanément. On mesure les effets après le décodeur ; la commutation triviale des affectations de codes n'est pas le résultat testé. Aucune continuation du générateur n'est exécutée.

## Résultats

- Deux chiffres demandés écrits : **5 644/10 240 = 55,12 %**, IC descriptif 95 % [51,18 ; 59,01].
- Un seul chiffre demandé écrit : **3 848/10 240 = 37,58 %**, IC descriptif 95 % [34,51 ; 40,62].
- Aucun des deux chiffres demandés écrit : **748/10 240 = 7,30 %**, IC descriptif 95 % [5,74 ; 9,00].
- Contrat complet — les deux cibles, les 79 autres cases et les huit séparateurs préservés : **5 576/10 240 = 54,45 %**, IC descriptif 95 % [50,59 ; 58,36].
- Contrats d'écriture isolée réussis : **1 889/2 560 = 73,79 %**, IC descriptif 95 % [71,09 ; 76,52].
- Conjonction des deux contrats isolés correspondants : **5 504/10 240 = 53,75 %**, IC descriptif 95 % [49,70 ; 57,93].
- Rétention du contrat double lorsque les deux contrats isolés passent : **5 206/5 504 = 94,59 %**, IC descriptif 95 % [93,40 ; 95,56].
- Contrats isolés tous deux réussis mais contrat double raté : **298/5504**, répartis sur **32/32 sources**.
- Contrat double réussi malgré l'échec d'au moins un contrat isolé : **370**.
- Double édition changeant au moins une des 79 autres cases : **89/10 240**, sur **8/32 sources**.
- Double édition changeant un séparateur : **0/10 240** ; séparateur invalide : **0/10 240**.
- Deux cases cibles toutes deux inchangées : **650/10 240**. « Aucun chiffre demandé » n'implique donc pas automatiquement l'absence de tout changement.

La perte nette par rapport à la conjonction isolée vaut **-0,70 point(s)**, IC descriptif 95 % [-1,83 ; 0,48]. Cette comparaison est appariée aux mêmes cases et cibles ; ce n'est pas le carré d'un taux moyen de réussite.

- `adjacent` : deux cibles 1230/2048 (60,06 %) ; contrat complet 1227/2048 ; rétention 972/1182 (82,23 %).
- `row_distant` : deux cibles 1028/2048 (50,20 %) ; contrat complet 1026/2048 ; rétention 955/1012 (94,37 %).
- `column_other_block` : deux cibles 1077/2048 (52,59 %) ; contrat complet 1014/2048 ; rétention 1007/1014 (99,31 %).
- `block_only` : deux cibles 1223/2048 (59,72 %) ; contrat complet 1223/2048 ; rétention 1197/1215 (98,52 %).
- `unrelated` : deux cibles 1086/2048 (53,03 %) ; contrat complet 1086/2048 ; rétention 1075/1081 (99,44 %).

Les demandes avec chiffres identiques et différents sont séparées dans `summary.json`. Les comparaisons entre configurations restent descriptives : les relations spatiales et Sudoku ne sont pas orthogonales.

## Décision et portée scientifique

La règle exploratoire, fixée avant les sorties, signalait une forte dégradation si la rétention était inférieure à 95 %, ou si la perte nette dépassait 5 points. Décision du runner : **`strong_composition_degradation`**. Ces seuils sont des choix pratiques du pilote, sans justification confirmatoire indépendante. Le pilote franchit au moins un seuil local de dégradation de la composition : la règle testée ne justifie pas d'investissement architectural supplémentaire.

La rétention globale de 94,59 % est proche du seuil heuristique et son IC recouvre 95 % : il ne faut pas transformer le franchissement ponctuel en rejet statistique confirmé. Le constat le plus marqué est local : pour les cases voisines, 210 des 1 182 conjonctions isolées réussies échouent après composition, soit **17,77 % de pertes** ; la rétention vaut **82,23 %**, IC descriptif 95 % [77,79 ; 85,80]. Les cas sans unité commune ont une rétention de **99,44 %**. Ces différences motivent une dépendance au contexte spatial dans le domaine testé, sans en localiser la cause dans la convolution, l'attention ou une autre partie du lecteur.

Les 370 gains compensent numériquement les 298 pertes : le taux global du contrat double est même supérieur de 0,70 point à la conjonction des contrats isolés. Il n'y a donc pas d'effondrement global établi de la précision moyenne. Le résultat négatif concerne la **fiabilité des contrats individuels sous composition**, particulièrement pour des cases voisines, et la fiabilité absolue de l'interface proposée. Je recommande de ne pas engager la Piste 3 comme projet architectural en l'état ; cela ne prouve pas l'impossibilité d'une autre règle d'écriture.

Un audit **post-hoc**, distinct des critères gelés, sépare les cibles identiques des cibles différentes. Entre voisines, la rétention est de 9/130 = 6,92 % pour deux chiffres identiques, contre 963/1 052 = 91,54 % pour deux chiffres différents ; **89/1 052 = 8,46 %** des contrats isolés réussis échouent donc encore avec des cibles différentes. La perte ne se réduit pas aux demandes de même chiffre dans une ligne. Des cibles différentes ne garantissent toutefois pas une grille Sudoku valide. Cet audit ne localise aucun mécanisme et ne change pas la décision prospective. Voir `audit_target_equality.py` et `results/posthoc_target_equality_audit.json`.

Un échec de la demande double ne démontre pas à lui seul une interaction destructrice : il faut regarder les deux écritures isolées correspondantes. Réciproquement, une bonne rétention conditionnelle ne transforme pas la règle d'écriture en interface totale : elle conditionne l'analyse sur les opérations qui fonctionnent déjà isolément, sans fournir une méthode gratuite pour les reconnaître avant décodage.

Une modification directe des deux tokens de sortie satisfait exactement le même contrat terminal par construction, sans intervention latente. Le présent pilote n'établit aucune capacité qui lui échappe. Il ne démontre ni raisonnement révisé, ni adaptation des autres cases, ni accélération, ni amélioration de résolution. Les chiffres demandés peuvent rendre le Sudoku invalide ; la validité de la grille n'est pas l'objectif de cette intervention. Aucun solveur n'a servi à choisir une cible favorable.

Le résultat est conditionnel au checkpoint, à la table, au corpus non uniforme et aux paires testées. Il ne couvre pas les 16² codes, toutes les positions, plusieurs checkpoints, trois éditions ou plus, ni des interventions suivies de continuation. Il ne constitue pas un certificat universel.

## Incertitude et contrôles

Les IC sont des percentiles de 10 000 rééchantillonnages des **32 puzzles sources entiers**, configurations et cibles conservées ensemble. Ils sont descriptifs, sans tests de significativité ni correction de multiplicité. Les bornes bootstrap dégénérées en l'absence d'événement ne prouvent pas un taux populationnel nul. Pour un événement absent de toutes les sources, la borne supérieure de l'intervalle Wilson bilatéral à 95 % sur la proportion de sources affectées vaut **10,72 %**, sous une approximation binomiale ; elle ne borne pas directement le taux par intervention.

Le script distinct `verify.py` recalcule les métriques par boucles scalaires, les intervalles globaux et la rétention. Il rejoue les **400 interventions de la première source** avec des appels singleton, sur les 89 tokens, et retrouve toutes les sorties stockées. C'est un audit logiciel par le même agent de recherche, pas une réplication indépendante ou une revue externe. Temps écoulé du runner : 143.5 s, chargement et contrôles compris ; ce temps n'est pas un benchmark comparatif.

## Artefacts et arrêt demandé

- [Protocole](protocol.json), [gel et empreintes](results/protocol_frozen.json), [journal](JOURNAL.md).
- [Résultats agrégés](results/summary.json), [toutes les observations](results/interventions.npz), [audit](results/verification.json).
- [Commandes de reproduction et vérification](README.md), [runner](experiment.py), [vérificateur](verify.py).

**Arrêt après l'étape 1.** Le benchmark causal de l'étape 2 et le travail sur Wyner de l'étape 3 n'ont pas été lancés. Le manuscrit et les campagnes antérieures sont inchangés.

# Métacognition NCM — résultats sous amendement v0.3

Rapport détaillé H1. État consolidé : [synthèse des deux études](SYNTHESE_METACOGNITION.md).

État du 23 octobre 2026. La règle ci-dessous a été fixée après le diagnostic v0.2 et avant les nouveaux ajustements ; il s'agit d'une étude exploratoire. Le protocole initial, les sorties instables et la tentative échouée de réentraînement sont conservés.

## Correction et contrôles

Pour les variables observables, supprimer les colonnes exactement constantes sur les données d'ajustement, centrer, puis diviser par `max(écart-type population, 0,1)`. Cette règle fixe borne l'amplification à ×10 par unité native ; aucun seuil n'a été choisi en maximisant l'AUROC. Elle concerne toutes les colonnes observables sous ce seuil, pas seulement les probabilités problématiques : elle modifie aussi leur régularisation effective. Les 16 composantes PCA internes et le témoin normal gardent leur standardisation initiale. Aucun écrêtage des observations de validation.

Les autres choix restent ceux de v0.2 : partage, niveau 9, variables, pooling interne, grille λ, objectif, optimiseur et métriques. Les sources des essais précédents sont conservées ; le nouveau code est isolé. Les poids NCM et le chemin public ne sont pas modifiés.

## Développement : 1 000 puzzles, validation croisée imbriquée

AUROC des prédictions externes regroupées :

- Puzzle seul : **0,5227**.
- Résumés observables : **0,8667**.
- Référence enrichie : **0,8451**.
- Référence + états internes : **0,8508**.
- Référence + bruit : **0,8444** ; + états permutés : **0,8452**.

Apport interne regroupé **+0,00569** ; les cinq différences par pli sont positives. Intervalle bootstrap descriptif des scores fixés [0,00391 ; 0,00767] : il ne réentraîne pas les modèles et ne constitue pas le résultat du test principal. Log-loss de la référence 0,3132 ; avec états 0,3097, résumés seuls 0,2868. La référence enrichie demeure moins performante que les résumés : davantage de variables ne garantit pas un meilleur observateur avec ces données et cette famille de sondes.

Les variables de validation de ces modèles restent sous 10 en valeur absolue, les logits sous 7,3, contre des millions auparavant. Relecture des coefficients et transformations réussie, écart FP64 maximal 4,18e-14. Exécution des six modèles : 189,57 s. Ce diagnostic élimine l'explosion constatée sur le développement ; il ne garantit pas une robustesse universelle.

## Sensibilité au réentraînement

Audit terminé et vérifié : vingt bootstraps stratifiés des 799 puzzles des plis d'apprentissage 1..4, évalués sur les 201 puzzles du pli 0. Les copies d'une solution restent dans le même pli interne ; PCA et choix de λ sont refaits. Le maintien de ce même ensemble d'évaluation permet d'observer la variabilité d'ajustement ; ses percentiles ne sont pas un intervalle confirmatoire.

Les 20 gains sont positifs sur ce même lot de validation : médiane **+0,00619**, étendue **[+0,00248 ; +0,06854]**. Les deux gains élevés correspondent à λ=10 pour la référence contre λ=1 pour la sonde. Cette sensibilité inclut donc la sélection de régularisation, pas seulement la variation des coefficients. Le nombre de solutions distinctes par tirage est inférieur à 799 ; cet audit ne reproduit pas un nouvel apprentissage sur 1 000 solutions indépendantes. Exécution : 368,61 s ; relecture des transformations et tirages réussie, écart numérique maximal 2,76e-14.

Anomalie d'implémentation : le premier lanceur de cet audit réutilisait les plis d'origine après exclusion du pli 0 et laissait un pli interne vide. Arrêt avant le premier résultat exploitable ; tentative conservée dans `stability-v0.3-failed-01`. Le correctif crée les cinq plis dans le seul pool admissible, avec regroupement des doublons, et ajoute un test de régression. Aucun ajustement des paramètres scientifiques motivé par cet échec.

## Évaluation tenue à l'écart

**Test principal terminé et vérifié : 500 puzzles, 432 succès et 68 échecs.** Les quatre prédicteurs finaux ont été ajustés sur les 1 000 exemples d'apprentissage ; λ=1 a été choisi pour chacun par validation croisée. Coefficients, sources, métriques, graines et effectif ont été gelés dans [le manifeste](audit/metacognition/final-v0.3/lock.json) avant toute nouvelle inférence de calibration/test. Platt a été ajusté sur les seuls 500 exemples de calibration ; ses paramètres ont été écrits avant lecture du test pour l'analyse.

AUROC sur les logits bruts du test :

- Référence enrichie : **0,88351**.
- Référence + états internes : **0,88777**.
- Référence + bruit : **0,88371**.
- Résumés observables seuls : **0,91023**.

**Différence principale : +0,00426 ; intervalle apparié à 95 % [ +0,00201 ; +0,00681 ].** Calcul fixé de 5 000 bootstraps stratifiés des groupes de solutions, conditionnellement aux prédicteurs ajustés. La borne inférieure est positive ; la borne supérieure reste nettement sous le repère pratique de 0,03. Ce résultat soutient un petit apport discriminatif dans cette comparaison, mais pas un gain de cette ampleur pratique pour cette configuration. Les sensibilités de réentraînement sont rapportées séparément ; elles ne sont pas incorporées dans cet intervalle.

Après calibration, Brier de la référence **0,07523**, avec états **0,07379**, résumés seuls **0,06274** ; log-loss respectivement **0,26342**, **0,25899**, **0,22596**. L'ajout interne ne dépasse donc pas notre meilleur comparateur observable. La supériorité des résumés au modèle enrichi reste un résultat de cette famille et de ce prétraitement ; l'information observable n'est pas épuisée par un modèle linéaire pénalisé. Aucun bénéfice opérationnel ni mécanisme métacognitif natif n'est démontré.

Parité cache exacte sur les 1 000 nouvelles inférences de calibration/test ; contrôle eager supplémentaire sur 120 puzzles ; toutes les réponses conformes aux archives, empreinte des poids inchangée. Extraction : **169,31 s**. Sur le test, maximum absolu des variables **9,99**, maximum des logits bruts **5,14** : pas de réapparition de l'explosion. Une relecture indépendante sans ajustement valide les transformations, les métriques brutes/calibrées et les 5 000 rééchantillonnages. **22 tests passent.** Les diagrammes de fiabilité et courbes risque/couverture sont conservés sous forme numérique dans le rapport brut ; pas d'usage pour sélectionner un seuil de politique.

Note de schéma : la fonction historique de métriques nomme le Brier `uncalibrated_brier`, y compris dans le sous-objet `calibrated`. Dans ce sous-objet, le calcul reçoit bien les logits calibrés : interpréter cette clé comme le Brier de la transformation indiquée par son parent. Les fichiers gelés et la vérification arithmétique sont conservés.

Ce test reste **exploratoire** : les résultats historiques étaient connus pour la stratification et l'amendement suit une première analyse de développement. **Écart au volet de dimensionnement :** les 20 réentraînements mesurent une sensibilité, pas une simulation complète de puissance incluant l'apprentissage. Le test original de 500 a été conservé exploratoire, avec effectif et analyse figés avant ses nouvelles inférences ; aucune promesse de puissance à 80 % ni aucune confirmation indépendante n'est revendiquée. Ce volet reste à compléter avant un corpus confirmatoire. Le niveau 9 est le seul critère principal ; les sensibilités 3/6/11, planifiées dès v0.2, ont été exécutées séparément après ces résultats avec la même méthode et ne peuvent les remplacer.

## Sensibilités préspécifiées : niveaux 3, 6 et 11

Même partage, même méthode v0.3 et même grille, coefficients ajustés sur l'apprentissage puis gelés avant les nouvelles inférences. Ces analyses ont été exécutées **après lecture du résultat principal**, comme indiqué dans leur manifeste. Leurs intervalles sont secondaires, conditionnels aux prédicteurs, sans correction de multiplicité ; aucune sélection d'un niveau « gagnant » ni nouvelle confirmation indépendante.

- Niveau 3 : référence **0,84675**, avec états **0,85866** ; différence **+0,01191**, intervalle [0,00763 ; 0,01678]. Résumés seuls : 0,85750.
- Niveau 6 : référence **0,86925**, avec états **0,87377** ; différence **+0,00453**, intervalle [0,00194 ; 0,00749]. Résumés seuls : 0,89355.
- Niveau 11 : référence **0,87401**, avec états **0,87762** ; différence **+0,00361**, intervalle [0,00143 ; 0,00620]. Résumés seuls : 0,91537.

Les différences sont positives mais restent sous le repère de 0,03. Au niveau 3, la sonde dépasse à peine les résumés seuls (+0,00116, comparaison non principale et sans intervalle dédié) ; aux niveaux suivants, les résumés font mieux. Davantage de niveaux/variables n'améliore pas systématiquement le modèle enrichi ; aucune monotonie de la décodabilité n'est supposée. Les résultats ne permettent pas d'attribuer une valeur d'information théorique supérieure aux états internes.

Parité cache et réponses archivées vérifiées sur les 1 000 nouvelles inférences secondaires, poids inchangés. Les maxima des variables/logits restent bornés ; relecture indépendante des prédictions, métriques et 15 000 rééchantillonnages passée. **Le résultat principal reste celui du niveau 9.**

## Préparation d'une confirmation indépendante

Un pilote du générateur produit **32/32 puzzles nouveaux à 30 indices**, chacun à solution unique vérifiée par deux approches : backtracking pour le retrait des indices, puis solveur indépendant par couverture exacte. Les solutions sont exclues si équivalentes aux solutions officielles par relabellisation des chiffres ou les huit symétries du carré ; les nouvelles solutions sont aussi distinctes entre elles selon cette règle. Ce n'est pas une déduplication exhaustive de tous les isomorphismes. Graines et corpus haché archivés. Génération et contrôles : 1,87 s.

**Aucun de ces nouveaux puzzles n'a été évalué par le NCM.** Il s'agit d'une validation technique du générateur, sans effectif confirmatoire ni puissance établis, pas d'une confirmation statistique. La distribution induite par backtracking et retrait glouton n'est pas uniforme ni garantie identique à la distribution officielle. Les deux solveurs et la clé de symétries sont testés.

Preuves : [amendement](experiments/metacognition/amendment-v0.3.json), [développement](audit/metacognition/development-v0.3/report.json), [relecture](audit/metacognition/development-v0.3/verification.json), [réentraînements](audit/metacognition/stability-v0.3/report.json), [correctif des plis](experiments/metacognition/stability-correction-v0.3.json), [test principal](audit/metacognition/final-v0.3/analysis/report.json), [vérification du test](audit/metacognition/final-v0.3/analysis/verification.json), [sensibilités](audit/metacognition/sensitivity-v0.3/analysis/report.json), [leur vérification](audit/metacognition/sensitivity-v0.3/analysis/verification.json), [générateur](audit/metacognition/confirmation-generator-pilot/report.json), [commandes](experiments/metacognition/README.md).

## Suite donnée à ce jalon

L'utilisateur a choisi l'anticipation des corrections : [étude dédiée terminée](RAPPORT_CORRECTION_FUTURE.md). Son corpus externe et sa métrique concernent les cellules, pas la confirmation du petit gain H1 sur les puzzles. Le résultat principal H1 reste donc celui des 500 puzzles officiels ; les sensibilités 3/6/11 ne le remplacent pas.

L'utilisateur a ensuite demandé de **consolider les résultats avant toute nouvelle expérience**. La [synthèse](SYNTHESE_METACOGNITION.md) rassemble conclusions, limites, anomalies et chaînes de preuves. Aucune nouvelle sonde, politique, adaptation ou intervention n'est engagée. Une reprise exigera une hypothèse et une évaluation réservée ; les tests déjà consultés ne redeviendront pas des tests intacts.

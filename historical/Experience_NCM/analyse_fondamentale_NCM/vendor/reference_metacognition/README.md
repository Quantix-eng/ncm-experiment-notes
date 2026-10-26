# Expériences métacognition — Sudoku hard

Référence : [plan v0.2](../../PLAN_RECHERCHE_METACOGNITION.md), [spécification](protocol-v0.2.json). Ces scripts ne modifient pas le paquet d'inférence ni les poids.

Depuis la racine :

```sh
.venv/bin/python experiments/metacognition/run_pilot.py --prepare-only
.venv/bin/python -m unittest discover -s tests
.venv/bin/python experiments/metacognition/run_pilot.py
.venv/bin/python experiments/metacognition/verify_pilot.py
```

La première commande crée ou vérifie `audit/metacognition/split-v0.2.json` ; un partage différent est refusé. Le pilote refuse d'écraser un dossier de sortie existant. Pour une réplication, spécifier un nouveau `--output audit/metacognition/pilot-replication-01` ; le vérificateur ci-dessus cible le premier pilote archivé.

`trace.py` observe les appels existants du bloc final et des têtes de codes ; les copies ne changent pas les tenseurs retournés au générateur. Le décodage des reconstructions intermédiaires est réalisé après génération. Ce replay suffit à étudier le contenu des états et son coût séparé ; ce n'est pas encore un moteur d'arrêt anticipé en ligne.

`features.py` ne reçoit aucune solution. Il conserve les positions fixes, y compris lorsque le modèle prédit zéro ou un séparateur mal placé. Les variables au niveau k n'utilisent que les niveaux 1..k. Les cibles et la correction des cellules restent dans un fichier d'évaluation séparé.

Les archives NPZ contiennent uniquement des tableaux numériques et sont chargées avec `allow_pickle=False`. Elles sont exclues de Git ; leurs empreintes et tailles figurent dans le rapport. Le pilote enrichi en erreurs (24 succès, 8 échecs) ne permet pas d'estimer les taux du corpus. Les commandes du pilote ci-dessus n'entraînent aucune sonde.

Développement sur les seuls 1 000 exemples d'apprentissage :

```sh
.venv/bin/python experiments/metacognition/extract_training.py
.venv/bin/python experiments/metacognition/describe_training.py
.venv/bin/python experiments/metacognition/train_probes.py
.venv/bin/python experiments/metacognition/verify_development.py
.venv/bin/python experiments/metacognition/train_probes.py --models puzzle summary hidden_shuffle_control --output audit/metacognition/development-controls-v0.2
.venv/bin/python experiments/metacognition/verify_development.py --directory audit/metacognition/development-controls-v0.2
```

Les extractions et entraînements refusent d'écraser une sortie existante. `extract_training.py --resume` reprend seulement une extraction compatible dont chaque lot déjà écrit est vérifié. Une réplication d'entraînement exige un nouveau `--output` ; les variables mises en cache sont réutilisées après contrôle de leurs empreintes.

`train_probes.py` réalise cinq plis externes et cinq plis internes par pli externe. Le choix de régularisation, la normalisation et la PCA sont refaits à l'intérieur de chaque ensemble d'ajustement. Chaque exemple reçoit ainsi une prédiction externe sans avoir participé à cet ajustement ni à sa sélection d'hyperparamètres. Les groupes de solutions étant uniques dans le corpus hard retenu, la stratification interne ne divise aucun groupe. Les indices, graines et choix d'implémentation figurent dans le rapport brut. Les transformations et coefficients de chaque modèle externe sont sauvegardés pour relecture numérique indépendante par `verify_development.py`.

Cette procédure mesure des performances de développement. L'AUROC regroupant les prédictions des cinq modèles et son bootstrap descriptif ne constituent ni le résultat du test principal ni une estimation complète de l'incertitude de réentraînement. La calibration de Platt n'est pas effectuée ; les Brier rapportés utilisent les probabilités brutes. Aucun de ces scripts ne lance d'inférence ou d'ajustement sur calibration/test. La référence riche, la sonde principale et les témoins partagent les mêmes plis, grille et objectif ; le témoin de permutation reste un diagnostic, pas un test d'indépendance conditionnelle.

Amendement de normalisation v0.3, conservant intégralement les essais v0.2 :

```sh
.venv/bin/python experiments/metacognition/study_v03.py
.venv/bin/python experiments/metacognition/verify_development.py --directory audit/metacognition/development-v0.3
.venv/bin/python experiments/metacognition/stability_v03.py
.venv/bin/python experiments/metacognition/verify_stability_v03.py
```

La règle est fixée dans `amendment-v0.3.json` : colonnes observables exactement constantes supprimées, échelle `max(écart-type, 0.1)`, sans recherche de seuil. La PCA interne et le témoin normal conservent leur standardisation initiale. Les différences entre versions ne mesurent donc pas seulement l'effet de la correction sur les sorties rares : toute colonne observable d'écart-type inférieur à 0,1 est concernée, et la régularisation effective change.

Utiliser `stability_v03.py`, pas l'ancien mode `study_v03.py --mode stability` : ce dernier est conservé avec les sources de l'essai initial et sa tentative échouée. Le correctif de plis vides est décrit dans `stability-correction-v0.3.json`. Le bootstrap réentraîne les sondes, avec choix de régularisation et PCA, sur 20 tirages stratifiés des plis d'apprentissage 1..4 ; les copies d'une même solution restent dans le même pli interne. Évaluation sur le seul pli d'apprentissage 0, non sur le test : il s'agit d'une sensibilité au réentraînement, pas d'un intervalle confirmatoire.

Après vérification et gel de l'analyse, trois étapes distinctes :

```sh
.venv/bin/python experiments/metacognition/final_v03.py --stage fit
.venv/bin/python experiments/metacognition/final_v03.py --stage extract
.venv/bin/python experiments/metacognition/final_v03.py --stage evaluate
.venv/bin/python experiments/metacognition/verify_final_v03.py
```

`fit` n'utilise que l'apprentissage et refuse de continuer sans rapports de développement/stabilité vérifiés. Il fige coefficients, transformations, sélection de λ, sources, métriques et effectif original du test. `extract` vérifie le gel puis effectue les nouvelles inférences de calibration et test avec parité de référence. Il ne conserve que les variables au niveau 9. `evaluate` ajuste Platt sur la calibration et écrit ses paramètres avant de lire le test ; l'AUROC principale utilise les logits bruts, avec 5 000 rééchantillonnages appariés stratifiés. Chaque étape refuse d'écraser sa sortie. Aucun téléchargement ou entraînement du NCM. Le résultat tenu à l'écart reste exploratoire après l'amendement ; aucune puissance de 80 % ou confirmation indépendante n'est revendiquée.

Prototype du corpus indépendant, sans évaluation NCM :

```sh
.venv/bin/python experiments/metacognition/confirmation_corpus.py
```

Le pilote génère 32 puzzles à 30 indices ; unicité vérifiée séparément par backtracking et couverture exacte. Il exclut les équivalences de solutions connues (relabellisation des chiffres et huit symétries du carré), sans audit exhaustif des isomorphismes. Ce lot technique ne constitue pas une confirmation statistique et la distribution peut différer de l'officielle. Graines, rejets, puzzles, solutions et empreintes sont archivés. Aucun modèle NCM n'est appelé.

Sensibilités secondaires préspécifiées aux niveaux 3, 6 et 11 :

```sh
.venv/bin/python experiments/metacognition/sensitivity_v03.py --stage fit
.venv/bin/python experiments/metacognition/sensitivity_v03.py --stage extract
.venv/bin/python experiments/metacognition/sensitivity_v03.py --stage evaluate
.venv/bin/python experiments/metacognition/verify_sensitivity_v03.py
```

Elles sont exécutées après le résultat principal avec la méthode v0.3 inchangée et un nouveau gel des coefficients avant leurs inférences. La grille λ est choisie uniquement sur les 1 000 exemples d'apprentissage. Même test, donc aucune nouvelle confirmation indépendante ; intervalles secondaires non corrigés pour multiplicité. Elles ne peuvent pas remplacer le niveau 9 ni justifier de sélectionner rétrospectivement un niveau favorable.

Consolidation des résultats existants, sans inférence ni nouvel ajustement des sondes d'étude :

```sh
.venv/bin/python experiments/metacognition/consolidate_results.py
```

Cette commande lit les scores et manifestes existants, contrôle leurs empreintes et groupes, recalcule les AUROC directement par comparaisons de paires et vérifie les liens des rapports. Elle écrit uniquement `audit/metacognition/consolidation-v1.json`, avec les empreintes des documents courants. Les intervalles enregistrés ne sont pas réestimés. Elle ne remplace pas une réplication externe ; les gros tableaux locaux sont nécessaires pour la reproduction numérique. La [synthèse](../../SYNTHESE_METACOGNITION.md) est le point d'entrée du dossier consolidé.

# Correction future — étude exploratoire 0.1

Référence : [rapport](../../../RAPPORT_CORRECTION_FUTURE.md), [protocole](protocol-v0.1.json). Ne pas utiliser l'ancien test H1 pour sélectionner cette méthode.

Depuis la racine du projet :

```sh
.venv/bin/python experiments/metacognition/correction/audit_training.py
.venv/bin/python experiments/metacognition/correction/extract_training.py
.venv/bin/python experiments/metacognition/correction/run_development.py
.venv/bin/python experiments/metacognition/correction/verify_development.py
.venv/bin/python experiments/metacognition/correction/external.py --stage generate
.venv/bin/python experiments/metacognition/correction/external.py --stage fit
.venv/bin/python experiments/metacognition/correction/external.py --stage extract
.venv/bin/python experiments/metacognition/correction/external.py --stage evaluate
.venv/bin/python experiments/metacognition/correction/verify_external.py
```

Chaque extraction/ajustement/évaluation refuse d'écraser sa sortie ; une réplication nécessite une copie de travail sans ces données dérivées. Le protocole et le design externe sont figés avant leurs ajustements/mesures. Le gel externe inclut les fichiers Python du sous-dossier, leurs dépendances numériques, les modèles et le corpus : ne pas les modifier pendant une exécution. Les cellules d'un puzzle restent dans le même groupe. Les solutions et états futurs ne sont jamais passés comme prédicteurs. Le constructeur observable peut recevoir les traces complètes mais ne lit que les niveaux 1..9 ; cette frontière est testée par corruption des niveaux futurs.

`persistent.npy` décrit l'erreur finale sur toutes les cellules initialement vides ; seuls les indices où `current_error.npy` vaut vrai entrent dans l'apprentissage et l'évaluation de correction. Ce masque est calculé avec la solution, ce qui définit une cohorte d'analyse et non un détecteur déployable de l'erreur courante. La métrique principale pondère également chaque puzzle mixte, sans confondre nombre de cellules et nombre de groupes indépendants. Les autres puzzles participent à l'ajustement et aux métriques regroupées.

Le bootstrap d'évaluation rééchantillonne les puzzles mixtes, conditionnellement aux sondes ajustées. Une absence d'apport des 16 composantes du décodeur ne limite pas toutes les représentations du générateur ; un gain ne prouverait pas une surveillance ou un contrôle natif. La distribution synthétique peut différer de l'officielle et le nombre de puzzles mixtes est inconnu avant l'inférence ; aucun arrêt ou ajout d'exemples guidé par le résultat.

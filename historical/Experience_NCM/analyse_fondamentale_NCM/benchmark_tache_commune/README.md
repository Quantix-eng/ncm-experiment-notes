# Benchmark à tâche commune et contrats d'accès

Lire le [rapport](RAPPORT.md), le [manuscrit consolidé, version 3](PAPER_MATCHED_TASK.md), le [protocole avant entraînement](PROTOCOLE.md) et le [complément du papier](PAPER_ADDENDUM.md). Rapports exclusivement Markdown. Les anciens checkpoints, campagnes et articles restent conservés.

Deux petits Transformers appris, AR et parallèle, partagent une tâche de rotation de séquence, les partitions, le nombre de paramètres et les budgets. Le parallèle n'est pas un NCM miniature. Trois graines sont entraînées séparément. La procédure de calibration sur développement ne trouve aucune dose admissible ; son repli préfixé est scellé avant les nouvelles sorties d'évaluation. Les contrats d'accès conservent le contexte du receveur ou échangent toute la mémoire ; le second coupe son chemin **sous régénération**, tandis qu'une suite figée du receveur reste un canal distinct. Voir [la note sur les estimands](CONTRATS_ET_ESTIMANDS.md).

## Artefacts

- [Données](data/families.json), `data/train.npz` : 140 orbites d'entraînement, 30 de développement, 40 d'évaluation ; variantes, références et orientations restent dans leur partition.
- [Gel avant apprentissage](results/protocol_frozen.json), [apprentissage complet](results/training.json), six checkpoints dans `checkpoints`.
- [Calibration](results/calibration.json), [gel avant évaluation](results/calibration_frozen.json), fichiers `results/development-*.npz` : toutes les doses candidates et tous les entraînements conservés.
- [Résumé](results/summary.json), [audit](results/verification.json) et traces dans `results/evaluation`. Les statistiques sont séparées par architecture, graine et coupure ; aucune inférence ne traite les graines comme des familles supplémentaires.

Les fichiers natifs ont les axes famille/variante A,B,Aplus/état ou lecture. Les états ont ensuite lot=1/11 positions/48 coordonnées. Les cinq états sont embeddings puis quatre blocs. Les huit résidus sont attention puis MLP pour chacun des quatre blocs. Les logits sauvegardés dans O/C/E concernent le **premier symbole** parmi huit ; `answers` stocke séparément les réponses de longueur quatre générées librement.

Dans `*-cutK-C.npz`, axes des logits : famille/ancrage A,B/perturbation/condition/vocabulaire. Perturbations 0+,0−,1+,1− ; conditions Y00,Y10,Y11,Y01,Y1wrong. La suite mal appariée porte l'indice `q^2`. Les deltas, suites et états finaux sont aussi conservés.

Dans `*-cutK-E.npz`, axes : famille/contrat retenu ou mémoire entière/comparaison A-B ou A-Aplus/receveur/donneur/politique/vocabulaire. Politiques native gauche, native droite, régénérée. Les suites régénérées et états finaux conservent les mêmes axes appropriés. Les contributions ne sont pas des tokens imposés.

## Vérification et reproduction

Depuis la racine du projet :

```sh
analyse_fondamentale_NCM/.venv/bin/python analyse_fondamentale_NCM/benchmark_tache_commune/verify.py
```

Ce contrôle recharge les modèles, recompose les observations et rejoue des branches par équations explicites. Il ne réentraîne pas et n'ajuste aucun critère. Il écrit `results/verification.json` ; il s'agit d'un audit interne, pas d'une réplication scientifique indépendante.

Pour reproduire sans écraser, copier les scripts `.py`, `config.json` et `PROTOCOLE.md` dans un **nouveau dossier frère** sous `analyse_fondamentale_NCM`. Utiliser l'environnement consigné dans le gel, sans installation supplémentaire. Exécuter successivement `prepare.py`, `freeze.py`, `train.py`, `calibrate.py`, `evaluate.py`, `analyse.py`, `verify.py`. La préparation ne génère pas de sorties de modèle. La calibration ne lit que les familles de développement ; l'évaluation exige ses empreintes scellées.

Les scripts refusent de remplacer les phases d'entraînement ou d'évaluation existantes. Les trois dossiers historiques de comparaison sont nécessaires pour leurs contrôles d'intégrité. Les rapports, journal et manifeste final sont établis après résultats. Le [journal](JOURNAL.md) documente tout incident éventuel et la portée de l'audit.

`build_paper.py`, ajouté après résultats, assemble éditorialement la version 3 à partir de la version 2 et du complément de cette campagne, en rétablissant leurs liens. Il n'effectue aucun calcul expérimental et ne modifie pas ses sources. Les anciennes versions de l'article restent disponibles dans leurs dossiers respectifs.

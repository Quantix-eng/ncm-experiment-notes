# Benchmark causal des révisions — calibration 0.1

Lire [RAPPORT.md](RAPPORT.md) pour les résultats exécutés et [PROTOCOLE.md](PROTOCOLE.md) pour le protocole et ses limites. Les rapports sont en Markdown ; aucune présentation visuelle n'est produite.

La [note d'identifiabilité](IDENTIFIABILITE.md) précise les équivalences algébriques et les projections d'accès effectivement exécutées : les traces internes d'audit ne sont pas les observations fournies aux méthodes. Le [journal](JOURNAL.md) conserve la chronologie et cette clarification de portée.

Cette ressource évalue des méthodes d'attribution, avec des témoins analytiques connus par construction et des paires Sudoku à réponse contrôlée. Elle n'établit pas encore une performance d'interprétabilité sur un modèle appris. Elle n'est ni un benchmark de référence validé ni une évaluation extérieure.

Depuis la racine du dépôt, audit sans changer les observations :

```sh
analyse_fondamentale_NCM/.venv/bin/python analyse_fondamentale_NCM/benchmark_causal_revision/verify.py
```

Pour une reproduction, copier les scripts `.py`, `protocol.json` et `PROTOCOLE.md` dans un **nouveau dossier frère** sous `analyse_fondamentale_NCM`, puis exécuter successivement son `run.py --freeze`, son `run.py`, son `verify.py` et son `publish.py`. Les scripts refusent de remplacer un gel ou un corpus existant. Les corpus historiques locaux servent à exclure des familles connues ; ils restent en lecture seule. L'environnement Python/Numpy et les versions des solveurs sont épinglés dans le gel et les résultats.

Les fichiers `data/families.json` et `data/evaluator_truth.json` contiennent les tâches et la vérité des programmes construits. Le second appartient à l'évaluateur et ne doit pas être donné à une méthode d'analyse. `results/observations.json` contient les observations et les ensembles de classes annoncés. `methods.project` impose l'accès à O, C ou E lors de l'évaluation ; cette projection est un contrat expérimental, pas une sandbox protégeant des participants hostiles. `results/raw_vectors.npz` conserve les scores, états et suites pour l'audit, avec chargement sans pickle. Ce fichier expose des variables de construction et n'appartient pas au régime d'observation O.

Les scripts de calibration ne prennent pas de modèle NCM en entrée : l'adaptateur appris et son protocole confirmatoire restent à écrire après cette étape. Ne pas mélanger les étiquettes de construction et les hypothèses sur un réseau.

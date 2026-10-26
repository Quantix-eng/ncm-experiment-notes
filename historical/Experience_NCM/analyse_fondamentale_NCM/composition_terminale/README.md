# Pilote de composition de deux écritures terminales

Étape 1 demandée le 25 octobre 2026. Lire `RAPPORT.md` pour le résultat et ses limites. Les campagnes historiques et le manuscrit restent inchangés.

Le protocole et les deux scripts sont gelés localement avant les nouvelles interventions, avec les empreintes des données, des tables et des sources importées. Le corpus existe déjà : il ne s'agit ni de puzzles nouvellement réservés ni d'une prérégistration indépendante.

Depuis la racine du dépôt, audit des résultats conservés :

```sh
analyse_fondamentale_NCM/.venv/bin/python analyse_fondamentale_NCM/composition_terminale/verify.py
```

`results/interventions.npz` contient toutes les sorties sur 89 tokens, les cibles, positions, codes et états natifs. `summary.json` contient les métriques et les intervalles descriptifs. `verification.json` consigne l'audit logiciel et le rejeu singleton de 400 interventions. Aucun résultat négatif n'est exclu.

Pour reproduire dans un nouveau dossier, copier `protocol.json`, `experiment.py` et `verify.py` dans un autre dossier directement sous `analyse_fondamentale_NCM`, puis exécuter son `experiment.py --freeze`, son `experiment.py`, et son `verify.py`. Les scripts refusent de remplacer un gel ou les observations existantes. La reproduction réutilise les mêmes données, poids, table et interventions ; elle n'est pas une réplication indépendante.

La comparaison à l'édition directe de sortie est le contrat exact construit à partir de la sortie native et des deux chiffres demandés. Elle ne requiert aucun solveur et ne représente pas une amélioration de résolution du Sudoku.

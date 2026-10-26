# Directions mécanistes NCM Sudoku

[Rapport complet](RAPPORT.md) · [Objectifs et protocole](OBJECTIFS_ET_PROTOCOLE.md) · [Journal](JOURNAL.md)

Deux questions : pourquoi le niveau 5 utilise-t-il si peu de codes, et peut-on prévoir puis influencer la valeur des révisions au niveau 9 ? Les compléments conditionnel et dropout sont explicitement post-hoc. Les résultats initiaux restent conservés.

Les scripts, figures PNG/SVG, paramètres, prédictions par puzzle et modèles ajustés sont dans ce dossier. L'archive `sources_directions_mecanistes.zip` contient tous les scripts utilisés et les dépendances du modèle copiées. Les poids, données et utilitaires de la campagne précédente sont référencés en lecture dans le dossier parent ; conserver cette arborescence.

Depuis ce dossier : `./run.sh` reproduit la campagne entière, ce qui est long. Pour contrôler les résultats existants :

```sh
PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" ../.venv/bin/python scripts/verify.py
```

`results/verification.json` donne les contrôles effectivement passés ; `results/final_manifest.json` contient les empreintes des artefacts finalisés. La vérification rejoue notamment les sondes sur un sous-échantillon test et une lecture des codes intervenus avec le décodeur gelé. Elle ne réexécute pas tous les entraînements ni toutes les générations.

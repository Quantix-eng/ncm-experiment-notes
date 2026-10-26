# Audit NCM entre Sudoku et texte

[Article anglais](PAPER.md) · [PDF](PAPER.pdf) · [Rapport français](RAPPORT.md) · [Protocole](OBJECTIFS_ET_PROTOCOLE.md)

L'étude distingue l'invariance d'un pooling, la diversité empirique des codes et la possibilité de court-circuiter une prédiction. Deux tokenizers texte officiels, LM1B et TinyStories, complètent les diagnostics Sudoku. Le générateur hard est testé avec 9, 10 et 11 prédictions.

L'archive `sources_generalisation.zip` contient les codes utilisés, configurations et dépendances de modèles copiées, sans les gros poids et corpus. `results/` conserve toutes les mesures et les prédictions par exemple. `figures/` propose quatre figures PNG/SVG/PDF. La première campagne et `directions_mecanistes/` restent inchangés.

Depuis ce dossier, `./run.sh` reproduit les expériences séquentiellement. L'environnement de campagne est celui de `../.venv` ; les versions sont dans `sources/environment.json` et `sources/requirements-lock.txt`. La création du PDF nécessite Pandoc et pdfLaTeX. Pour le vérifier, PyMuPDF est installé seulement dans `.deps` ; il peut être installé localement avec `PIP_CACHE_DIR="$PWD/.cache/pip" ../.venv/bin/python -m pip install --target .deps pymupdf`.

```sh
PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ../.venv/bin/python scripts/verify.py
```

Les limites, interruptions et amendements sont documentés dans [JOURNAL.md](JOURNAL.md) et [AMENDEMENT_MEMOIRE.md](AMENDEMENT_MEMOIRE.md). Les vérifications effectivement passées sont dans `results/verification.json`, les empreintes finales dans `results/final_manifest.json`.

Le vérificateur contrôle par défaut les empreintes finales déjà présentes. Le runner complet appelle `verify.py --finalize` après reconstruction volontaire des résultats et du rapport pour établir les nouvelles empreintes ; les manifestes antérieurs et les entrées officielles restent vérifiés dans les deux modes. PyMuPDF est épinglé dans `sources/pdf-requirements.txt` et les versions de Pandoc/pdfLaTeX sont enregistrées dans `sources/environment.json`.

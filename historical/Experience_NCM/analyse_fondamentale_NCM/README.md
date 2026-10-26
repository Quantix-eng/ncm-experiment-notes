# Analyse fondamentale NCM — campagne Sudoku isolée

Lire `RAPPORT.md` pour la méthode, les résultats et les limites, `TODO.md` pour le protocole et `JOURNAL.md` pour les amendements concis. Le rapport est généré lorsque les expériences sont terminées.

- `scripts/` : code exécuté ; `vendor/` : copies des implémentations.
- `artifacts/` et `data/` : copies des poids et corpus.
- `results/` : traces par puzzle, sondes, statistiques et vérification.
- `figures/` : PNG et SVG exportables ; `logs/` : sorties des processus.
- `sources_utilisees.zip` : archive des codes et du protocole, construite à la livraison.

Reproduction complète, depuis ce dossier, sous Python 3.13 :

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --cache-dir .cache/pip -r requirements.txt
./run.sh
```

Si `.venv` existe déjà, elle peut être réutilisée. Les entrées copiées `artifacts/` et `data/sudoku/` sont nécessaires ; `results/input_manifest.json` conserve leurs empreintes SHA-256. `sources/requirements-lock.txt` documente l'environnement complet de campagne, y compris des paquets système non utilisés ; `requirements.txt` liste les dépendances directes réellement utilisées.

Vérification sans réexécuter la campagne :

```sh
PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" .venv/bin/python scripts/verify.py
```

La campagne initiale utilise CPU FP32, un thread par processus, avec quelques étapes indépendantes concurrentes. `run.sh` reproduit séquentiellement les mêmes expériences. Le temps de campagne n'est pas un benchmark de vitesse des modèles.

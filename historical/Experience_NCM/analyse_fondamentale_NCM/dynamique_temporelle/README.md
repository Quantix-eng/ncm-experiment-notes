# Étude temporelle séparée

Lire [RAPPORT.md](RAPPORT.md) pour les conclusions et [OBJECTIFS_ET_PROTOCOLE.md](OBJECTIFS_ET_PROTOCOLE.md) pour les critères. La note anglaise est dans PAPER.md et PAPER.pdf. Les campagnes précédentes restent intactes.

Reproduction complète : `./run.sh` depuis ce dossier. Vérification des artefacts, recalcul des métriques et relecture d'un lot d'interventions :

```sh
PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ../.venv/bin/python scripts/verify.py
```

Environnement : réutilisation en lecture du venv parent et des checkpoints, sans nouvelle installation. Python, NumPy, PyTorch et dépendances sont identifiés dans sources/environment.json et le verrou du parent. La publication PDF nécessite Pandoc et pdfLaTeX aux chemins explicités dans scripts/publish.py. Une reproduction numérique sur une autre architecture n'est pas garantie bit à bit.

Les grands NPZ d'observation et les poids ne sont pas dupliqués ; leurs empreintes sont contrôlées dans les manifestes antérieurs. sources_temporelles.zip contient tous les nouveaux scripts et les archives de sources amont nécessaires. La vérification rejoue un lot de 16, pas toutes les branches ; run.sh reconstruit la campagne entière. Les fichiers générés et les caches sont confinés à ce dossier.

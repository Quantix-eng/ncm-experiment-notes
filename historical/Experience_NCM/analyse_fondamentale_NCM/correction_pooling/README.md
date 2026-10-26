# Correction du pooling : variante et tests

[Rapport français](RAPPORT.md) · [Article intégré](PAPER.md) · [PDF](PAPER.pdf) · [Supplément](SUPPLEMENT.md) · [Protocole](OBJECTIFS_ET_PROTOCOLE.md)

Cette extension teste une pondération fixe dépendant de la position, avec neuf positions de sortie et poids NCM gelés. Elle distingue sensibilité à l'ordre, diversité des codes, reconstruction et robustesse, puis contrôle l'utilisation effective du patch par le générateur. Les articles et manifestes précédents restent inchangés.

Reproduire depuis ce dossier : `./run.sh`. Vérifier les résultats présents :

```sh
PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ../.venv/bin/python scripts/verify.py
```

Le runner utilise l'environnement `../.venv` et les données/poids copiés dans le dossier d'analyse parent. La version intégrée inclut les résultats antérieurs, dont le runner demeure `../generalisation_architecturale/run.sh`. Pandoc et pdfLaTeX produisent le PDF ; PyMuPDF, déjà installé dans `../generalisation_architecturale/.deps`, sert à son inspection. Les versions de l'environnement précédent sont archivées dans `sources/environment_previous.json`.

`sources_correction.zip` contient tous les nouveaux scripts et les sources des modèles utilisées ; les gros poids et corpus restent dans leurs dossiers. `results/verification.json` indique les contrôles passés. La vérification ordinaire contrôle aussi les empreintes finales existantes ; `--finalize` les renouvelle après une reproduction volontaire, en conservant le contrôle des campagnes précédentes.

L'usage du patch réversible est explicite dans `scripts/repair.py` :

```python
with patched(model.vqvae.vq, "weighted_1"):
    codes = model.vqvae.encode(solutions)
    logits = model.vqvae.decode(codes)
```

Il s'applique uniquement aux représentations Sudoku 89×256 et au niveau à neuf positions. Le chemin de génération gelé est vérifié séparément ; modifier cet encodeur ne signifie pas que le générateur prédit les nouvelles cibles.

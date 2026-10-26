# Pistes mécanistes de révision

Lire [ANALYSE_ET_PISTES.md](ANALYSE_ET_PISTES.md) : conclusion de l'audit, hypothèses concurrentes, plan croisé de compensation, localisation et critères de validité. Cette extension ne modifie pas les articles antérieurs.

L'audit réutilise les tableaux `decoded` et `reconstructed` déjà vérifiés et rejoue uniquement le codec sur 16 solutions hard test. Les nouvelles expériences proposées ne sont pas exécutées. Toutes les données avaient été examinées auparavant.

Depuis ce dossier, reproduction de l'audit et de l'archive :

```sh
PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ../.venv/bin/python scripts/audit.py
PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" ../.venv/bin/python scripts/package.py
```

Le venv, les poids et les traces du parent sont lus sans modification. Python 3.13, PyTorch 2.8.0, NumPy 1.26.4 ; versions et dépendances complètes dans `../sources/requirements-lock.txt`. L'archive de sources inclut le code des modèles, les configurations et les utilitaires importés, sans dupliquer les poids ou les grands NPZ.

Les IC de ce nouveau bootstrap ne sont pas identiques à la dernière décimale à ceux de l'étude temporelle, car sa graine diffère. Les comptes bruts et les sous-échantillons correspondants sont identiques. Le manifeste final de cette extension inclut les résultats et l'archive, hors caches.

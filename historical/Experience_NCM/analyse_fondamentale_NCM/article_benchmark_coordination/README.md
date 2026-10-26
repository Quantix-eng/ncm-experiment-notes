# Papier sur l'attribution des révisions et la coordination

Entrée principale : [PAPER.md](PAPER.md), manuscrit scientifique en anglais. Lire aussi le [rapport de consolidation](RAPPORT_CONSOLIDATION.md), la [note Wyner](THEORIE_WYNER.md) et le [registre de preuves](PREUVES.md). Aucun support visuel n'est produit.

Ce dossier réutilise la calibration et l'étude NCM conservées dans les dossiers frères. Il ajoute une analyse **post-hoc** de codes déjà générés et une sensibilité aux seuils, sans appel de réseau. Les campagnes antérieures et l'ancien manuscrit ne sont pas modifiés. Les preuves mathématiques concernent des lois et des restrictions architecturales explicitement spécifiées, pas un mécanisme identifié du NCM.

Depuis la racine du projet, vérifier les empreintes historiques, les comptes du papier, les liens locaux et les analyses auxiliaires :

```sh
analyse_fondamentale_NCM/.venv/bin/python analyse_fondamentale_NCM/article_benchmark_coordination/verify.py
```

L'audit écrit `results/verification.json` dans ce dossier et ne relance aucune ancienne expérience. Il refuse toute discordance avec les manifestes historiques et le gel auxiliaire. Les empreintes finales de la livraison sont conservées séparément dans `results/final_manifest.json`.

Pour reproduire le calcul auxiliaire, créer un **nouveau dossier frère** sous `analyse_fondamentale_NCM`, y copier `auxiliary.py` et `PROTOCOLE_AUXILIAIRE.md`, puis exécuter, dans l'ordre, ces commandes en remplaçant `NOUVEAU_DOSSIER` :

```sh
analyse_fondamentale_NCM/.venv/bin/python analyse_fondamentale_NCM/NOUVEAU_DOSSIER/auxiliary.py --freeze
analyse_fondamentale_NCM/.venv/bin/python analyse_fondamentale_NCM/NOUVEAU_DOSSIER/auxiliary.py
```

Le script refuse d'écraser le gel ou les observations existantes. Il utilise les mêmes traces et ne constitue donc pas une réplication sur de nouvelles données. Les dépendances de calcul sont Python et NumPy de l'environnement local existant.

Le [journal](JOURNAL.md) documente le statut post-hoc et la relecture. Les détails de reproduction des inférences originales restent dans les README de la [calibration](../benchmark_causal_revision/README.md) et de l'[étape 3a](../benchmark_ncm_reel/README.md).

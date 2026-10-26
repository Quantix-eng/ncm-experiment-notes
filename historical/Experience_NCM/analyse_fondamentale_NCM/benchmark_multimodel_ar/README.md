# Extension du benchmark à GPT-2

Lire le [rapport](RAPPORT.md) et le [manuscrit consolidé, version 2](PAPER_MULTIMODEL.md). Le [protocole gelé](PROTOCOLE.md) et le [complément AR autonome](PAPER_ADDENDUM.md) donnent les détails de cette campagne. Rapports exclusivement Markdown. Ce dossier ajoute une seconde architecture apprise, GPT-2 small, sur une tâche textuelle de copie ; les expériences et l'article NCM antérieurs restent conservés.

Les interventions portent sur le calcul du **prochain token**, entre les blocs 6 et 7. Le contexte antérieur du receveur est conservé. Elles n'étudient pas la réécriture de tokens déjà émis. Les trois politiques de suite sont des contrôles expérimentaux, jamais des étiquettes d'architecture.

## Vérifier les observations conservées

Depuis la racine du projet :

```sh
analyse_fondamentale_NCM/.venv/bin/python analyse_fondamentale_NCM/benchmark_multimodel_ar/verify.py
```

L'audit recharge GPT-2, recompose les 1 024 sorties croisées et factorielles, rejoue les sous-couches sur les deux premières familles et recalcule les pertes et diagnostics. Il ne sélectionne aucune nouvelle expérience et ne change pas les observations. Il écrit `results/verification.json`. Ce contrôle par le même agent ne constitue pas une réplication scientifique indépendante.

## Artefacts et axes

- [Familles](data/families.json) : prompts A, B, Aplus, mots et références à un token, identifiants et tokens complets. Choix sans consultation de sorties GPT-2.
- [Gel local](results/protocol_frozen.json) : chemins absolus, SHA-256, versions réellement importées, nombre de paramètres. [Journal](JOURNAL.md) : ordre des opérations et statut.
- `results/native.npz` : logits `[famille, variante, 13 lectures, vocabulaire]`, séquences résiduelles complètes `[famille, variante, 13 états, lot=1, positions, 768]`, suites natives `[famille, variante, 12 contributions, 768]`, tokens et références.
- `results/crossed.npz` : logits `[famille, ancrage A/B, 4 perturbations, 5 conditions, vocabulaire]`. Conditions Y00,Y10,Y11,Y01,Y1wrong. Directions 0+,0−,1+,1− ; la suite mal appariée est à l'indice `q^2`. États, deltas et contributions sont aussi conservés.
- `results/exchanges.npz` : logits `[famille, comparaison A/B ou A/Aplus, receveur, donneur, politique de suite, vocabulaire]`. Politiques native gauche, native droite, régénérée. Les suites régénérées et états finaux sont sauvegardés.
- [Résumé](results/summary.json), `results/metrics.npz`, [audit](results/verification.json) : statistiques groupées par famille, valeurs individuelles et comptes complets.
- [Comptes appariés descriptifs](results/paired_descriptive.json) : export ajouté après résultats par `postprocess.py`, sans nouvelle inférence ni changement des diagnostics. [Versions installées et effectivement utilisées](results/environment_packages.json).

Les logits stockent tout le vocabulaire ; le score de justesse est un argmax global, pas un choix forcé entre deux mots. Les références ne sont utilisées ni par le modèle ni par la sélection des perturbations.

## Reproduire sans écraser

Créer un **nouveau dossier frère** sous `analyse_fondamentale_NCM`, y copier les scripts `.py`, `PROTOCOLE.md`, `protocol.json` et `data/families.json`. Installer les dépendances dans son propre sous-dossier `deps`, en utilisant les versions consignées dans le gel ; les versions effectives de NumPy et PyTorch sont celles de l'environnement historique. Le bootstrap les importe **avant** d'ajouter `deps` au chemin Python. Ne pas installer dans l'environnement des campagnes anciennes.

Les fichiers de `artifacts/gpt2` viennent de `openai-community/gpt2` à la révision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Poids safetensors, configuration, tokenizer et métadonnées de téléchargement doivent être disponibles. Les deux dossiers historiques de comparaison sont aussi nécessaires pour leurs contrôles d'intégrité. Tous les appels d'inférence utilisent ensuite les fichiers locaux, sans réseau.

Dans ce nouveau dossier, exécuter dans l'ordre `run.py --freeze`, `run.py`, `analyse.py`, `verify.py` avec `analyse_fondamentale_NCM/.venv/bin/python`. `prepare.py` n'est utile que pour régénérer les métadonnées du corpus dans un dossier sans `data/families.json` ; il n'exécute pas le modèle. Le runner refuse de remplacer des observations natives existantes. Une reproduction sur les mêmes poids et familles ne constitue pas une réplication externe.

`postprocess.py` exporte ensuite les comptes appariés descriptifs, avec refus d'écrasement. Il a été ajouté après résultats. `build_paper.py`, également postérieur au gel expérimental, assemble éditorialement la version 2 depuis l'article historique et les sections AR ; il n'effectue aucun calcul scientifique et ne modifie pas son texte source. Le rapport et les références restent Markdown ; aucun support visuel n'est généré.

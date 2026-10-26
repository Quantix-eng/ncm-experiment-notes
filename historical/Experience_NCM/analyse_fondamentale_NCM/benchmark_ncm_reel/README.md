# Étape 3a : benchmark sur NCM réel

Lire [RAPPORT.md](RAPPORT.md) et [PROTOCOLE.md](PROTOCOLE.md). Rapports Markdown uniquement. Les anciennes campagnes, poids et données sont en lecture seule.

Le [complément de lecture](COMPLEMENT_SORTIES.md) donne les comptes de justesse, d'indices et de format pour toutes les branches, ainsi que le contrôle essentiel de suite native du receveur figée. Son exporteur descriptif a été ajouté après lecture des résultats principaux ; les seuils et décisions gelés sont conservés.

Depuis la racine du dépôt, vérification des observations conservées :

```sh
analyse_fondamentale_NCM/.venv/bin/python analyse_fondamentale_NCM/benchmark_ncm_reel/verify.py
```

Pour reproduire dans un **nouveau dossier frère** sous `analyse_fondamentale_NCM`, y copier les fichiers `.py`, `protocol.json` et `PROTOCOLE.md`, puis exécuter successivement son `run.py --freeze`, son `run.py`, son `analyse.py`, son `verify.py`, son `publish.py` et son `audit_sorties.py`. Le runner refuse de remplacer le gel ou les sorties natives existantes. Les mêmes 16 familles et poids doivent rester disponibles dans le parent. Cette reproduction n'est pas une réplication indépendante ; son gel inclura l'exporteur ajouté après le gel de la campagne initiale.

`native.npz` contient les 12 niveaux, avec dimensions famille/variante/niveau. `crossed.npz` utilise famille/ancrage A ou B/perturbation/condition Y00,Y10,Y11,Y01,Y1wrong pour ses logits. `exchanges.npz` utilise famille/comparaison A-B ou A-Aplus/entrée réceptrice/état donneur/type de suite (native gauche, native droite, recalculée). Les suites sont sauvegardées contribution par contribution pour préserver l'ordre d'addition FP32.

Les diagnostics ne prétendent pas fournir la vérité mécaniste du NCM. Les références sont des instruments d'évaluation et ne sont jamais utilisées par les branches génératrices. Un cas insuffisant ou ambigu est conservé et rapporté sans ajustement post-hoc.

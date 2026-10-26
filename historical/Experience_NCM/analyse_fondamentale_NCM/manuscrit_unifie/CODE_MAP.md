# Organisation des preuves et du code

`tools/` contient uniquement les outils de synthèse : `assemble.py` copie, `claims.py` indexe, `build.py` compose, `check.py` contrôle les documents et empreintes, `package.py` archive. Aucun outil documentaire n'importe Torch, les modèles ou les runners historiques.

`code/history/<campagne>/scripts/` est une copie du code utilisé, avec `documents/` pour ses protocoles, rapports, journaux et commandes d'origine. Les huit alias sont résolus dans `evidence/campaigns.json`. La séparation par campagne préserve leurs différentes méthodes et leur chronologie ; fusionner leurs fonctions en un nouveau runner modifierait les sources qui servent de preuves.

`code/models/ncm_mac/` est l'implémentation CPU exécutée et instrumentée. `code/models/upstream-ncm/` contient le code amont utilisé pour les contrôles de parité. `code/configs/` et `code/environment/` conservent configurations et versions. Les dépendances tierces sont documentées, sans être attribuées à nos scripts.

`evidence/records/<campagne>/` contient les JSON historiques copiés ; `evidence/manifests/` leurs manifestes finaux. `evidence/traces/` contient des membres NPY copiés **octet pour octet** à partir des NPZ historiques, sans transformation des valeurs. `traces.json` identifie chaque fichier source, son empreinte et les membres conservés. Les états volumineux omis restent repérables dans les historiques.

`evidence/claims.json` associe chaque affirmation aux résultats, au code et aux traces. `numbers.json` contient les pointeurs des valeurs utilisées ; `document_numbers.json` distingue celles effectivement imprimées. Les identifiants E01–E14 présents dans le manuscrit renvoient à cette carte.

`figures/overview.tex` est le nouveau schéma conceptuel : il ne représente pas une nouvelle mesure. Les autres figures sont copiées avec leurs empreintes. Les flèches de calcul du schéma ne doivent pas être interprétées comme une causalité nouvelle entre les quatre résultats.

Les nouveaux contrôles vérifient l'assemblage documentaire. Ils n'exécutent pas les anciens `verify.py`, dont certains rejouent les modèles. Un audit des documents ne doit jamais être annoncé comme une réplication des expériences.

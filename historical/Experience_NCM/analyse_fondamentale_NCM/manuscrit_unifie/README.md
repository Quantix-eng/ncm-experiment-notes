# Manuscrit unifié : point d'entrée

La synthèse rassemble huit campagnes terminées autour de la question : **que permettent réellement d'attribuer les révisions apparentes du NCM ?** Elle ne lance aucune nouvelle expérience. Les campagnes historiques et leurs conclusions restent intactes.

- [Manuscrit PDF](MANUSCRIPT.pdf) : texte principal, références, puis annexes identifiées comme supplément.
- [Texte principal modifiable](MANUSCRIPT.md) et [annexes](SUPPLEMENT.md) ; sources éditoriales dans `paper/`.
- [Plan et portée révisés](PLAN_ET_PORTEE.md) et [décisions après critique](REPONSE_A_LA_CRITIQUE.md).
- [Carte des preuves](EVIDENCE_MAP.md) : E01–E14, résultats, code, traces et limites ; [chiffres et pointeurs JSON](evidence/numbers.json).
- [Organisation du code](CODE_MAP.md) ; [journal de synthèse](JOURNAL.md).

Le dossier organise des copies pour la lecture et l'audit, sans déplacer ni modifier les huit campagnes. Il contient les résumés JSON complets copiés, les scripts et documents historiques, deux implémentations des modèles, les configurations et des traces compactes. Les poids et grands états restent dans les dossiers historiques avec leurs empreintes. Les archives de sources complètes de chaque campagne y sont également conservées. L'archive supplémentaire du manuscrit inclut cette vue compacte ; elle ne prétend pas remplacer tous les gros artefacts historiques.

Depuis ce dossier :

```sh
./run.sh
```

Cette commande assemble les copies, extrait les valeurs **déjà enregistrées**, compose le PDF et vérifie les artefacts. Elle n'importe aucun modèle, ne lance aucun runner historique et ne recalcule aucun IC. Python 3, Pandoc, une installation LaTeX avec TikZ et BibTeX sont nécessaires ; les caches et sorties de composition restent ici. Les anciennes commandes d'expériences dans `code/history/` sont conservées comme preuves et ne doivent pas être confondues avec ce parcours documentaire. Pour vérifier uniquement : `PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" python3 tools/check.py`.

Pour auditer l'archive compacte hors de ce projet : `python3 tools/check.py --local-only`. Ce mode vérifie les copies livrées et signale l'absence de contrôle des originaux ; il ne nécessite ni poids ni installation de Torch. Le contrôle documentaire n'est pas une reproduction des mesures.

Le style officiel TMLR est épinglé et conservé sans modification, en mode `preprint`. Les annexes suivent les références, comme le permet le [guide officiel](https://jmlr.org/tmlr/author-guide.html). Il s'agit d'un brouillon : aucune soumission ou acceptation n'est annoncée. L'auteur responsable, les affiliations et déclarations doivent être renseignés et vérifiés avant toute soumission. Les [règles éditoriales](https://jmlr.org/tmlr/editorial-policies.html) demeurent la référence pour celle-ci.

La traçabilité est un moyen d'audit, pas une garantie de validité ou de nouveauté. Le codec oracle ne prouve pas une origine exclusive des revirements ; le pooling n'a pas été relié causalement aux oscillations ; la compensation spécifique testée n'est pas établie ; le certificat est externe et conditionnel. Aucune expérience n'opérationnalise la conscience.

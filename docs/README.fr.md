# Notes d’expériences sur NCM

[English](../README.md) · [Comparer les 20 expériences](../README.md#all-20-experiments-at-a-glance) · [Liens entre les études](../experiments/README.md) · [Reproduction](reproducing.md)

La question de départ était simple : quand une réponse intermédiaire change, que peut-on en déduire du calcul interne ? Nous avons essayé plusieurs pistes sur NCM, puis quelques contrôles sur GPT-2 et Pythia. Beaucoup n’ont pas abouti. Le dépôt garde leurs résultats et le code, avec les observations qui méritent encore d’être discutées.

## Vue d’ensemble des 20 expériences

Le [tableau du README principal](../README.md#all-20-experiments-at-a-glance) rassemble les 20 expériences : domaine, type de résultat, mesure principale et interprétation, avec un lien vers chaque fiche. Les identifiants d’origine sont conservés à côté de l’ordre de lecture.

Pour filtrer ou trier, télécharge le [catalogue CSV](../experiments/catalog.csv) dans un tableur. Un tableau Markdown GitHub reste statique. La page [Liens entre les études](../experiments/README.md) explique quels résultats et contrôles lire ensemble. Chaque fiche détaille le résultat, son interprétation, ses limites et ses sources. Les métriques de tâches différentes ne sont pas directement comparables.

## Commencer sans télécharger les modèles

Depuis la racine du dépôt, sous Python 3.10 ou plus :

```sh
python3 scripts/check_pooling_identity.py
python3 scripts/recompute_ioi.py
python3 scripts/verify_release.py
```

Ces commandes vérifient respectivement une identité de pooling, des moyennes calculées depuis les observations IOI conservées, puis l’intégrité et les liens du dépôt. Elles ne rejouent pas les expériences neuronales.

Le [guide](reproducing.md) distingue ces vérifications d’un petit rejeu NCM et de la reproduction complète des campagnes historiques. Les scripts d’origine sont présents. Les poids, les dépendances installées et les grosses traces doivent être téléchargés ou régénérés ; il reste du travail pour rendre tous les rejeux portables.

## Comment lire les résultats

Chaque expérience sépare ce qui a été mesuré, notre interprétation et ce qui reste ouvert. Les résultats négatifs ne sont pas cachés. Les anciens rapports sont conservés tels quels dans `historical/`, même lorsque leur interprétation a évolué.

Ce travail est indépendant des auteurs de NCM. Il ne prétend pas établir une théorie de la métacognition, corriger l’architecture ou améliorer les modèles. Les outils d’IA ont largement participé au travail ; leurs vérifications ne constituent pas des réplications humaines indépendantes.

Les [limites et la provenance](scope.md) et les [travaux voisins](related-work.md) complètent les notes. Pour signaler un problème, indique l’expérience, le fichier et, si possible, la commande concernée.

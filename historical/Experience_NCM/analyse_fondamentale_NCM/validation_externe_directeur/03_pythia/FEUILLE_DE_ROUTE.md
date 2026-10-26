# Pythia-160M : feuille de route du contrôle de lecture finale

25 octobre 2026. Proposition 3 du directeur. Responsable : sous-agent de recherche interne. Les campagnes historiques sont conservées. Les états d'avancement et résultats effectivement obtenus figurent dans [RAPPORT.md](RAPPORT.md).

## Question, résultat attendu et limite

Une normalisation finale peut changer les logits après remplacement d'une tête alors qu'aucune contribution neuronale en aval ne s'adapte. Le protocole doit enregistrer cet effet de lecteur sans le transformer en preuve d'une continuation adaptative nécessaire. Ce test porte sur un contrôle externe structurel ciblé ; il ne valide aucune taxonomie universelle et ne démontre pas une nouveauté par réplication.

## P1 — Vérifier la cible extérieure

Lire l'article de [Rushing et Nanda](https://arxiv.org/html/2402.15390v2#S3), le [checkpoint officiel](https://huggingface.co/EleutherAI/pythia-160m) et le [graphe GPT-NeoX effectivement importé](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/gpt_neox/modeling_gpt_neox.py). Vérifier les douze couches, douze têtes et le branchement attention/MLP parallèles. Retenir L11H2 et L11H10 suivant le texte du directeur et de l'article ; signaler l'incohérence avec la légende de la figure 5 qui mentionne L11H0. Aucun remplacement de cible selon un effet favorable.

Livrable : révision exacte, empreintes des poids et du code importé ; équations précises du lecteur figé et du lecteur recalculé.

## P2 — Constituer les partitions sans consulter les effets

Les expériences publiées utilisent The Pile. La présente extension utilise un corpus accessible et plus petit, [WikiText-2 officiel](https://huggingface.co/datasets/Salesforce/wikitext), configuration `wikitext-2-raw-v1` : 128 lignes de la validation officielle et 512 lignes du test officiel. Cette substitution est fixée avant inférence et interdit de présenter l'étude comme une reproduction exacte de la distribution publiée.

Une ligne admissible fournit au moins 33 tokens. Un seul préfixe de 32 tokens et son token suivant sont retenus par ligne ; les textes dupliqués sont exclus avant tirage. Tirage déterministe sans remise, sans sélection sur la justesse, le dommage ou l'effet direct. Deux donneurs distincts sont fixés dans chaque partition pour le même site d'intervention. La dépendance entre lignes et entre échanges interdit de présenter les requêtes comme des réplications indépendantes.

Livrable : fichiers sources, table des 640 séquences et donneurs, règles de sélection et nombres de lignes admissibles.

## P3 — Geler avant inférence

Écrire le runner, son audit distinct, les données et le protocole avant de calculer les logits. Épingler les scripts, poids, données et implémentation GPT-NeoX par SHA-256 dans `results/protocol_frozen.json`. Le gel est local ; il n'est pas une prérégistration indépendante. Tout défaut découvert après gel impose un amendement et conservation des versions antérieures.

Livrable : [protocol.json](protocol.json), gel horodaté et environnement exact.

## P4 — Développement puis test réservé

CPU FP32, un thread, attention eager et cache désactivé. Capturer l'entrée de la dernière couche, le MLP parallèle, les 12 sorties de tête avant projection et le résidu avant LayerNorm finale. Remplacer uniquement la sortie de la tête choisie à la dernière position par celle d'un donneur. Conserver biais de projection, autres têtes et MLP.

Recomposer les résidus avec l'ordre d'addition du graphe réel. Rejouer par le réseau complet les huit premières corruptions préspecifiées de chaque partition et vérifier le MLP inchangé. L'audit de lecteur recalcule des pertes depuis les résidus stockés.

Le test est autorisé si les contrôles techniques de développement passent : écart absolu des logits inférieur à `2e-4`, parité de recomposition du résidu sous `1e-5`, MLP inchangé. Ce contrôle de structure n'exige pas de dommage positif pour être informatif ; aucun seuil de dommage favorable ne décide de l'ouverture du test. Pour une revendication de récupération, le dommage et sa portée resteraient néanmoins indispensables.

Livrables : observations de développement, audit, puis observations des 512 séquences réservées sans recherche de tête favorable.

## P5 — Séparer deux estimands

Le lecteur est `D_s(r)=W_U[((r-moyenne(r))/s)*gamma+beta]`, avec `s=sqrt(var(r)+epsilon)`. Seul le dénominateur est figé ; le centrage reste celui du résidu testé. Pour le résidu natif `r0`, le résidu corrompu `rd`, les facteurs natif `s0`, propre `sd` et autre donneur `so`, conserver `D_s0(r0)`, `D_s0(rd)`, `D_sd(rd)`, `D_sd(r0)` et `D_so(rd)`.

Les contrastes NLL `D`, `R`, `I`, `S` appliqués à ces cinq lectures sont des **contrastes de lecteur**. Au contrat qui place la LayerNorm dans le lecteur, la continuation neuronale après la contribution attention finale est vide : les contrastes d'adaptation neuronale valent zéro. Les deux analyses doivent garder des noms distincts.

Calculer également la métrique publiée brute `Δlogit_centré − ΔDE_tête`, avec effet direct utilisant le facteur natif. Aucun clipping de ratio, aucune sélection des 2 % de positions les plus favorables. Rapporter NLL, logit cible centré, exactitude vocabulaire complet, facteur de normalisation et signes des effets par séquence, donneur et tête.

## P6 — Restituer sans surinterpréter

Rapporter les résultats défavorables, la différence de corpus et les coûts réellement chronométrés. Les intervalles par rééchantillonnage de séquences sont descriptifs ; ils ne corrigent ni les donneurs partagés ni l'origine documentaire commune. La réussite du contrôle technique établit l'implémentation ; l'absence de continuation neuronale est une propriété du graphe vérifiée, pas une vérité découverte par classification.

Critère scientifique : une lecture modifiée avec contributions aval inchangées ne doit pas être attribuée à l'adaptation de blocs aval. Une erreur de ce type falsifierait l'interprétation du diagnostic. Une simple réplication de l'effet de LayerNorm ne suffit pas à établir la valeur ajoutée du benchmark.

## Reproduction prévue

Depuis la racine du projet :

```sh
analyse_fondamentale_NCM/.venv/bin/python analyse_fondamentale_NCM/validation_externe_directeur/03_pythia/scripts/prepare.py
analyse_fondamentale_NCM/.venv/bin/python analyse_fondamentale_NCM/validation_externe_directeur/03_pythia/scripts/seal_amendment.py
analyse_fondamentale_NCM/.venv/bin/python analyse_fondamentale_NCM/validation_externe_directeur/03_pythia/scripts/seal_fp64.py
analyse_fondamentale_NCM/.venv/bin/python analyse_fondamentale_NCM/validation_externe_directeur/03_pythia/scripts/run.py --split validation
analyse_fondamentale_NCM/.venv/bin/python analyse_fondamentale_NCM/validation_externe_directeur/03_pythia/scripts/verify.py --split validation
analyse_fondamentale_NCM/.venv/bin/python analyse_fondamentale_NCM/validation_externe_directeur/03_pythia/scripts/run.py --split test
analyse_fondamentale_NCM/.venv/bin/python analyse_fondamentale_NCM/validation_externe_directeur/03_pythia/scripts/verify.py --split test
```

Après exécution, `prepare.py` ne doit pas être relancé dans le dossier historique : une nouvelle reproduction doit utiliser une copie indépendante et conserver le premier gel. L'[amendement logiciel 1.1](AMENDEMENT_LOGICIEL.md), l'[amendement de mesure 1.2](AMENDEMENT_FP64.md) et l'échec technique initial sont conservés ; les scripts de scellage rendent ces corrections explicites dans une reproduction complète. Le recalcul historique de l'effet direct du développement est documenté séparément : il utilise des états existants et n'est pas nécessaire à une nouvelle exécution complète déjà en version 1.2.

## Clôture du 25 octobre 2026

- P1 à P3 : sources, graphe, partitions et gels réalisés ; les versions 1.0, 1.1 et 1.2 sont conservées avec leurs motifs.
- P4 : développement 128 séquences puis test réservé 512 séquences exécutés. Parité native exacte, MLP parallèle inchangé et huit rejeux complets par exécution sous les tolérances fixées.
- P5 : contrastes de lecteur et métrique publiée exportés ; aucun effet positif n'a conditionné l'ouverture du test. Les moyennes de récupération restent faibles, avec IC descriptifs recouvrant zéro.
- P6 : [rapport complet](RAPPORT.md), observations, audits et manifeste de livraison produits. Sur le test, la normalisation propre à la corruption ne change aucun statut correct/incorrect par rapport au facteur natif figé.

L'audit gelé de développement est passé. Pour le test, le script gelé qui recharge les poids n'a pas été exécuté ; un audit supplémentaire par mémoire mappée a été ajouté après les résultats pour éviter un chargement concurrent de modèles. Il recalcule la même formule FP64, les mêmes 32 premiers essais et la même tolérance, sans nouveau forward. Il passe 160 pertes et 8 192 contrastes ; son statut post-hoc est explicite dans [son export](results/test_memmap_verification.json). Cette adaptation d'audit ne remplace aucune observation ou critère scientifique.

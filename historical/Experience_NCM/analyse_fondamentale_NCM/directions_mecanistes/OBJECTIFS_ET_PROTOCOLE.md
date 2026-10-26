# Deux directions mécanistes NCM — objectifs et protocole v1

## Ce que nous voulons établir

**A. Niveau 5.** Expliquer la faible diversité des codes : le résidu est-il déjà presque constant à cette échelle, ou le quantificateur perd-il une variabilité encore présente ? Distinguer le rôle de la contribution latente et le coût de sa prédiction. L'origine historique de l'apprentissage ne pourra pas être établie sans trajectoires d'entraînement ou réentraînements comparables.

**B. Révision 8 → 9.** Tester si les informations disponibles avant le niveau 9 prédisent le prochain chiffre et la qualité du changement, au-delà des probabilités déjà lisibles au niveau 8. Puis déterminer si une direction prédictive peut influencer la révision par l'intermédiaire des codes générés, et pas seulement par un effet direct sur le décodeur.

## Cadre commun

- Checkpoint principal : Sudoku hard ; diagnostic des tokenizers easy/medium en complément. Sources et résultats nouveaux dans ce sous-dossier, campagne antérieure conservée.
- Partitions officielles conservées : 1 000 train / 400 validation / 600 test par famille de solution. Ces données ont déjà été examinées : étude exploratoire, pas confirmation indépendante.
- Réplication externe : 256 puzzles nouveaux de la campagne précédente, sans ajuster les sondes sur eux. Même générateur de corpus et mêmes limites que précédemment.
- CPU FP32 déterministe, batch 16, seed 73129. Bootstraps appariés par puzzle, 2 000 tirages ; jamais par cellule supposée indépendante. IC conditionnels aux sondes ajustées.
- Pas de revendication de conscience, de contrôle métacognitif, de circuit monosémantique ou de transition de phase.

## A. Diagnostic du niveau 5

1. Pour chaque tokenizer : 256 solutions de train et 256 de test, mêmes familles à travers difficultés. À chaque niveau : variabilité du résidu avant/après pooling, distances aux codes, marges de sélection, contribution résiduelle et MSE avant/après, statistiques EMA sauvegardées.
2. Vérifier les codes par accumulation explicite et comparer les distances rapides au calcul direct en double précision. Une égalité avec l'amont ne suffit pas à exclure un bug d'entraînement original.
3. Contrôles diagnostiques hard sur 64 solutions : permutation des chiffres valide ; permutation interne à chaque ligne ; permutation globale ; chiffres aléatoires. Les trois derniers ne sont pas des puzzles valides : ils testent la représentation, pas la performance en Sudoku.
4. Comparer pooling 8/9/10/18 positions et moyenne exacte par ligne sur le même résidu avant niveau 5 ; ce ne sont pas des modèles réentraînés.
5. Réajuster un k-means à 16 centres sur les seuls résidus de train et mesurer la quantification sur test. Cela mesure une variabilité géométriquement exploitable, sans prouver un gain de décodage après réentraînement.
6. Sur les 600 tests hard et les 256 externes : remplacer la prédiction du niveau 5 par le code 11 déterminé sur train, en conservant sa contribution ; comparer à supprimer sa contribution en régénérant la suite. Rapport des codes, logits, sorties et réussites. Pas de promesse de gain de vitesse sur la seule base du nombre de passes.

## B. Sondes prospectives

1. Extraire l'état cumulatif z8 et les probabilités du décodeur au niveau 8. Les références ne servent qu'aux cibles et à l'évaluation. Les codes, logits et sorties du niveau 9 sont interdits dans les prédicteurs.
2. Unité de prédiction : case initialement vide. Cible principale : token prédit au niveau 9 (12 classes pour conserver aussi les tokens invalides). Cible secondaire : inchangé / corrigé / dégradé / erreur remplacée par une autre erreur.
3. Baseline : probabilités niveau 8, chiffre actuel, candidats autorisés par les indices, coordonnées, statistiques globales observables. Comparaisons : états seuls ; baseline + z8 local + projection fixe du résumé global de couche 8 ; contrôle permutant les états de train entre puzzles.
4. Sondes linéaires ridge multi-sorties ; alpha dans {10,100,1000}, température softmax dans {0.1,0.25,0.5,1}, choisis par log-loss validation. Standardisation sur train seulement. Pas d'optimisation sur test/externe.
5. Critère principal prédictif : différence d'exactitude du prochain token entre combinaison et baseline sur les cases qui changent effectivement (sous-groupe d'évaluation, jamais sélection accessible au déploiement). Rapporter aussi toutes les cases, log-loss, F1 macro et la baseline du deuxième choix du décodeur.

## C. Intervention sur la direction de révision

- 96 premiers puzzles du test fixé. Choisir une case et un chiffre alternatif via la sonde gelée, sans consulter la transition réelle ni la référence.
- Direction locale dans z8 : différence des poids de la sonde pour le chiffre alternatif et le chiffre actuel, ramenée aux coordonnées brutes.
- Perturbations de norme 0.25 puis 0.5 fois la norme de z8 à cette case : direction positive, négative, trois directions aléatoires appariées et sham. La dose 0.25 est principale ; 0.5 teste la robustesse descriptive.
- Mesurer adoption du chiffre demandé, log-rapport de probabilités demandé/actuel, autres cases touchées, codes du niveau 9 et réussite finale.
- Contrôles de lecture : (i) intervention complète ; (ii) soustraire la perturbation du z9 au moment du décodage, en conservant les nouveaux codes 9 ; (iii) ajouter la perturbation au z9 natif sans changer ses codes. Cela distingue influence via le générateur et effet direct de lecture, sans identifier une médiation naturelle unique dans un réseau non linéaire.
- Critère causal principal : gain du log-rapport demandé/actuel dans la lecture (ii), direction positive contre moyenne des directions aléatoires, bootstrap par puzzle. Un effet sur la seule lecture (i) ne suffit pas à établir une direction de révision utilisée par le générateur.

## Livraison et limites prévues

Rapport avec résultats positifs et négatifs, figures exportables, prédictions et interventions par puzzle, sondes sauvegardées, sources, commandes et manifestes SHA-256. Contrôles de parité contre le runner précédent avant toute interprétation. Les trajectoires d'entraînement ne sont pas disponibles ; aucun petit réentraînement substitutif ne sera présenté comme l'explication de l'entraînement original.

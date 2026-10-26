# Analyse fondamentale NCM — protocole v1

Périmètre : checkpoints Sudoku easy, medium et hard ; étude approfondie hard.
Conclusions limitées à ces checkpoints. Toutes les écritures restent dans ce dossier.
Article : https://arxiv.org/html/2610.14162v1 ; code : https://github.com/HazyResearch/ncm.

## 0. Préspécification et traçabilité
- [ ] Geler ce protocole avant les résultats ; consigner chaque amendement.
- [ ] Copier sources, poids et corpus ; SHA-256, versions, machine, seeds, commandes et durées.
- [ ] Partitions par puzzle source ; contrôler doublons et symétries connues ; garder toutes ses variantes dans sa partition.
- [ ] Monitoring opérationnel : prédiction de réussite finale depuis les observations disponibles au niveau k. Une sonde externe ne prouve ni monitoring natif ni contrôle.
- [ ] Critère principal : gain d'AUROC test de baseline+état contre baseline au niveau 6 sur hard. Brier, ECE et risque/couverture secondaires. Score constant : AUROC=0,5 si deux classes.
- [ ] 1 000/400/600 puzzles hard train/validation/test (regroupement prioritaire en cas de doublons). Standardisation sur train ; hyperparamètres sur validation. Bootstrap apparié par puzzle, 2 000 tirages ; autres niveaux exploratoires.
- [ ] Pas de meta-d′ sans modèle SDT adapté à cette tâche multiclasse structurée.

## 1. Reproduction et instrumentation
- [ ] Réévaluer 2 000 puzzles par checkpoint : exactitude stricte, cases correctes, validité, indices préservés ; formats invalides conservés comme échecs.
- [ ] Parité CPU avec/sans cache sur 16 puzzles par checkpoint ; accumulateurs intermédiaires, alignement logits/codes et nettoyage des hooks/cache.
- [ ] Distinguer couches Transformer (8 pour Sudoku) et niveaux de génération (12).

## 2. Tokenizer — solutions connues, items 1–3
- [ ] Solutions officielles : histogrammes par niveau, entropie (nats), perplexité, utilisation, codes non observés et couverture selon taille du corpus.
- [ ] Reconstruction cumulative token/case/grille. Entropie marginale ≠ information complémentaire ou causalité ; code non observé ≠ définitivement mort.
- [ ] Distances d'embeddings puis substitutions fonctionnelles dans un contexte identique : 32 solutions hard, toutes les paires au premier emplacement de chaque niveau.
- [ ] Distinguer proximité géométrique et interchangeabilité au décodage, sans seuil universel présumé.
- [ ] Comparer codes encodés et générés. Ne jamais encoder la solution pour construire une sonde de résolution.

## 3. Stabilité et symétries, items 4–6
- [ ] Trois répétitions de 32 puzzles ; écarts d'états, logits, codes.
- [ ] Batch 1/8/32, ordre inversé et compositions différentes ; écarts FP32 et divergences discrètes séparés.
- [ ] 128 puzzles hard : permutation des chiffres, rotation et réflexion ; références transformées, sorties réalignées.
- [ ] Variance intra/inter-puzzles et équivariance des sorties. Une variation d'activations n'invalide pas en soi une représentation.
- [ ] Score de confiance avec sonde gelée ; bootstrap groupé par puzzle pour variantes.

## 4. Généralisation, difficulté et confiance, items 7–10
- [ ] Croiser checkpoint et difficulté sur 256 puzzles communs par corpus en plus des diagonales complètes.
- [ ] Générer 256 puzzles à 30 indices, unicité vérifiée par deux solveurs distincts ; exclure doublons exacts et transformations connues des solutions officielles (pas toutes les isomorphies).
- [ ] Comparer officiel/nouveau : décalage du générateur, pas identification vu/non-vu. Sans données d'entraînement, contamination indéterminée.
- [ ] Difficulté : candidats initiaux, célibataires nus, progression par propagation simple ; proxies algorithmiques, pas échelle humaine universelle.
- [ ] Comparer prior constant, difficulté, sorties observables, états seuls, sorties+états et labels mélangés. Surveiller nombre d'erreurs et déséquilibre.
- [ ] AUROC avec IC, Brier, ECE (10 intervalles fixes), risque/couverture. Distinguer calibration et discrimination.
- [ ] Dernier niveau : validité Sudoku peut révéler la réussite ; ne pas appeler cela anticipation métacognitive.

## 5. Mécanismes ciblés exploratoires, items 11–13
- [ ] Sonder activations locales finales : cases initialement naked single parmi cases vides ; train/test par puzzle, contrôle fondé sur candidats.
- [ ] Décodabilité ≠ neurone monosémantique ou utilisation effective de la règle.
- [ ] Niveau 6/couche finale : retirer projection sur direction de confiance apprise sur train ; sham et cinq directions aléatoires de même norme ; mesurer amplitude réelle, corrections et erreurs introduites sur test uniquement.
- [ ] Distinguer effet ciblé et dégradation générique ; effet nul ≠ absence de rôle causal (redondance/puissance).
- [ ] Suivre décodabilité de solution par sonde linéaire (projection fixe 128 dimensions, ridge alpha=100), qualité intermédiaire et révisions correct→incorrect / incorrect→correct.
- [ ] Décrire progression ou saut ; pas de transition de phase ni de commit sans test dédié et réplication.

## 6. Livraison
- [ ] Rapport autonome : méthode, effectifs, résultats/incertitudes, interprétation, résultats négatifs, limites, état des 13 items.
- [ ] Sources utilisées, sorties par puzzle, poids des sondes, figures exportables et commande de reproduction/vérification.
- [ ] Distinguer exécuté, exploratoire, non exécuté et impossible à établir.

Ordre : 0 → 1 → 2/3 → 4 → 5 → 6. Avancement : JOURNAL.md ; paramètres : protocol.json.

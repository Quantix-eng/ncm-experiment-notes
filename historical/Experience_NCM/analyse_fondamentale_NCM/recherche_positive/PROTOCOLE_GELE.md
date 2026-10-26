# Alphabet terminal et lecture certifiée : test réservé

Gel après exploration des anciens train/validation et pilote causal sur 32 puzzles validation. La table terminale prédit 99,38 % des chiffres natifs hard sur validation ; l'écriture causale du pilote atteint seulement 81,25 %. Il s'agit donc d'une hypothèse d'alphabet partiel, pas d'un canal indépendant du contexte. La lecture seule du résidu terminal reconstruit beaucoup de chiffres mais zéro grille complète dans ce pilote ; ce résultat reste conservé.

## Deux effets annoncés

1. Sur les 128 premiers nouveaux puzzles, remplacer un code terminal par celui associé à un chiffre doit augmenter la fréquence du chiffre ciblé par rapport à une perturbation gaussienne locale appariée en norme à chaque position. Choisir une case initialement vide par graine, indépendamment des sorties. Tester tous les chiffres et exclure les cibles déjà présentes avant intervention ; agréger dans chaque puzzle avant bootstrap. L'édition peut rendre la grille fausse : c'est un test mécaniste, pas une amélioration de résolution.
2. Sur les 1 024 nouveaux puzzles hard, plus de 30 % des lectures par table doivent être certifiables sans décodeur. La règle est gelée : si contraintes, indices et séparateurs passent, accepter ; sinon décoder normalement. Aucune référence ne participe à cette décision.

IC percentile à 97,5 % pour ces deux effets, ajustement Bonferroni ; IC à 95 % descriptifs. 4 000 tirages par puzzle source, les neuf cibles et les contrôles restant ensemble. Le seuil de couverture 30 % est un minimum pratique fixé sur développement avant le test. Aucun seuil de causalité absolue à 90 % n'est revendiqué : le pilote ne le soutient pas.

## Garantie conditionnelle

Pour un puzzle p à solution unique s, si le vérificateur est sain, toute grille acceptée est s. Si le décodeur natif retournait s, le système hybride renvoie soit s par certificat, soit exactement la même sortie native en fallback. Ainsi son succès exact ne peut pas diminuer sur chaque puzzle satisfaisant les préconditions. Cette propriété n'exige pas une bonne table. Pour des puzzles à plusieurs solutions, elle concerne la validité avec indices et ne garantit pas la fidélité à une référence arbitraire.

Sur machine, baseline et fallback décodent chacun un puzzle à la fois, avec le même état latent, afin d'éviter une réorganisation de batch qui modifierait les arrondis FP32. Mesurer séparément le décodeur natif en batch pour documenter cette distinction. Le vérificateur n'emploie pas le solveur d'unicité à l'exécution ; l'unicité est une précondition des données et du théorème.

## Corpus et gel

Générer après ce gel 1 024 solutions par backtracking aléatoire, puis retirer des indices en conservant l'unicité jusqu'à 30. Vérifier MRV et Algorithm X. Exclure les clés D4/normalisation des chiffres des solutions officielles, externes initiaux, nouveaux puzzles de compensation et nouvelles solutions déjà retenues. L'audit ne couvre pas toutes les isomorphies ni la contamination d'entraînement inconnue. L'échantillon et le générateur sont fixés ; aucune exclusion selon la réponse du NCM.

Tables et alphabets proviennent de train seulement ; validation a servi à choisir cette direction. Les études antérieures ne seront pas présentées comme des confirmations aveugles. Répliquer secondairement la lecture/hybridation easy et medium sur les 256 premières nouvelles familles partagées, avec tables distinctes.

## Temps de calcul et livrables

Benchmarks après les expériences, sans calcul concurrent de campagne : CPU FP32, un thread, warmup exclu. Utiliser les 64 premiers puzzles nouveaux sans sélection sur acceptation. Pour chaque puzzle, trois paires AB/BA randomisées de génération native puis lecture native/hybride ; mêmes primitives de génération et sorties contrôlées. Mesurer aussi la seule lecture, avec seize répétitions internes. IC sur les log-rapports de latence, bootstrap par puzzle après moyenne des répétitions. Un gain de l'étape de lecture ne sera pas présenté comme un gain identique du NCM complet.

Préserver toutes les mesures, y compris les gains nuls ou inversés. Fournir rapport français, article anglais/PDF, scripts, tables, prédictions, tests du certificat, archive et empreintes. Les autres campagnes restent intactes. Aucun patching d'attention n'est nécessaire.

# Complément exploratoire : apprendre la valeur de remplacement

Motif observé après v1 : entraînées sur toutes les cases, les sondes du prochain token privilégient la persistance. Sur les cases qui changent, la combinaison atteint 4,50 %, alors que le deuxième choix du décodeur atteint 49,40 %. Le gain v1 de 1,52 point contre la baseline apprise ne constitue donc pas une anticipation substantielle des valeurs de remplacement.

Le critère principal v1 et ses résultats restent conservés. Complément explicitement exploratoire :

- Ajuster les sondes de valeur de remplacement sur les seules cases changées de train ; choisir alpha/température sur les seules cases changées de validation. Exclure le token actuel des prédictions alternatives. Ne jamais utiliser le masque futur de test comme prédicteur.
- Comparer baseline, états, combinaison et contrôle à états de train permutés, ainsi que le deuxième choix du décodeur.
- Ajouter six statistiques locales accessibles au niveau 8 : entropie, maximum, deuxième probabilité, marge, probabilité du token actuel, nombre de candidats autorisés normalisé.
- Ajuster séparément une porte binaire qui prédit le changement ; former une distribution jointe : probabilité de rester × token actuel + probabilité de changer × distribution conditionnelle. Rapporter AUROC de la porte, qualité de la distribution jointe et exactitude conditionnelle sans masquer l'accès futur requis pour définir ce sous-groupe d'évaluation.
- Tester à nouveau les interventions à dose 0,25 uniquement sur les mêmes 96 puzzles, avec direction de la sonde conditionnelle. Choisir la case par probabilité prédite de changement × probabilité du chiffre alternatif. Aucune transition réelle ou référence ne sert à sélectionner la case.
- Réutiliser les contrôles direct/indirect, signe négatif et trois directions aléatoires à norme appariée. Conserver les deux séries causales, sans remplacer la première par la seconde.

Le corpus externe a déjà été consulté pour v1 : ce complément fournit une évaluation externe descriptive, pas un jeu de confirmation demeuré intact.

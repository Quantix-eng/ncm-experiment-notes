# Bascule temporelle, nécessité et réversibilité

## Hypothèses falsifiables

L'objectif est de rechercher une éventuelle bascule fonctionnelle, sans imposer une conclusion de transition de phase. Distinguer : contribution utile des derniers niveaux, accélération locale des corrections, verrouillage des valeurs décodées, stabilité de la qualité et résistance aux perturbations.

- Un arrêt précoce suivi d'une mauvaise réponse établit la nécessité de continuer, pas celle du seul niveau 9.
- Un pic d'argmax peut résulter des seuils de lecture du décodeur et de l'ajout d'information. « Artefact de lecture » et « rôle fonctionnel » ne sont pas une alternative exhaustive.
- Le verrouillage strict à 9 est réfuté si des valeurs corrigées à 9 redeviennent incorrectes plus tard sur la trajectoire native. Un autre critère mesure simplement le changement de valeur, même entre deux erreurs.
- Une transition de phase au sens fort demanderait une famille de systèmes, un paramètre de contrôle, des répétitions et une analyse de taille finie. Douze niveaux de longueurs et poids différents ne fournissent pas cette preuve ; davantage de puzzles ne constitue pas une augmentation de la taille du système.

## Données et arrêt temporel

Réutiliser explicitement les traces déjà vérifiées des campagnes antérieures : hard/easy/medium officiels, hard test de 600 et externe de 256. Les poids sont gelés. Les analyses sont séparées par checkpoint, avec familles de solutions corrélées entre difficultés.

À l'arrêt k, décoder l'accumulation des contributions des niveaux 1..k, déjà rééchantillonnée à 89 positions. Ne pas décoder le seul bloc k ni ajouter des codes fins nuls : l'indice de code 0 est un code valide à contribution non nulle.

Rapporter exactitude, validité avec respect des indices, cases exactes, taux de changement, corrections et dégradations à chaque niveau. Comparer les arrêts 8, 9, 10, 11 à 12 sur test et externe avec bootstrap apparié par puzzle. Quantifier les revirements après 9 et les derniers changements observés, sans les extrapoler au-delà de l'horizon 12.

Le contraste d'accélération de la grille JSON est secondaire. Ajuster des droites segmentées à titre descriptif seulement, avec abscisse niveau puis nombre cumulé de codes, pour tester la dépendance à la paramétrisation. Les fits sont sélectionnés sur ces mêmes courbes : leur baisse d'erreur n'est pas une validation indépendante d'une rupture.

## Interventions, sans nouvelle sélection

Sur les 128 premiers puzzles hard test fixés :

1. Contrôle sham : reproduire codes, décodages et sortie native avec le runner explicite.
2. Supprimer séparément les contributions des niveaux 8, 9, 10, 11 et 12, puis régénérer normalement la suite. La prédiction supprimée est calculée ; son apport résiduel est omis. Les perturbations de niveaux différents n'ont pas la même norme : ne pas assimiler leurs effets à une comparaison de sélectivité parfaitement appariée.
3. Lecture fractionnaire : décoder z8 + gamma × contribution9(native), 21 gamma dans [0,1]. Conserver les logits/probabilités de référence et les marges, l'argmax, ainsi que l'alignement des changements de valeurs. Il s'agit d'un contrôle de lecture à codes fixes, pas d'une trajectoire native ou d'une preuve d'artefact sans fonction.
4. Reprendre les niveaux 10..12 depuis les états fractionnaires gamma ∈ {0,0.5,1,1.5}. Gamma 1 est le sham, gamma 0 omet le résidu 9. Comparer les réponses et les codes régénérés.

Sur les 64 premiers de ces puzzles : bruit gaussien isotrope dans l'état cumulatif, avant le niveau 9 ou juste après. Deux doses et trois graines ; la même direction et exactement la même norme sont employées avant/après pour un puzzle. La norme vaut dose × norme(z8) dans les deux cas. Comparer la susceptibilité finale, sans affirmer qu'une faible sensibilité à ces perturbations prouve un bassin irréversible.

## Statistiques et livraison

Intervalles conditionnels aux poids et au protocole, 2 000 tirages par puzzle. Pour les bruits, agréger les trois graines au sein de chaque puzzle avant bootstrap. Les comparaisons multiples sont exploratoires et non corrigées. Aucun seuil n'est optimisé sur test.

Documenter les contre-exemples, un rapport complet français, une note anglaise, figures PNG/SVG/PDF, codes, prédictions et manifestes. Toutes les écritures restent dans ce nouveau sous-dossier. La piste de correction du pooling et toutes les campagnes précédentes restent intactes.

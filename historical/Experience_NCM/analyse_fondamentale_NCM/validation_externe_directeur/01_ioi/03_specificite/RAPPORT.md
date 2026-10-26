# D05 — Rapport de spécificité à site constant

## Mise en œuvre

Chaque source possède deux donneurs distincts fixés avant effets. Pour une tête donnée, les corruptions ont le même support — dernière position, même tête entière. Le suffixe mal apparié a été adapté au second donneur, puis appliqué à la première corruption, et réciproquement. Les normes ne sont pas automatiquement égales ; le contrôle réduit le changement de support sans l'éliminer comme explication géométrique générale.

## Résultat réservée

512familles,1024échanges. S=0.0379738, IC descriptif 98,75 % [0.02404077 ; 0.05402197]nat. La récupérationR=1.909953, IC descriptif 98,75 % [1.81382 ; 2.006202]nat reste une mesure distincte. Ajouter le seuilS>0,01 àD>0,05/R>0,01 fait passer le nombre de familles retenues de507à192. Aucun donneur n'a été choisi pour accroîtreS. Les familles avecS nul ou négatif sont conservées.

L'argmax vocabulaire est correct pour622/1024suites appariées et581/1024mal appariées. Une différence de confiance peut subsister sans changer cet argmax. Une suite mal appariée nuisible peut aussi augmenterS ;S ne prouve ni une représentation symbolique de l'erreur ni une réparation unique.

## Preuves et limites

Les identités de donneurs, mesures, états et suites sont conservées dans `../results/test_observations.json` et `../results/test_states.npz`. Le runner réservé ajoute normes effectives au point de coupure et normes projetées idéales. Le pilote initial ne les exportait pas séparément ; aucune collection absente n'est présentée comme conservée. L'ensemble de sources sert aussi de banque donneuse, ce qui limite l'interprétation des intervalles. L'évaluation sur des donneurs futurs globalement disjoints reste à réaliser pour une précision de prédiction de spécificité.

Les intervalles rééchantillonnent les familles en conservant leurs deux donneurs. Les sources sont aussi réemployées comme donneurs à d'autres familles : cette dépendance n'est pas entièrement traitée par ce bootstrap. Ces intervalles sont des approximations conditionnelles aux poids, aux sources et au plan de donneurs ; ils ne certifient ni une population de toutes les formulations IOI ni la variation entre réentraînements. Les deux réplications secondaires ne participent pas à la famille confirmatoire primaire. Les quatre intervalles principaux D,R,S,B utilisent le niveau98,75% annoncé ; ceux des autres quantités sont descriptifs, sans nouvelle revendication de contrôle de multiplicité global.

Les seuils de couverture locale sont des seuils d'effet descriptifs appliqués aux observations, pas un taux d'attributions mécanistes correctes. Les mesures DLA, self-repair centré et NLL n'ont pas le même estimand. Un désaccord ne prouve pas qu'une méthode antérieure se trompe.

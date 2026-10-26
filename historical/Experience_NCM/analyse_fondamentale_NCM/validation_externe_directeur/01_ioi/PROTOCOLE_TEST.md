# IOI : décision de passage et protocole d'évaluation réservée

25 octobre 2026. Écrit après les résultats de développement A2 et avant toute inférence réservée. Le gel horodaté `results/freeze_test.json` en fixe les fichiers. Il s'agit d'un gel local, pas d'une prérégistration extérieure.

## Passage de la porte

Développement :128familles,256remplacements de L9H9. D=2,461812932 nat ;127/128familles dépassent0,01nat de dommage moyen. La porte ≥0,05nat et ≥25%familles est franchie. L'audit retrouve huit interventions par une passe complète officielle avec hook, toutes bit à bit, et112vérifications de pertes/recomposition. Les incidents et versions initiale/A1 sont conservés ; A2 change seulement la projection de **mesure** DE enFP64, après détection de valeurs non finies. Aucun seuil scientifique n'a été changé. Toutes les sorties effectivement publiées du pilote sont finies.

Les résultats dev motivent l'ouverture du test, sans devenir des résultats réservés. R,S et blocage sont positifs sur développement ; ces signes ne sont pas exigés pour conserver une famille sur test. La justesse native, la norme de perturbation et la réussite de récupération ne servent à aucune exclusion.

## Plan et analyses maintenus

Les512familles de `data/selected.json` sont fixées. L9H9 reste primaire ; L9H6 et L10H0 sont deux réplications secondaires distinctes. Pour chaque tête :1536lectures natives A/B/Aplus ;1024rééchantillonnages par deux donneurs au même site ; cinq croisements et deux blocages ; plan E complet sur A/B et A/Aplus. La frontière est après attention de la couche de la tête, avant son MLP. Les ensembles backups/témoins sont restreints aux couches strictement aval. Aucun ajustement de composant sur test.

Les équations et marges du [protocole dev](PROTOCOLE_DEV.md) s'appliquent : D,R,I,S,B ; marges D0,05nat, R/S/B0,01nat ;20 000tirages bootstrap de familles avec les deux donneurs groupés. Les intervalles98,75% D/R/S/B sont les quatre intervalles principaux ; autres quantités et deux têtes secondaires restent descriptives. Les templates ne sont pas traités comme des milliers de réplications. L'audit d'entrées constate neuf templates dev et neuf test sans chaîne de template commune ; cette propriété vient des partitions officielles, pas d'une sélection sur sorties.

Les donneurs proviennent des mêmes listes que les receveurs : chaque source sert aussi à d'autres familles. Le bootstrap par receveur ne modélise donc pas toute la dépendance induite par cette banque. Les intervalles sont des approximations conditionnelles aux poids, aux sources et au plan de donneurs ; aucune garantie de couverture populationnelle ou d'indépendance stricte n'est revendiquée.

## Clarification préalable du blocage et de l'abstention

B mesure la perte sous blocage backups moins la perte sous blocage témoin. B>0 ne prouve pas seul que les backups sont nécessaires : le témoin pourrait simplement améliorer davantage. Une attribution restreinte corroborée requiert aussi que bloquer les backups augmente la perte par rapport à Y11. Cet effet absolu est exporté depuis le script de développement, avec son intervalle. Il ne sera ni occulté ni remplacé par B. Ce critère logique supplémentaire ne change aucune marge fixée.

On rapporte séparément : dommage utile, récupération NLL, appariement spécifique de la suite et effet sélectif du blocage. Les catégories analytiques ne constituent pas la vérité terrain de GPT-2. Une moyenne favorable ne classe pas chaque famille. Un self-repair positif n'est pas censé prédire, par définition, S>0 ; le désaccord éventuel entre ces deux objets ne sera pas nommé erreur de la mesure publiée.

## Scripts et vérification

`run_test.py` reprend le runner A2 sans changer ses interventions ; il ajoute seulement les normes effectives et idéales de perturbation aux observations. Le script dev gelé demeure inchangé. Les seuils/métriques sont ceux d'`analyse.py`. `verify.py` audite explicitement L9H9 ; les shams et recompositions de chaque source/tête secondaire sont vérifiés par leurs runners, sans appeler cela une réplication indépendante.

Les contrôles de finitude, les sommes FP32 et la parité native doivent tous passer. Un échec arrête la branche ; le résultat primaire n'est jamais remplacé par une tête secondaire plus favorable. Aucune nouvelle recherche de circuit, aucun entraînement, aucune sélection des exemples abîmés. Les temps sont des temps écoulés réels de session ; aucune mesure de latence matérielle isolée n'est revendiquée.

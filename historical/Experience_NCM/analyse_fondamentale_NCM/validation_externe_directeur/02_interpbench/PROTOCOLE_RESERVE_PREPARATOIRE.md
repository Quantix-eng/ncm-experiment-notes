# InterpBench — protocole de réserve préparatoire

25 octobre 2026. **Statut : préparation, aucun forward réservé.** Ce fichier ne déclenche pas une campagne ; des préconditions restent ouvertes. [Feuille de route](FEUILLE_DE_ROUTE.md).

## Sélection et données

Les cas 13, 18, 19 et 21 sont retenus indépendamment de leurs effets. Préparer 256 séquences sources uniques par cas avec les vocabulaires et longueurs officiels, sans consulter de sortie du modèle. Les variantes, donneurs et relectures d'une source restent groupés. Le fichier `data/reserved_inputs.json` contient les entrées et des identifiants de donneurs, aucune réponse. Il s'agit d'une collection finie à nouvelle graine ; elle n'est pas un échantillon uniforme sur tous les programmes ou toutes les séquences possibles.

Les donneurs i+1 et i+2 sont distincts du donneur i+128 utilisé pour une falsification nommée. Cela garantit une séparation dans une requête, pas une partition globale des sources ; avant tout ajustement d'une méthode, imposer une partition de familles disjointe ou une procédure préquentielle empêchant de consulter les sorties de falsification. Les données préparées ne suffisent pas, seules, à établir cette protection.

## Unités et oracle

Le domaine primaire est l'échange de valeurs natives sur têtes entières et MLP, aux unités de l'alignement publié. Inventorier toutes les variables éligibles selon le graphe avant effets ; conserver une liste gelée des variables alignées et non alignées. Une permutation de variables ou de labels peut servir de contrôle négatif de prédiction si elle est annoncée et ne change pas le réseau de référence.

L'oracle est le programme Tracr compilé avec les encodeurs publiés et la correspondance au réseau entraîné. Un oracle reconstruit à partir de notre diagnostic est interdit. Pour chaque intervention, il doit produire une sortie et un masque de positions définies. Les positions `None`, BOS et PAD ne sont pas transformées en classes arbitraires. Une différence de masque est conservée séparément de la perte sur les positions comparables.

Avant réserve, établir : identité des encodeurs ; accord natif ; échange natif au bon composant ; prédiction de programme sous échange aligné ; invariance sous échange non aligné ; granularité correcte ; absence de fuite de la banque de falsification. La parité officielle sur les deux tâches triviales satisfait seulement une partie de ces préconditions. Elle ne démontre pas l'alignement des quatre programmes nouveaux.

## Accès des méthodes et sorties

L'extension [EXTENSION_PREDICTIVE.md](EXTENSION_PREDICTIVE.md) définit le type commun d'affirmation. Les règles exécutables d'émission et d'abstention, le nombre de sondes et les budgets d'accès doivent encore être arrêtés, puis gelés avant la sortie réservée correspondante. Aucune classe générale ne sera injectée comme label oracle.

O, C et E sont des ensembles d'informations accessibles. C conserve Y00, Y10, Y11, Y01 et Y1wrong avec donneurs au même site ; E ajoute les changements d'entrée/état déclarés et les blocages d'adaptation. Les hypothèses sont écrites avant les réponses qui les falsifient. Les versions privées d'un contrôle ont les mêmes requêtes finales, et leurs différences de budget sont rapportées.

## Mesures prévues

Rapporter par tâche et type de variable : accord natif programme/réseau, IIA pour les variables alignées, SIIA pour les variables hors circuit, NLL sur positions définies si les scores ont ce sens, exactitude catégorielle et exacte de séquence, changements du masque, couverture des affirmations et fréquence des contradictions conditionnelle à cette couverture. L'IIA fournit l'évaluation de conformité ; elle ne devient pas artificiellement une méthode de prédiction à six classes.

Le résumé agrégé donne une moyenne équipondérée entre les quatre tâches et conserve leurs résultats séparés. Les donneurs, sites, positions et conditions ne sont pas des unités indépendantes. Aucun IC populationnel sur des milliers d'échanges ne remplace la variation entre tâches. Si des IC descriptifs sont ajoutés, leur niveau de regroupement et leur règle doivent être gelés avant les sorties.

## Gel et arrêt

Avant exécution, un manifeste final doit couvrir les programmes, compilateur, poids, alignements exportés, scripts, données, méthodes, prédictions et règles d'arrêt. Le présent fichier préparatoire ne prétend pas que ce gel existe déjà. Toute divergence réseau/programme est conservée avant d'évaluer les diagnostics. Si les préconditions échouent, arrêter la réserve, conserver l'échec et amender sur développement ; aucune recherche de tâche favorable dans les quatre sorties n'est autorisée.

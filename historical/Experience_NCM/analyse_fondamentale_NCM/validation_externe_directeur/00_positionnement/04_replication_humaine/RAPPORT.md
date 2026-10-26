# Rapport — dossier d'exécution extérieure préparé

25 octobre 2026. Aucun contact avec une personne ou une équipe externe, aucune revue humaine et aucune réplication indépendante ne sont annoncés comme réalisés.

## Pourquoi cette proposition reste distincte

Les données MIB, les hypothèses IOI et les programmes InterpBench ont des auteurs extérieurs. Cela réduit la circularité des objets testés. Si notre équipe implémente encore tous les calculs, règle tous les seuils et formule toutes les conclusions, l'évaluation de notre diagnostic reste interne. Le relancement de notre code par une autre personne mesure surtout sa reproductibilité informatique. La demande du directeur exige une implémentation propre et de nouvelles familles.

## Spécification à transmettre à l'implémenteur

Le périmètre minimal est le contraste de récupération NLL à état perturbé fixé, avec témoin de lecture :

`R = NLL(Y10,y) − NLL(Y11,y)`.

`Y10` et `Y11` utilisent le même état perturbé, le même modèle, le même lecteur et la même réponse, mais respectivement des contributions aval natives rejouées et régénérées. Les termes sont moyennés par source avant agrégation ; les donneurs et variantes ne sont pas assimilés à des réplications indépendantes.

L'implémenteur doit aussi calculer `damage=NLL(Y10,y)−NLL(Y00,y)` et `specificity=NLL(Y1wrong,y)−NLL(Y11,y)`. Le contrat impose que `Tother` soit adapté à un autre donneur du même site, puis rejoué sur le même état perturbé principal. Les conventions exactes du lecteur, de la normalisation et des frontières sont à recopier depuis le protocole expérimental gelé, avant exécution.

Le calcul des pertes repose sur des logits conservés avec `logsumexp` stable, sans clipping des contrastes, sans exclusion des cas défavorables et sans modification du nom correct. Une réimplémentation peut commencer par ce calcul à partir de tenseurs exportés, mais cela ne valide pas l'adaptateur de réseau. La version complète doit produire elle-même au moins un sous-ensemble d'interventions du forward.

## Contrôles d'implémentation requis

- Exécution sans intervention et sham reproduisant le modèle natif selon la tolérance annoncée pour le backend.
- Recomposition du résidu final à partir des contributions réellement sauvegardées ; ordre des additions et biais documentés.
- Vérification du placement de tête, de sa projection et de la position finale, avec le même nombre de tokens et la même convention BOS que le protocole.
- Vérification que figer une contribution n'autorise pas une autre passe à modifier silencieusement son cache.
- Vérification de l'identité de décomposition entre effet direct, variation des contributions aval et changement du lecteur, avec facteur LN natif pour l'effet direct.
- Calcul scalaire de quelques pertes et contrastes, distinct de leur agrégation vectorisée.

Les tolérances effectives doivent être choisies sur des contrôles numériques avant test, puis annoncées en unités de logits et de NLL. Une différence FP32 faible ne doit pas être convertie automatiquement en divergence scientifique. Les valeurs ne sont pas fixées a posteriori depuis l'accord observé.

## Données réellement nouvelles

Une nouvelle famille signifie un receveur et ses variantes ou un groupe contrôlé de templates selon une définition annoncée. Il faut distinguer : nouvelle réalisation lexicale d'un template déjà utilisé ; nouveau template ; nouvelle tâche ; nouveaux poids. Une nouvelle permutation d'une famille déjà évaluée ne devient pas une réplication indépendante.

Le choix des sources et des donneurs, les exclusions techniques et le nombre de familles doivent être fixés avant mesure des effets. Aucune sélection sur réussite native, dommage, compensation ou accord des deux implémentations n'est autorisée dans les métriques principales. Les cas non admissibles selon le protocole sont comptés et leur motif conservé.

Le protocole d'indépendance devra annoncer si l'implémenteur connaît les résultats précédents, s'il a vu notre code, et quels échanges ont eu lieu avant son gel. L'absence de cécité parfaite n'annule pas tout intérêt, mais elle ne doit pas être cachée.

## Résultats attendus du rapport extérieur

Le rapport doit rendre trois conclusions séparées :

1. **Accord d'implémentation** : conditions, erreurs maximales, cas divergents et causes identifiées ;
2. **Réplication d'effet** : dommage, récupération, spécificité et incertitude au niveau des familles nouvelles ;
3. **Accord d'interprétation** : quelles affirmations de nécessité ou de contribution sont corroborées, réfutées ou demeurent non identifiables.

Une divergence déclenche un amendement conservant les versions et résultats antérieurs. Le rapport extérieur ne doit pas être réécrit pour faire disparaître ses désaccords avec le diagnostic initial.

## Livrables préparés et reste à faire

Le présent dossier est prêt comme spécification de travail, avec le [rapport des méthodes](../02_methodes/RAPPORT.md) et les futurs protocoles de campagne. Il reste à désigner une personne, convenir de son périmètre scientifique, sceller son code, créer ses données et exécuter ses tests. Aucun délai réel ni engagement extérieur n'est inféré des estimations du directeur. Cette étape ne peut être déclarée achevée par le seul travail de sous-agents.

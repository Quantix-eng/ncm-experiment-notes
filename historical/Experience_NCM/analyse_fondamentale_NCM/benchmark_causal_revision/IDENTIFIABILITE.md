# Ce qui est identifiable dans les témoins de calibration

Note de portée rédigée après l'exécution du 25 octobre 2026. Les équations, scripts, critères et observations gelés restent inchangés. Cette note explicite les conséquences des équations et le régime d'accès réellement appliqué ; elle n'ajoute ni nouvelle expérience ni critère de réussite.

## 1. Une limite d'observation exacte

Pour une source donnée, les programmes de la bibliothèque ont la même sortie native finale `s` et le même lecteur temporel `Y_t=s+e_t`. Les quatre vecteurs et leurs argmax sont donc identiques entre programmes. Une fonction de ces seules observations doit produire la même réponse pour tous ces programmes. Si elle force une classe unique, elle est nécessairement fausse pour les autres classes.

Une abstention n'est donc pas un échec d'analyse dans ce régime. Inversement, retourner toutes les classes sans jamais réduire cet ensemble ne doit pas recevoir le même crédit qu'une attribution identifiable. C'est pourquoi le protocole rapporte à la fois couverture, taille d'ensemble et exactitude des décisions uniques. Les fréquences de classes de notre collection ne sont pas un prior naturel sur les mécanismes des réseaux.

## 2. Compensation et recalcul à entrée constante

Dans les témoins adaptatifs, l'état natif contient `a=s(x)` et `b=s(x)-v`. Les interventions du régime C changent seulement `b`, jamais `a` ou `x`. Pour **toute** valeur du contenu perturbé `b'` :

`C_etat(a,b') = a-b' = s(x)-b' = C_entree(x,b')`.

De même, pour tout coefficient alpha :

`C_mixte(x,a,b') = alpha*a+(1-alpha)*s(x)-b' = s(x)-b'`.

Les suites sont donc égales, pas seulement leurs effets ou leurs pertes. Les sorties des quatre croisements et des témoins mal appariés sont identiques. Cette équivalence ne dépend pas du nombre de puzzles : augmenter l'échantillon dans ce même régime ne sépare pas ces programmes. Elle demeure vraie si l'on observe les vecteurs des suites tout en conservant ces interventions et l'ancre native inchangée.

Une récupération spécifique peut ainsi être établie fonctionnellement sans que sa cible soit attribuable à l'entrée ou à l'ancre. Le recalcul et la compensation ne sont pas des explications mutuellement exclusives.

## 3. Pourquoi changer l'entrée et l'état apporte une distinction

À état A fixé, remplacer l'entrée A par B donne respectivement :

- compensation depuis l'état : sortie `s_A` ;
- recalcul oracle : sortie `s_B` ;
- mélange : sortie `alpha*s_A+(1-alpha)*s_B`.

L'échange réciproque de l'état à entrée A fixée donne `s_B`, `s_A` et `alpha*s_B+(1-alpha)*s_A`. Une paire dont les réponses sont distinctes est donc nécessaire à ce diagnostic. La variante d'entrée à réponse inchangée sert de contrôle, mais ne remplace pas une telle paire.

Dans cette bibliothèque et avec ces équations exactes, les sorties séparent les programmes. Hors bibliothèque, d'autres programmes peuvent reproduire les mêmes sorties sur les requêtes finies : « mélange » devient alors « mélange ou mécanisme non modélisé ». Une faible distance d'un réseau à l'une de ces signatures ne prouve pas l'identité de son programme interne.

## 4. R et I positifs ne suffisent pas à une spécificité de perturbation

Soit `k=||d0||²/m`, où m est le nombre de composantes de score. Le programme de réponse fixe détecte un écart à l'état natif et ajoute toujours `-d0`, sans distinguer les perturbations non nulles. Pour `delta=d0`, ses pertes sont :

`L00=0`, `L10=k`, `L11=0`, `L01=k`.

Il possède donc `R=k>0` et `I=2k>0`. La récupération est utile pour cette intervention. Mais une suite produite en réponse à une corruption sur un autre support est la même : `Y1wrong=Y11`, d'où `S=0`. La correction ne dépend pas de l'identité de la corruption.

Ce contre-exemple borne l'interprétation des contrastes. Il n'invalide pas leur mesure ni une conclusion correctement limitée à l'intervention donnée ; il exclut leur emploi comme preuve générale et suffisante d'une adaptation à l'identité de la perturbation. Le critère de l'étude NCM historique n'est pas réécrit à la lumière de ce contrôle.

## 5. Portée précise des accès effectivement exécutés

Le contrat appliqué par `methods.project` est :

- O : les quatre lectures discrètes natives ;
- C : ces lectures et les pertes/contrastes issus des sorties croisées aux quatre perturbations ;
- E : C, puis les distances de sortie après suppression de suite et échanges réciproques entrée/état, et le contrôle invariant.

La référence est utilisée par l'évaluateur pour produire ces pertes et distances. Aucun nom de programme, coefficient de mélange, nombre d'appels ou état d'ancre n'est donné au diagnostic. Les scores continus, suites et états stockés dans `raw_vectors.npz` sont des traces **d'audit** ; leur sauvegarde ne les rend pas accessibles dans ces projections. Le plan naturel à 12 conditions est exécuté et audité ; le diagnostic E utilise les distances correspondant aux échanges réciproques, pas une recherche sur les 12 sorties.

Il faut en particulier distinguer les deux ambiguïtés présentées : la compensation et le recalcul ont des suites exactement égales à entrée fixe, tandis que la relecture oracle et la tolérance n'ont que les mêmes sorties croisées dans C. Leurs suites natives sont respectivement `0` et `v`. Si l'on ajoutait l'observation de ces vecteurs de suites au contrat C, ces deux programmes seraient déjà séparables dans la bibliothèque annoncée. L'abstention `{relecture,tolérance}` vaut donc pour la projection C exécutée, pas pour un accès arbitraire à toutes les traces.

Ces restrictions permettent de comparer des budgets d'observation, mais ne sont pas une protection informatique : le code et les classes sont publics. Une future évaluation d'outils d'interprétabilité disposant d'un accès aux activations devra déclarer un régime supplémentaire et recalculer les classes d'équivalence pertinentes. Les chiffres de cette calibration ne peuvent pas être transférés tels quels à ce régime plus riche.

## 6. Limite statistique sur un système appris

Les égalités ci-dessus sont algébriques dans les témoins. Sur un modèle appris, un intervalle contenant zéro ne démontre ni absence d'effet ni équivalence. Des marges d'équivalence, des seuils d'effet utiles et des budgets de précision doivent être fixés avant les nouvelles observations. Une méthode ne devra pas forcer une classe dès que l'un des contrastes semble favorable.

Les puzzles à deux solutions fournissent une réponse contrôlée et un changement sémantique vérifiable. Ils ne garantissent ni distribution native des états échangés, ni identification de circuits internes, ni unicité d'une explication mécaniste.

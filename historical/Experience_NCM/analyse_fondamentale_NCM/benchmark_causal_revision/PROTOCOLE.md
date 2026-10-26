# Protocole d'évaluation de l'attribution des révisions — version 0.1

25 octobre 2026. Étape 2 du programme de recherche. Cette version est un protocole et un banc de calibration sur systèmes construits, pas un benchmark de référence validé. Les rapports sont exclusivement en Markdown. Les résultats et limites d'exécution seront consignés dans `RAPPORT.md`.

## 1. Question et objets évalués

Une méthode d'analyse distingue-t-elle une amélioration de lecture, une tolérance à une perturbation, un recalcul ancré dans l'entrée et une compensation dépendant de l'état perturbé ? L'objet évalué est **la méthode d'attribution**, pas la précision finale du système. La vérité de construction des témoins vient de leurs équations et de leur code ; aucune étiquette mécaniste n'est assignée aux NCM appris.

Ces propriétés ne sont pas mutuellement exclusives. Un recalcul depuis l'entrée peut réaliser une compensation résiduelle spécifique. Les noms des témoins désignent des programmes distincts dans une bibliothèque finie ; ils ne forment pas une taxonomie universelle du raisonnement.

Un résultat admissible est un ensemble de témoins compatibles avec les observations, ou « non identifiable dans ce régime d'accès ». L'identification exacte n'est exigée que lorsque les interventions autorisées distinguent les témoins de la bibliothèque. Aucune suite finie de requêtes ne prouve l'unicité d'un mécanisme parmi tous les programmes possibles.

## 2. Modèle causal des témoins

Pour l'entrée `x`, `s(x)` est un vecteur one-hot de réponse fourni par un oracle de tâche. L'état au point d'intervention comprend une ancre `a` et un contenu modifiable `b`. `v` est un résidu fixe non nul ; `t` désigne la continuation. Toutes les opérations sont en FP64 sur CPU ; ces états ne sont pas des latents NCM. Les six programmes sont :

- **Relecture oracle** : `a=s(x)`, `b=s(x)-v`, `C(x,a,b)=0`, `D(a,b,t)=a`. La réponse est disponible avant les lectures. La continuation ne contribue pas à la sortie.
- **Tolérance** : `a=b=s(x)-v`, `C(x,a,b)=v`, `D(a,b,t)=a+t`. Le canal `b` perturbé n'est pas utilisé ; la continuation fixe reste nécessaire à une restitution exacte. Tolérance ne signifie pas absence de tout calcul.
- **Compensation depuis l'état** : `a=s(x)`, `b=s(x)-v`, `C(x,a,b)=a-b`, `D(a,b,t)=b+t`. La correction utilise l'ancre conservée ; un changement d'entrée à état fixé n'est pas consulté.
- **Recalcul oracle depuis l'entrée** : même état initial, `C(x,a,b)=s(x)-b`, `D(a,b,t)=b+t`. Le résidu compense aussi la perturbation ; sa cible provient de l'entrée. Ici `s(x)` est un accès oracle à la réponse correcte de l'entrée, pas un réseau entraîné ni une preuve d'algorithme Sudoku interne.
- **Réponse fixe à la détection d'une perturbation** : même état initial, `C=v` si `b=a-v`, sinon `C=v-d0`. Le système détecte un écart, mais réutilise la même correction `-d0` quelle qu'en soit l'identité. Il peut donner une récupération parfaite pour une corruption particulière et échouer pour une autre. Ce témoin protège contre une inférence de spécificité à partir d'une seule dose.
- **Mélange réservé** : `C=alpha*a+(1-alpha)*s(x)-b`, avec `alpha` dans `{0.25,0.50,0.75}`. Les composantes état/entrée sont combinées. Ces mélanges sont absents du développement ; leur famille et la règle d'abstention sont néanmoins annoncées avant l'évaluation.

Les témoins utilisent tous une réponse oracle pour leur construction. « Oracle » dans le premier nom ne signifie pas que les autres seraient des solveurs déployables dépourvus de cet accès. Les programmes sont volontairement simples et ne sont pas des familles apprises indépendantes.

## 3. Trajectoires observationnelles appariées

Les quatre lectures natives utilisent le même décalage de lecteur : `D_t=D_final+e_t`, avec `e_t` successivement `1.25*d0, 0, 1.25*d0, 0`. Le vecteur `d0` transfère une unité du chiffre de référence vers un autre chiffre sur une position éligible tirée avant les interventions. Les lectures discrètes sont des argmax avec la même convention de départage.

Tous les témoins d'une même source présentent donc exactement la même trajectoire, y compris erreur, correction, dégradation et correction finale, et la même précision finale parfaite. La ressemblance ne dépend pas d'un ajustement après examen des réponses. C'est une construction adversariale analytique ; démontrer qu'elle trompe une règle observationnelle est un contrôle de logique expérimentale, pas une découverte empirique sur les réseaux.

## 4. Interventions autorisées et régimes d'accès

**Régime O — observations seules.** Les quatre sorties et la référence sont disponibles. Aucun état, nom de témoin, numéro de classe, temps d'exécution ou journal d'appels n'est accessible. Les six témoins présents sont observationnellement indiscernables par construction ; une méthode scientifiquement prudente doit conserver l'ambiguïté.

**Régime C — continuations croisées à entrée fixe.** Une perturbation ajoute `delta` à `b`, sans modifier `a` ni l'entrée. `T0=C(x,a,b)` et `Tdelta=C(x,a,b+delta)` sont calculées et sauvegardées séparément. Les sorties sont :

- `Y00=D(a,b,T0)` ;
- `Y10=D(a,b+delta,T0)` ;
- `Y11=D(a,b+delta,Tdelta)` ;
- `Y01=D(a,b,Tdelta)` ;
- `Y1wrong=D(a,b+delta,Tother)`, où `Tother` est adaptée à une autre perturbation, de même norme et de support distinct.

Deux supports distincts sont tirés à partir des cases éligibles, avec signes `+` et `-`, soit quatre perturbations par source et programme. Pas de sélection sur dommage observé ; le signe négatif peut modifier les scores sans changer leur argmax. Un sham `delta=0` doit reproduire `Y00`. Les résidus, sorties continues et lectures discrètes sont conservés ; un changement de résidu ne vaut pas amélioration.

La perte primaire de calibration est la MSE entre scores et réponse one-hot, moyennée sur toutes les composantes. Elle n'est pas une NLL de modèle et aucune signification probabiliste n'est attribuée aux scores négatifs. Les argmax constituent une mesure secondaire. Les contrastes sont `R=L10-L11`, `I=L10+L01-L11-L00` et `S=L1wrong-L11`. Une réponse fixe à une seule corruption peut avoir `R>0` et `I>0` ; une comparaison de perturbations est requise pour tester sa spécificité. Les systèmes de compensation et de recalcul ont les mêmes réponses dans ce régime, ce qui doit entraîner une abstention sur leur distinction.

**Régime E — changements d'entrée, échanges d'état et ablations.** On complète C par : suppression de toute la continuation ; changement de `x_A` en `x_B` à état `z_A` fixé ; échange de l'état `z_A` avec `z_B` à entrée fixée ; et plan naturel `2 entrées × 2 états × 3 suites` (suite native A figée, suite native B figée, suite recalculée depuis la combinaison entrée/état). Les 12 conditions sont conservées. Une modification d'entrée préservant la réponse est également testée. Tous les appels ont le même chemin d'interface, sans révéler le nom du témoin. Les coûts d'oracle et d'intervention sont documentés, sans prétendre être appariés à ceux d'un réseau.

La suppression de continuation différencie ici relecture et tolérance ; le changement d'entrée différencie ici compensation depuis l'ancre et recalcul oracle. Cela établit une distinction fonctionnelle pour cette bibliothèque, pas une preuve de l'origine cognitive des réponses. Les états échangés restent artificiels, même entre puzzles naturels proches.

## 5. Puzzles Sudoku contre-factuels

Pour chacune des 16 familles Sudoku nouvelles :

1. Générer une solution complète par backtracking aléatoire. Exclure les clés normalisées D4/permutation des chiffres déjà présentes dans les corpus historiques locaux ou déjà retenues. Cela ne couvre pas toutes les isomorphies Sudoku ni un entraînement inconnu.
2. Retirer des indices en ordre aléatoire, en conservant un puzzle tant qu'il a au plus deux solutions ; retenir le premier état ayant exactement deux solutions. Le comptage utilise une limite de **trois**, afin que « deux » signifie exploration exhaustive sans troisième solution.
3. Contrôler indépendamment les deux solutions par MRV et Algorithm X ; vérifier leurs contraintes et leur respect des indices.
4. Sur une case vide `j` où elles diffèrent, construire `P_A=B+[j=S_A[j]]` et `P_B=B+[j=S_B[j]]`. Chacun doit être unique selon les deux solveurs. Ces puzzles ont le même nombre d'indices et diffèrent exactement en une case.
5. Construire `P_A_plus` en ajoutant un indice vrai sur une autre case vide : réponse inchangée, unicité revérifiée par les deux solveurs. Ce contrôle change le nombre d'indices ; il ne supprime donc pas ce facteur de confusion. Un futur protocole pourra compléter par déplacement d'indice à cardinalité constante.
6. Les positions utilisées pour les perturbations sont tirées parmi les cases restant vides dans A et B ; les métriques de changement de réponse excluent l'indice ajouté `j`. Toutes les variantes, symétries et états d'une même famille restent dans la même partition.

Les sorties de programmes construits sur ces réponses ne sont pas des résultats de résolution par NCM. Les artefacts conservent B, A, B, A_plus, les solutions et les comptes complets sous la limite 3, le nombre d'indices et les distances de réponse hors indice.

## 6. Partitions, méthodes et métriques

Développement : 12 familles de contraintes d'égalité et 12 de parité paire. Évaluation structurelle réservée : 12 familles de permutations et 16 familles Sudoku. Pour les familles analytiques, les entrées sont des clés de tâches à réponse contrôlée ; ce ne sont pas des puzzles de difficulté comparable. Les clés de sources sont uniques ; les invariances introduisent des variantes d'entrée de même réponse, pas de nouvelles unités indépendantes. Les mélanges sont exécutés uniquement sur les familles réservées. Aucun réglage de seuil n'est permis sur ces sorties.

Les méthodes de calibration sont fixées avant génération :

- Règle naïve : attribuer la trajectoire avec correction à une compensation depuis l'état. Elle force une attribution unique.
- Règle observationnelle prudente : retourner tous les témoins de la bibliothèque compatibles avec une trajectoire commune.
- Règle croisée : examiner dommage figé, récupération aux quatre doses et écart aux suites mal appariées ; conserver `{relecture,tolérance}` ou `{compensation,recalcul,mélange}` lorsque les sorties ne les séparent pas ; isoler la réponse fixe lorsqu'elle varie avec la dose sans spécificité.
- Règle interventionnelle complète : ajouter l'ablation de suite et les changements entrée/état. Elle distingue les programmes dans la bibliothèque annoncée ; une réponse intermédiaire entre A et B est étiquetée mélange dans cette bibliothèque. Hors bibliothèque, ce diagnostic doit devenir « mélange ou mécanisme non modélisé », pas une découverte certaine de mélange.

Les règles ne reçoivent ni classe, ni paramètres `alpha`, ni états internes de construction. Leurs entrées sont les sorties et les pertes autorisées. Les identifiants sont opaques et indépendants de la classe. La bibliothèque et son code sont publics : il ne s'agit pas d'un concours à test secret ni d'une évaluation indépendante.

Rapporter séparément par partition, famille et programme : couverture de la vraie classe par l'ensemble annoncé ; taille de cet ensemble ; fraction d'attributions uniques ; précision parmi ces attributions ; score d'identification exacte lorsque permis ; et violations des équivalences attendues. Un ensemble contenant toutes les classes ne doit pas gagner par seule couverture. Aucun classement ne doit utiliser la précision finale, identique par construction.

Les résultats numériques de cette calibration sont des comptes sur une collection finie et construite. Pas de p-value ni d'IC populationnel sur les milliers de requêtes supposées indépendantes. Le niveau pertinent de généralisation est la famille de tâche et de mécanisme ; quatre familles et un auteur ne suffisent pas à établir une performance externe.

## 7. Gel, audit et conditions d'arrêt

Avant génération des données, épingler ce protocole, `protocol.json`, les scripts de génération, contrôles, diagnostic, exécution et audit, les deux solveurs et leurs imports, les corpus lus et les manifestes historiques. Ce gel est local, sans registre externe. Une défaillance logicielle entraîne l'arrêt ; toute correction après gel conserve l'ancienne version et nécessite un amendement horodaté, sans modification silencieuse des critères.

L'audit doit recomposer les équations avec une implémentation distincte, vérifier les trajectoires appariées, les shams, les signes des contrastes sur contrôles, les cas d'indiscernabilité, les 12 croisements et les partitions. Vérifier les comptes Sudoku avec deux solveurs et cap 3. Les assertions logicielles ne sont pas des réplications scientifiques indépendantes.

Critères d'arrêt scientifique : si le protocole ne distingue rien au-delà de la précision finale, si les étiquettes apprises nécessitent nos hypothèses sur NCM, ou si des observations équivalentes sont forcées vers des mécanismes différents, réviser le dispositif avant toute campagne apprise. Si les témoins passent, ne conclure qu'à la calibration interne du protocole. L'étape 2 s'arrête au rapport ; la théorie de Wyner reste à l'étape 3.

## 8. Passage ultérieur aux systèmes appris

L'adaptateur NCM devra déclarer un point de coupure, reconstruire les suites résiduelles, conserver les quatre croisements et les témoins mal appariés, contrôler la parité native et les caches, puis réaliser le plan entrée/état/suite sur les paires naturelles. NLL sur cases vides communes, exactitude, indices et format seront rapportés séparément. Les seuils d'effet, doses, budgets, partitions et contrôles de distribution devront être gelés avant ces nouvelles sorties. Aucune classe mécaniste certaine ne sera déduite d'un score isolé ; l'attribution restera fonctionnelle et conditionnelle aux interventions.

Un benchmark susceptible de servir de référence devra ensuite inclure au moins une famille apprise distincte et une évaluation extérieure. Ces campagnes ne sont pas implicites dans la présente validation logicielle et ne seront pas présentées comme exécutées.

## 9. Positionnement bibliographique vérifié

[MIB, ICML 2025](https://mib-bench.github.io/) évalue notamment la localisation de circuits et de variables causales et inclut un modèle à circuit connu. Le présent dispositif vise plus étroitement l'attribution d'un changement temporel en présence d'un lecteur progressif. Cette différence d'objectif ne suffit pas à établir une nouveauté ou une priorité.

[Tyen et al., ACL 2024](https://aclanthology.org/2024.findings-acl.826/) distinguent la localisation d'une erreur et la capacité à la corriger lorsqu'une indication est fournie. Ici aussi, l'origine des informations accessibles à chaque programme doit être explicite. Il ne s'agit pas d'une réplication de leurs expériences.

Ces deux sources primaires ont été consultées le 25 octobre 2026 ; cette vérification ciblée n'est pas une revue systématique de littérature.

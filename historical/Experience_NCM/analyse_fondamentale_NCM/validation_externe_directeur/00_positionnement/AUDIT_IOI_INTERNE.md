# Relecture interne de l'implémentation IOI

25 octobre 2026. Relecture du premier pilote de développement, avant ouverture des sorties réservées. Lecture des scripts et du protocole, sans modification d'un fichier gelé et sans nouveau calcul du modèle par ce relecteur. Ce travail est une vérification interne par un autre agent, sans indépendance scientifique humaine.

## Périmètre et version relue

Fichiers examinés dans `01_ioi` : `PROTOCOLE_DEV.md`, `runtime.py`, `run.py`, `analyse.py`, `verify.py`, avec `support.py` et `prepare.py` pour les conventions et les contrôles de données.

Empreintes de la version examinée :

- protocole : `a5b6cecd344f7c91be0db7fc4d3c1a8bf6603785a6ce3763ee643227d5c3c4d4` ;
- runtime : `5920b77f7e77755d95f8bd724a63db3bf5c212b7bc06560871d943adc8edc3f3` ;
- runner : `589d47cd70048fbb8d53f3651e6cb5c8ca40525727f9a0e1eb1da59f2587441a` ;
- analyse : `9ff937114bad5f44260a7f71dfe0359f9e3f91fce3e63fde5ae0ee361236d336` ;
- vérificateur : `75cd0b6de71572e928a7da74be950726d1b09a15340a8282c2ff5feb8e5bac67` ;
- support : `eaf015c2ea90292281ecd508b8aaf4b29fd41165cac1ba0ec6fb120a1232e2a6` ;
- préparation : `d7390b976558681ef58776d40367d8c34c422abb2adaf62dd845263cbd0eff75`.

Les liens ci-dessous désignent le dossier de campagne ; si une correction amendée remplace une version courante, les empreintes ci-dessus définissent la version réellement relue.

## Résultat de la lecture statique

Aucune erreur centrale de site, d'orientation de projection, de composition des suites ou de choix des backups strictement aval n'a été identifiée. Cette conclusion de lecture ne certifie pas l'absence de bug. Les contrôles numériques restent nécessaires et l'export initial a ensuite révélé une valeur non finie, comme indiqué en fin de rapport.

### Intervention et frontière

`boundary` remplace les 64 composantes d'une tête au dernier token avant `c_proj`, puis utilise la projection officielle et son biais une seule fois. Cela correspond au resampling annoncé. La projection GPT-2 de type Conv1D utilise la convention entrée × sortie : la tranche `weight[h*64:(h+1)*64]` est correcte pour projeter la différence de cette tête.

La frontière est après l'attention totale de la couche 9, avant son MLP. Les autres sorties de têtes de couche 9 restent natives ; le MLP de couche 9 et les couches suivantes peuvent s'adapter. Le gel des tokens antérieurs dans la continuation est cohérent avec une intervention au dernier token d'un Transformer causal. L'assertion qui compare les tokens antérieurs au cache natif est pertinente. [Runtime](../01_ioi/runtime.py)

### Suites croisées

La suite commence par le MLP de la couche d'intervention, puis alterne attention et MLP des couches suivantes. `replay` reproduit cet ordre d'addition ; les shams et l'égalité de recomposition contrôlent le chemin natif. Les conditions du runner utilisent bien le même état perturbé pour Y10, Y11 et Y1wrong. La suite mal appariée provient de l'autre activation donneuse au même site. [Runner](../01_ioi/run.py)

Les décalages 37 et 73 sont distincts et non nuls modulo 128 ou 512. Ils donnent donc deux donneurs différents du receveur dans les effectifs prévus, sans sélection selon les effets. Leurs amplitudes ne sont pas appariées ; le protocole ne prétend pas qu'elles le sont. Des noms partagés entre donneur et receveur ne sont pas un bug et doivent rester dans l'évaluation annoncée.

### Blocage et contrôle

Le blocage substitue les activations pré-projection natives des têtes sélectionnées dans les couches strictement suivantes. Les autres têtes et MLP restent adaptatifs. Pour L9H9, l'ensemble de six backups et le témoin de six têtes sont bien répartis 4+2 sur les couches 10 et 11. Pour la réplication L10H0, le filtrage conserve deux backups et deux témoins en couche 11. Les têtes simultanées de couche 9 ne sont pas traitées comme réponses adaptatives à la substitution de sortie L9H9.

Cela bloque les contributions des composants annoncés, pas toute compensation possible. Le témoin contient des composants potentiellement actifs ; il ne doit pas être décrit comme neutre.

### Effet direct

`direct_effect` utilise le résidu final natif, sa variance et epsilon de LN, le gain appris, le centrage du vecteur caché et l'unembedding centré sur le vocabulaire. La différence donneur−receveur réellement projetée est utilisée, sans approximation par la suppression de la contribution native. Ces conventions correspondent à la mesure annoncée. Le facteur est reconstruit en FP64 à partir du résidu FP32, tandis que les logits sont calculés par le modèle FP32 : il faut conserver ce petit écart numérique dans l'audit.

## Zéro self-repair avec suite et facteur LN figés

La prédiction est correcte **en arithmétique réelle** avec cette frontière. Soit `q0` le résidu après l'attention native totale, `d` le changement projeté d'une tête et `T0` la suite native. Alors :

`q1=q0+d` ; `r00=q0+sum(T0)` ; `r10=q1+sum(T0)` ; `r10−r00=d`.

Fixons le dénominateur natif `sigma0`, mais conservons le centrage linéaire de la LN sur le résidu courant. Pour le logit correct centré, le lecteur est affine :

`f0(r) = dot(w_y, gamma*P(r)/sigma0 + beta) + constant`.

Il suit exactement que `f0(r10)−f0(r00)=DE0(d)`, donc `SR_frozen_scale = 0`. La présence des autres têtes dans la somme native n'invalide pas cette identité : elles sont inchangées et leur projection est linéaire.

**Une égalité bit à bit n'est toutefois pas attendue pour ce test de formule.** Le runner recalcule la projection totale de 768 composantes, puis l'addition au résidu et les additions de suite en FP32. `delta_DE` provient d'une projection de différence de 64 composantes, avec d'autres arrondis. Le déplacement effectif `r10−r00` peut donc différer légèrement du déplacement idéal `W_O*(head_donor−head_native)`. En outre, le facteur LN reconstruit en FP64 diffère à l'arrondi de celui utilisé dans le forward FP32.

En revanche, un sham identique et une recomposition utilisant exactement les opérations natives peuvent légitimement être exigés bit à bit. Ces deux sortes de contrôles ne doivent pas être confondues.

## Audit supplémentaire proposé, distinct du gel initial

Le vérificateur actuel n'implémente pas la séparation explicite entre facteur LN final et contributions aval. Un audit post-hoc peut être ajouté dans un nouveau script, avec statut et empreinte propres, sans changer le pilote gelé ni ses critères. Il devra :

1. Utiliser les résidus finaux stockés r00, r10 et r11 ; ne pas rechercher de nouveaux cas, têtes ou donneurs.
2. Reconstruire en FP64 le lecteur à facteur `sigma0` fixé, en conservant le centrage sur le résidu courant.
3. Comparer le SR à facteur figé à `DE0(r10−r00)−delta_DE_stocké`. Ce terme mesure la différence entre déplacement effectivement obtenu et déplacement projeté idéal, avec les conventions numériques retenues.
4. Vérifier l'identité algébrique entre SR total, variation des contributions aval sous facteur figé et variation due au lecteur libéré.
5. Rapporter les erreurs absolues et leurs distributions, avec référence à l'erreur des logits et projections, sans choisir une tolérance à partir d'une interprétation favorable des effets.
6. Rejouer éventuellement un petit sous-ensemble avec un hook complet et gel du facteur pour vérifier le lecteur. L'algèbre sur des tenseurs stockés seule ne valide pas un nouvel adaptateur.

Le lecteur naturel utilisé par Y10 recalcule la LN finale. Un self-repair non nul sous suite figée et LN libre est donc possible et n'est pas une erreur. Il doit être attribué au changement de lecture, sans être converti en preuve d'adaptation de bloc.

## Réserve d'interprétation : le contraste différentiel B

`B = L_block_backups − L_block_controls` est correctement codé. Il ne suffit pas, seul, à montrer que bloquer les backups réduit la récupération par rapport à la continuation adaptée libre. Exemple : `L11=2`, `L_backups=1`, `L_controls=0,5` donne B positif alors que les deux blocages améliorent la perte.

Le runner conserve déjà `L_backups−L11`. Une revendication de corroboration doit exiger que ce déplacement absolu soit nuisible, en plus du contraste différentiel et d'une récupération utile. À défaut, B doit rester une différence entre interventions, sans conclusion de nécessité des backups. Cette clarification doit être inscrite au protocole de test avant ses sorties, sans changer silencieusement les marges du pilote. [Analyse](../01_ioi/analyse.py)

Les comptes de couverture locaux portent sur des contrastes observés au-dessus de marges ; ils ne sont pas une précision contre une vérité de mécanisme. Les versions sans contrôle figé ou sans spécificité sont des ablations du protocole, pas des concurrents indépendants.

## Réserve statistique : donneurs partagés

Les deux essais d'un receveur sont correctement regroupés avant bootstrap. Cependant, chaque source sert aussi de donneur à deux autres receveurs. Il subsiste donc une dépendance entre contrastes de différentes familles via la banque d'activations. Le libellé `conditionally independent source families` de l'analyse test est trop fort.

Les intervalles doivent être présentés comme une approximation conditionnelle au modèle, aux sources retenues et au schéma de donneurs, sans couverture démontrée de la variabilité du choix de donneurs. Une campagne ultérieure peut séparer banque donneuse et receveurs, ou annoncer le rééchantillonnage et le rejeu du plan conjoint. Ce point ne justifie pas de modifier le schéma gelé après inspection des effets.

De même, le compte Y00 sur 256 remplacements correspond à 128 sources natives répétées deux fois. Les comptes de justesse sont exacts comme comptes de requêtes ; ils ne représentent pas 256 sorties natives indépendantes.

## Portée réelle du vérificateur

Le vérificateur recalcule les pertes par une expression scalaire et compare un forward complet hooké à la reprise au point de coupure sur quatre sources, deux donneurs chacune. Il recompose les sept conditions sur ces cas et recalcule la porte sur toutes les familles. Ces contrôles sont utiles, mais la version relue est codée pour L9H9 et ne couvre pas automatiquement les futures réplications L9H6/L10H0. Elle ne vérifie pas les effets directs, les conditions E, ni les blocages par un second chemin de forward complet. Ces limites devront rester explicites ou être couvertes par un audit amendé.

## Arrêt logiciel communiqué pendant la relecture

Le responsable de campagne signale un échec à l'export JSON de la première exécution de développement : `allow_nan=False` a rejeté une valeur non finie, avant écriture des observations. Aucun test réservé n'a été ouvert. Les assertions de sham et de recomposition précédentes n'identifient pas la cause.

La lecture statique n'isole pas de source nécessaire de NaN dans les formules, lorsque tous leurs arguments sont finis. Une parité `torch.equal` n'établit cependant pas la finitude : des infinis identiques passent cette égalité. De même, toutes les conditions croisées ne sont pas comparées à des logits natifs. Un logit infini peut produire `inf−inf` dans le centrage ou dans la réduction NLL.

La conduite appropriée est un amendement logiciel : conserver la première version et son journal ; sauvegarder les tableaux bruts avant l'export ; détecter les chemins exacts de valeurs non finies ; contrôler la finitude des paramètres, états, logits et métriques correspondants ; arrêter avant toute conclusion scientifique tant que la cause n'est pas localisée. L'hypothèse d'une cause matérielle ou d'une propriété du modèle n'est pas justifiée à ce stade. Le présent audit n'anticipe pas le résultat de cette investigation.

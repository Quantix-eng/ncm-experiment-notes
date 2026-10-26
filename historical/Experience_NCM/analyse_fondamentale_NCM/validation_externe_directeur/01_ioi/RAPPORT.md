# GPT-2 small × MIB–IOI — rapport d'application

25 octobre 2026. [Feuille de route](FEUILLE_DE_ROUTE.md), [protocole dev](PROTOCOLE_DEV.md), [protocole réservé](PROTOCOLE_TEST.md), [amendements conservés](AMENDEMENTS.md). Rapports exclusivement Markdown, sans entraînement supplémentaire.

## État réellement atteint

L’évaluation primaire sur512familles réservées est exécutée. Le dommage matériel est désormais présent sur IOI, contrairement aux anciens cas peu dommageables. Cela rend la récupération testable dans ce contrat ; cela ne transforme pas une description fonctionnelle en vérité complète du mécanisme de GPT-2.

Le développement fixé avant inférence donne D=2.461813nat et 99.21875% des familles au-dessus de0,01nat. La porte pratique ≥0,05nat et ≥25%familles est franchie. Le sham est exact et huit interventions L9H9 ont été comparées bit à bit à la passe complète officielle avec hook avant ouverture du test. Aucun cas n'a été conservé ou exclu selon son dommage, sa justesse native ou sa récupération.

## Données, indépendance et modèle

MIB–IOI provient des [données publiques externes](https://huggingface.co/datasets/mib-bench/ioi), révision épinglée dans [la provenance](data/provenance.json). Les partitions officielles sont respectées :128sources dev et512test ; variantes A, s2_io_flip B et s1_io_flip Aplus groupées. Les étiquettes et la tokenisation ont été contrôlées avant sorties ; les corpus officiels lus comportent10000dev et1000test, notre sélection est128/512. Aucun des10000dev ou1000test n'est exclu par les critères techniques. L'audit relève neuf templates distincts dans chaque sélection et aucun template littéral commun aux deux. Les variantes d'une même famille ne sont pas de nouvelles unités indépendantes.

GPT-2 small local, révision607a30d783dfa663caf39e06633721c8d4cfcd7e. CPU FP32, un thread, eager attention, aucun cache KV mutable ; BOS50256 explicite et noms chacun à un token avec espace initial. Il ne s'agit pas d'une reproduction bit à bit de tout le pipeline officiel MIB. Les données, le choix des composants et les antécédents sont extérieurs ; l'implémentation, les métriques et les audits restent internes.

## Contrat causal et contrôles

La tête primaire est L9H9, dernière position. Sa sortie64dimensions est remplacée par la valeur native d'une autre source. La projection pleine officielle reste FP32. La frontière est après attention totale et avant le MLP de la même couche. La continuation recalcule ce MLP puis les blocs restants, avec les représentations des tokens antérieurs du receveur fixées à leurs valeurs natives. Elle a donc encore accès à un canal contextuel du receveur. L9H6 et L10H0 ont chacune leur frontière et restent secondaires.

Deux donneurs sont fixés par décalages37/73 à site identique. Y00,Y10,Y11,Y01,Y1wrong croisent état natif/corrompu et suites natives/adaptées/mal appariées. Les deux derniers bras de confirmation bloquent l'adaptation des backups publiés strictement aval, ou d'un témoin égal en nombre et stratifié par couche. Les têtes simultanées L9H0/H7 ne sont pas appelées « aval ». Les conditions E croisent A/B et A/Aplus avec état et suite, sans attribuer une classe d'architecture à partir d'une suite figée.

## Résultats de l'évaluation réservée primaire

512familles et1024remplacements. D,R,I,S,B sont en nats ; self-repair et delta_DE sont des effets de logit centré, à ne pas additionner aux NLL.

- `D` : 2.752369, IC descriptif 98,75 % [2.657156 ; 2.846061].
- `R` : 1.909953, IC descriptif 98,75 % [1.81382 ; 2.006202].
- `I` : 1.275902, IC descriptif 98,75 % [1.192133 ; 1.361652].
- `S` : 0.0379738, IC descriptif 98,75 % [0.02404077 ; 0.05402197].
- `B` : 0.2109682, IC descriptif 98,75 % [0.1886046 ; 0.2345672].
- `block_backups_minus_adapted` : 0.2510591, IC descriptif 98,75 % [0.2265358 ; 0.277048].
- `block_controls_minus_adapted` : 0.04009091, IC descriptif 98,75 % [0.03048847 ; 0.04968924].
- `self_repair` : 2.35486, IC descriptif 98,75 % [2.240201 ; 2.470372].
- `delta_DE` : -3.385907, IC descriptif 98,75 % [-3.486256 ; -3.283417].
- `ordinary_ablation_damage` : 0.8424164, IC descriptif 98,75 % [0.797105 ; 0.8907108].

Le dommage conventionnel est L11−L00 : il conserve le dommage résiduel après toute adaptation. D−R lui est égal par définition. Une récupération positive ne signifie donc ni retour exact au natif ni amélioration de l'entrée non perturbée.

## Justesse : deux noms et vocabulaire entier

- `Y00` : deux noms 1024/1024 ; vocabulaire entier 952/1024.
- `Y10` : deux noms 698/1024 ; vocabulaire entier 59/1024.
- `Y11` : deux noms 1015/1024 ; vocabulaire entier 622/1024.
- `Y01` : deux noms 1024/1024 ; vocabulaire entier 1001/1024.
- `Y1wrong` : deux noms 975/1024 ; vocabulaire entier 581/1024.
- `block_backups` : deux noms 1007/1024 ; vocabulaire entier 492/1024.
- `block_controls` : deux noms 1016/1024 ; vocabulaire entier 599/1024.

Y00 est répété pour les deux donneurs :1024lignes Y00 ne sont pas autant de sources natives différentes. La réussite parmi deux candidats ne suffit pas à garantir le bon token dans tout le vocabulaire. Les sorties Y01 ou mal appariées sont conservées même lorsqu'elles améliorent un score.

## Couverture, abstention et confirmation

La règle d'effets locaux D>0,05/R>0,01 conserve507/512familles ; ajouter S>0,01 en conserve192/512. Ces comptes ne sont pas une précision de classification : ils décrivent combien de cas satisfont les contrastes observés. La version sans contrôle figé ne peut établir la nécessité de l'adaptation. La version sans appariement ne peut établir sa spécificité.

Le blocage doit être lu avec ses deux effets absolus, pas seulement B. Une différence backups−témoin positive ne suffit pas si les deux blocages améliorent la perte. Toute attribution à ces backups requiert aussi une perte augmentée par rapport à Y11. Le rôle éventuel des MLP, d'autres têtes et du lecteur reste ouvert ; aucune exclusivité du groupe n'est démontrée. Les [résultats par template](results/test_summary.json) restent visibles et ne servent pas à choisir après coup une formulation favorable.

Les intervalles rééchantillonnent les familles en conservant leurs deux donneurs. Les sources sont aussi réemployées comme donneurs à d'autres familles : cette dépendance n'est pas entièrement traitée par ce bootstrap. Ces intervalles sont des approximations conditionnelles aux poids, aux sources et au plan de donneurs ; ils ne certifient ni une population de toutes les formulations IOI ni la variation entre réentraînements. Les deux réplications secondaires ne participent pas à la famille confirmatoire primaire. Les quatre intervalles principaux D,R,S,B utilisent le niveau98,75% annoncé ; ceux des autres quantités sont descriptifs, sans nouvelle revendication de contrôle de multiplicité global.

Les seuils de couverture locale sont des seuils d'effet descriptifs appliqués aux observations, pas un taux d'attributions mécanistes correctes. Les mesures DLA, self-repair centré et NLL n'ont pas le même estimand. Un désaccord ne prouve pas qu'une méthode antérieure se trompe.

## Réplications secondaires

### L9H6 — secondaire descriptif

- `D` : 1.204908, IC descriptif 98,75 % [1.124831 ; 1.286887].
- `R` : 1.04936, IC descriptif 98,75 % [0.9747103 ; 1.125231].
- `I` : 0.580505, IC descriptif 98,75 % [0.5258227 ; 0.6368122].
- `S` : 0.02458429, IC descriptif 98,75 % [0.01704573 ; 0.03485458].
- `B` : 0.05446117, IC descriptif 98,75 % [0.04154229 ; 0.06731509].
- `block_backups_minus_adapted` : 0.1194024, IC descriptif 98,75 % [0.1056554 ; 0.1327122].
- `block_controls_minus_adapted` : 0.06494124, IC descriptif 98,75 % [0.0556613 ; 0.07433372].
- `self_repair` : 1.538112, IC descriptif 98,75 % [1.445652 ; 1.631764].
- `delta_DE` : -1.714347, IC descriptif 98,75 % [-1.811008 ; -1.619292].
- `ordinary_ablation_damage` : 0.1555477, IC descriptif 98,75 % [0.1321792 ; 0.1799833].

- `Y00` : deux noms 1024/1024 ; vocabulaire entier 952/1024.
- `Y10` : deux noms 967/1024 ; vocabulaire entier 428/1024.
- `Y11` : deux noms 1024/1024 ; vocabulaire entier 904/1024.
- `Y01` : deux noms 1019/1024 ; vocabulaire entier 983/1024.
- `Y1wrong` : deux noms 1016/1024 ; vocabulaire entier 883/1024.
- `block_backups` : deux noms 1024/1024 ; vocabulaire entier 874/1024.
- `block_controls` : deux noms 1024/1024 ; vocabulaire entier 895/1024.

### L10H0 — secondaire descriptif

- `D` : 0.7058246, IC descriptif 98,75 % [0.6504103 ; 0.7619031].
- `R` : 0.4247691, IC descriptif 98,75 % [0.3885428 ; 0.4613057].
- `I` : 0.1781936, IC descriptif 98,75 % [0.1577644 ; 0.199532].
- `S` : 0.00817203, IC descriptif 98,75 % [0.00595302 ; 0.01090571].
- `B` : 0.1229637, IC descriptif 98,75 % [0.1077994 ; 0.1388149].
- `block_backups_minus_adapted` : 0.1332931, IC descriptif 98,75 % [0.1182461 ; 0.1490515].
- `block_controls_minus_adapted` : 0.01032937, IC descriptif 98,75 % [0.008419081 ; 0.01236387].
- `self_repair` : 0.677964, IC descriptif 98,75 % [0.6290941 ; 0.727176].
- `delta_DE` : -1.101848, IC descriptif 98,75 % [-1.175154 ; -1.02828].
- `ordinary_ablation_damage` : 0.2810556, IC descriptif 98,75 % [0.2555395 ; 0.307902].

- `Y00` : deux noms 1024/1024 ; vocabulaire entier 952/1024.
- `Y10` : deux noms 1015/1024 ; vocabulaire entier 654/1024.
- `Y11` : deux noms 1023/1024 ; vocabulaire entier 867/1024.
- `Y01` : deux noms 1022/1024 ; vocabulaire entier 982/1024.
- `Y1wrong` : deux noms 1022/1024 ; vocabulaire entier 851/1024.
- `block_backups` : deux noms 1020/1024 ; vocabulaire entier 800/1024.
- `block_controls` : deux noms 1023/1024 ; vocabulaire entier 864/1024.

## Vérification, incidents et coûts

Le premier passage s'arrête à l'export non fini, puis A1 localise un NaN dans la projection de mesure DE. Un produit vectoriel FP32 a produit181NaN avec entrées finies ; ce comportement n'est pas reproduit dans toutes les dispositions mémoire, notamment après copie des poids. La cause exacte n'est pas établie. A2 passe cette mesure seulement enFP64, vérifiée contre NumPy et sommation scalaire ; le réseau et toutes ses interventions restent FP32. Les archives initiale/A1 et les marges inchangées sont conservées. Le [diagnostic isolé](results/projection_incident.json) ne doit pas être présenté comme une reproduction universelle du NaN.

L'audit primaire est passé : les pertes sont recalculées par sommation scalaire et les sorties officielles sont rejouées sur quatre sources fixées, deux donneurs chacune. Les shams et recompositions sont contrôlés à chaque source par le runner. Le script initial ne couvre pas les secondaires ; un audit supplémentaire distinct peut les couvrir, ainsi que les blocages et E. Ces vérifications logicielles restent internes.

Temps du runnerdev valide :109.3376s. Temps du runnerréservée décrit ici :432.4207s, chargement et calcul jusqu'à la sauvegarde des états inclus ; les audits sont séparés. Les incidents et installations ont aussi un coût qui n'est pas contenu dans ce seul temps valide. La machine8Go a connu une pression mémoire et des travaux concurrents : ces durées ne sont pas un benchmark matériel ni une preuve d'efficacité relative d'une méthode.

## Portée et arrêt

Les contrôles de contributions et de normalisation figées sont des antécédents directs de [Rushing et Nanda](https://arxiv.org/html/2402.15390v2). Les backups ont été décrits par [Wang et al.](https://arxiv.org/html/2211.00593v1), notamment sous knockout conjoint des trois Name Movers, différent de notre intervention unitaire. Cette campagne apporte une épreuve tenue à l'écart du développement, avec un dommage utile et des contrôles de donneurs et de blocage. Elle ne découvre pas la compensation de GPT-2 et ne démontre pas une supériorité prédictive générale face aux mesures publiées.

L'utilité supplémentaire doit porter sur une conclusion précise corroborée, et non sur un simple désaccord d'unités ou un taux d'abstention plus élevé. Aucun score de vérité mécaniste générale ni statut de benchmark de référence n'est assigné. Une implémentation par une personne extérieure et une comparaison de méthodes indépendamment développées restent nécessaires pour ces revendications plus larges.

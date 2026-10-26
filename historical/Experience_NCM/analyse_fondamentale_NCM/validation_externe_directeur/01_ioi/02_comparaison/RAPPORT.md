# D02 — Rapport de comparaison des estimands

## Calcul effectivement réalisé

Évaluation réservée primaire :512familles,1024remplacements. Tous les comparateurs utilisent les mêmes sources, donneurs et poids. Les résultats complets et leurs intervalles sont dans [le rapport IOI](../RAPPORT.md).

- Effet total d'ablation régénérée, L11−L00 :0.8424164, IC descriptif 98,75 % [0.797105 ; 0.8907108]nat.
- Variation d'effet direct de la tête, facteur LN original :-3.385907, IC descriptif 98,75 % [-3.486256 ; -3.283417]logit centré.
- Self-repair publié, Δlogit centré−ΔDE :2.35486, IC descriptif 98,75 % [2.240201 ; 2.470372]logit centré.
- Récupération NLL sous adaptation, L10−L11 :1.909953, IC descriptif 98,75 % [1.81382 ; 2.006202]nat.
- Appariement spécifique de la suite, L1wrong−L11 :0.0379738, IC descriptif 98,75 % [0.02404077 ; 0.05402197]nat.

## Ce que la comparaison permet

L'effet total conserve toutes les réponses du système après perturbation ; DLA isole la projection directe selon un lecteur annoncé ; le self-repair soustrait cette variation directe au logit centré ; R mesure un changement de perte sous deux continuations. Ces objets ne sont pas des estimateurs concurrents d'une unique vérité. Les mesures restent dans leurs unités, sans classement de « classification correcte » inventé pour les auteurs antérieurs.

SansY10, la nécessité d'une continuation adaptée n'est pas testée. SansY1wrong, sa spécificité n'est pas testée. La règle complète conserve192familles à ses seuils locaux, contre507sans spécificité. Il ne s'agit pas d'une courbe de précision validée par un oracle interne de GPT-2. Le blocage annoncé fournit une corroboration fonctionnelle supplémentaire, pas des labels de mécanismes universels.

## Valeur ajoutée et travail restant

Le calcul joint rend les hypothèses de lecture plus explicites et permet de conserver les cas négatifs. Les versions amputées sont des ablations de notre protocole ; elles ne comptent pas comme autant de méthodes développées indépendamment. EAP-IG, méthode de localisation, n'est pas ajouté à un classement de compensation sans sortie commune. Cette évaluation ne démontre donc pas une supériorité générale sur plusieurs méthodes indépendantes. Le [rapport méthodologique](../../00_positionnement/02_methodes/RAPPORT.md) précise les estimands et critères.

Les intervalles rééchantillonnent les familles en conservant leurs deux donneurs. Les sources sont aussi réemployées comme donneurs à d'autres familles : cette dépendance n'est pas entièrement traitée par ce bootstrap. Ces intervalles sont des approximations conditionnelles aux poids, aux sources et au plan de donneurs ; ils ne certifient ni une population de toutes les formulations IOI ni la variation entre réentraînements. Les deux réplications secondaires ne participent pas à la famille confirmatoire primaire. Les quatre intervalles principaux D,R,S,B utilisent le niveau98,75% annoncé ; ceux des autres quantités sont descriptifs, sans nouvelle revendication de contrôle de multiplicité global.

Les seuils de couverture locale sont des seuils d'effet descriptifs appliqués aux observations, pas un taux d'attributions mécanistes correctes. Les mesures DLA, self-repair centré et NLL n'ont pas le même estimand. Un désaccord ne prouve pas qu'une méthode antérieure se trompe.

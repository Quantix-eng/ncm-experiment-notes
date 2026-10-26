# Consolidation du benchmark et rédaction du papier

25 octobre 2026. Rapports exclusivement Markdown. Le [manuscrit en anglais](PAPER.md) est rédigé, avec méthodes, résultats, contrôles, limites, positionnement bibliographique et références. La [note théorique](THEORIE_WYNER.md) développe le lien avec Wyner. Les expériences historiques et l'ancien manuscrit restent inchangés ; aucune nouvelle inférence NCM ni aucun apprentissage n'ont été exécutés.

## Contribution soutenue par les données

Le papier porte sur **l'attribution d'une révision apparente**, avec des contrôles calibrés et une application réelle. La calibration analytique vérifie les distinctions attendues dans une bibliothèque construite. L'étape 3a établit ensuite que les interventions sont exécutables sur les tenseurs FP32 du NCM, avec parité native exacte et des conclusions fonctionnelles bornées.

Le diagnostic fixé classe 15/16 familles comme ayant une préférence de réponse dominée par l'entrée, dans le régime testé. La dernière reste non identifiable, sans réajustement. C'est une observation empirique exploitable dans le papier ; elle ne permet pas de mesurer une précision de classification mécaniste sur NCM, faute de vérité interne.

Le résultat le plus discriminant pour la méthode est le contrôle des suites : les suites recalculées donnent 32/32 solutions du receveur après échange d'état, mais les suites natives figées du receveur aussi. Sans ce contrôle, on pourrait attribuer à une adaptation nécessaire une réussite déjà obtenue par une continuation antérieure. À la dose locale prévue, toutes les conditions C restent exactes : le protocole conserve donc aussi un test peu informatif sur la réparation, au lieu de sélectionner une corruption favorable après coup.

La formulation retenue est : **protocole calibré avec étude de cas sur réseau appris**, et non benchmark de référence validé extérieurement. La pertinence pour une publication repose sur ces distinctions expérimentales et leur auditabilité. L'acceptation, la nouveauté et la généralisation ne sont pas établies.

## Analyses auxiliaires effectuées

Le [plan auxiliaire](PROTOCOLE_AUXILIAIRE.md) a été scellé après connaissance des résultats historiques, avant ses calculs. Cette procédure est explicitement post-hoc. Elle n'a changé aucun seuil ni résultat de l'étape 3a.

1. **Distinction des entrées dans les codes conservés.** Le niveau 5 est constant, mais les niveaux 8 et 9 distinguent A et B dans 16/16 familles. Sur la loi finie équilibrée à deux entrées par famille, les codes grossiers 1..9 et fins 10..12 partagent exactement un bit. Ce constat ne mesure pas l'information mutuelle d'une population de puzzles, ni celle du tokenizer encodant des solutions, ni le coût de coordination du réseau. Il n'implique pas une reconstruction injective des codes.
2. **Dépendance au seuil.** Avec les mêmes critères d'éligibilité, 15/16 familles dépassent 0,70 ou 0,80, huit dépassent 0,90 et quatre dépassent 0,95. Le seuil principal reste 0,80. La majorité est descriptive et sa force dépend du seuil retenu.
3. **Exemples théoriques vérifiables.** Pour la parité uniforme, la production finale locale mutuellement indépendante exige C_W=n−1 bits, malgré une corrélation totale d'un bit et des paires indépendantes pour n≥3. Pour un mélange uniforme des deux solutions d'une base ambiguë, C_W=1 bit ; un indice commun qui révèle la branche donne une borne conditionnelle nulle. Les preuves générales sont écrites et les cas finis contrôlés par énumération.

Les [résultats détaillés](results/auxiliary.json) et le [registre de preuves](PREUVES.md) conservent tous les niveaux, familles et seuils annoncés.

## Ce que l'on peut relier à Wyner

L'entrée est une information latérale partagée à laquelle les prédicteurs parallèles ont accès. Les résultats sont compatibles avec son rôle de signal de conditionnement commun. Ils ne prouvent pas qu'elle constitue l'information commune minimale de Wyner, ni qu'elle compense une perte due au pooling.

Trois obstacles précis empêchent cette conclusion : la génération actuelle est déterministe à entrée fixée, le décodeur final peut recoupler les positions, et le pooling n'a pas été manipulé dans l'expérience. Pour les puzzles uniques, C_W(sortie | entrée complète)=0 ; ce résultat informationnel ne rend pas leur résolution calculatoirement gratuite. Le niveau constant de longueur 9 est le niveau 5, alors que le point de coupure est le niveau 9 de longueur 37. Les deux ne doivent pas être confondus.

Une relecture théorique par un second agent a vérifié les preuves et fait préciser le conditionnement et le statut hypothétique d'un futur sampler. Il s'agit d'une relecture interne, pas d'une réplication indépendante ou d'une validation par les pairs.

## Ce qui manque pour une revendication plus large

Le papier explicite les besoins d'une phase ultérieure : méthodes d'attribution indépendantes, au moins une autre famille de systèmes appris, régimes d'accès interne distincts, calibration des corruptions sur développement séparé, et évaluation externe. Aucun résultat favorable de ces campagnes n'est implicite dans le texte actuel.

Pour Wyner, une nouvelle expérience devrait fixer une véritable loi cible sur les solutions d'un même puzzle ambigu, contrôler la diversité et le sampler, puis séparer lecteurs locaux et globaux. Une causalité attribuée au pooling exigerait une intervention sur l'opérateur réellement utilisé en génération, avec accès à l'entrée et budgets contrôlés. Ces expériences ne sont pas lancées dans cette consolidation.

Les contrôles d'intégrité et les recomptages indépendants de cette livraison sont consignés dans [verification.json](results/verification.json). Ce sont des vérifications logicielles ; ils ne remplacent pas les validations scientifiques manquantes.

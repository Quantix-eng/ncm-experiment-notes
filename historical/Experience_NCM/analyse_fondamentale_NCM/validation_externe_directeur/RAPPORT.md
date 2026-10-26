# Bilan — InterpBench et réplications IOI

25 octobre 2026. Les deux évaluations demandées sont exécutées, sans nouveau sous-agent. La [feuille de route](FEUILLE_DE_ROUTE.md) conserve la correspondance des recommandations et livrables.

## Réplications IOI achevées

Même checkpoint et mêmes 512 familles, deux donneurs chacune, sans changement des règles gelées :

- **L9H6** : dommage 1,20491 nat ; récupération **1,04936 nat** ; spécificité **0,02458 nat**, IC descriptif 98,75 % [0,01705 ; 0,03485], au-delà de la marge 0,01. Justesse vocabulaire : 428/1024 figée, 904/1024 adaptée.
- **L10H0** : dommage 0,70582 nat ; récupération **0,42477 nat** ; spécificité **0,00817 nat**, IC [0,00595 ; 0,01091]. **Le critère de spécificité échoue**, sans retouche. Justesse : 654/1024 figée, 867/1024 adaptée.

Le blocage des backups corrobore leur contribution pour les deux cibles, dans leurs frontières respectives. Chaque tête, primaire comprise, passe 40 rejeux officiels complets bit à bit et 24 contrôles algébriques supplémentaires sur quatre sources fixées. Ce sont des audits internes, pas des réplications scientifiques indépendantes. [Rapport des réplications](01_ioi/REPLICATIONS_SECONDAIRES.md), [rapport IOI complet](01_ioi/RAPPORT.md).

## InterpBench : réserve exécutée, validation nuancée

Le programme historique du cas 13 a été retrouvé. Les auxiliaires sans masque vérifiable restent hors domaine. Le [protocole C2](02_interpbench/PROTOCOLE_C2.md) précède les sorties ; les prédictions ont été scellées avant leur falsification.

Sur **1024 sources, quatre tâches et 5632 requêtes**, RASP et Craft concordent sur les 5618 requêtes à cible définie ; 14 masques vides restent non évaluables. Les cas 13, 19 et 21 présentent une conformité élevée du réseau au programme. Le **cas 18 échoue à la fidélité native** : 119/256 sources exactes. Avec le padding officiel, le contrôle post-hoc atteint 122/256 seulement. Ce cas est conservé, sans attribuer toutes ses erreurs à notre diagnostic.

La règle C cinq conditions couvre en moyenne **50,52 %** des requêtes par tâche, avec **2,35 %** de contradictions neuronales, contre 50,90 % et 2,34 % sans contrôle mal apparié. Les couvertures diffèrent : **aucun avantage prédictif supplémentaire du protocole complet n'est établi ici**. Risques contre réseau et programme sont séparés.

L'audit vérifie 45056 propriétés scalaires ; 18432 régénérations E reproduisent exactement le donneur, comme prévu par le canal receveur bloqué. [Rapport InterpBench C2/C3](02_interpbench/RAPPORT_C2.md).

## Portée et suite

Les calculs demandés sont terminés et les résultats négatifs conservés. La fidélité du cas 18 reste un échec de validation de ce checkpoint dans le régime testé, pas une campagne remplacée. Les observations historiques et le primaire IOI sont préservés.

Les résultats IOI, Pythia et InterpBench sont désormais intégrés au [manuscrit révisé](00_positionnement/PAPER_REVISE.md), avec leurs méthodes, résultats négatifs, discussion et références. Cette intégration est éditoriale ; les artefacts scientifiques et les versions antérieures sont préservés.

Restent pour des revendications plus larges : expliquer la divergence du cas 18 dans une nouvelle campagne, couvrir les auxiliaires exclus, comparer des prédicteurs indépendants à couverture égale et réaliser une réplication humaine. Aucune de ces étapes n'est présentée comme exécutée.

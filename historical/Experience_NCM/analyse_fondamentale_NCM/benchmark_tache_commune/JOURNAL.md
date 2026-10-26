# Journal : tâche commune et accès au contexte

25 octobre 2026. Objectif : consolider le benchmark en réduisant les différences de tâche et en testant explicitement les contrats d'accès. Aucun résultat favorable n'est présupposé. Aucun fichier des campagnes antérieures n'est modifié.

## Préparation et gel

Lecture des limites de la comparaison NCM/Sudoku–GPT-2/texte. Choix de deux modèles appris de même taille sur une tâche commune, trois graines appariées et partitions par orbites. Le modèle parallèle est un Transformer conditionnel simple ; aucun nom NCM ne lui est attribué. L'entraînement AR utilise le teacher forcing, contrairement aux requêtes fixes parallèles : cette différence reste une limite à toute causalité attribuée à l'architecture seule.

Consultation ciblée de sources primaires sur les abstractions causales et les choix de corruptions. Les antécédents sont cités dans le protocole ; aucune priorité ni nouveauté générale de l'intervention d'échange n'est revendiquée.

`prepare.py` génère les 210 orbites et leurs partitions avant apprentissage. Tous les critères, scripts d'entraînement, calibration, évaluation, analyse et audit sont écrits avant gel. Un contrôle syntaxique est réalisé, sans chercher les performances d'un modèle pilote. Le gel local avant apprentissage inclut 785 fichiers et consigne 77 576 paramètres par modèle. Les versions de Python, NumPy et PyTorch et les empreintes de sources importées sont conservées dans `results/protocol_frozen.json`.

## Phases et conditions de conservation

Six entraînements sont prévus : deux architectures × trois graines 17,29,43 ; 1 200 pas par entraînement. Mêmes paramètres initiaux et tirages d'exemples pour chaque paire. Le checkpoint final est conservé sans sélection du meilleur sur validation. Toutes les pertes sont sauvegardées.

La calibration teste les quatre doses sur développement seulement et choisit une dose commune selon une règle avant résultats. Ses sorties et les poids sont scellés dans `calibration_frozen.json` avant évaluation. L'évaluation applique cette décision à toutes les familles et aux deux coupures fixées. Un échec de calibration reste un résultat et n'autorise pas une recherche supplémentaire.

Le contrat mémoire entière échange tout le tenseur disponible au point de coupure ; sous régénération, le receveur n'apporte aucun autre canal. Son invariance est attendue par construction. Une suite native figée du receveur apporte une information extérieure différente et n'est pas incluse dans cette assertion d'absence. L'éventuel changement de diagnostic entre contrats ne signifie pas un changement de l'algorithme appris. Les noms désignent l'objet fonctionnel testé et son accès.

Les comptes effectifs et l'état final de chaque phase sont consignés dans les JSON d'exécution et le rapport. Aucun ajustement scientifique après sortie n'est prévu. Un incident logiciel doit être conservé avec amendement horodaté avant reprise. Les rapports et le complément de papier sont rédigés après résultats.

L'audit séparé utilise des calculs scalaires de pertes et des équations explicites de sous-couches, puis vérifie les anciens manifestes en lecture seule. Le même agent programme et audite ; aucune validation externe n'est revendiquée.

## Résultats conservés

Les six entraînements terminent leur budget et leurs 120 variantes réservées sont toutes exactes, également en génération libre. La calibration sur développement ne trouve aucune dose satisfaisant le critère commun de dommage : repli préfixé rho=3, statut d'échec explicite, sans agrandir la grille. La frontière de calibration du 2026-10-25T15:40:53Z est conservée avant évaluation.

Après deux blocs, la régénération sur les états échangés suit toujours le donneur. Après un bloc, le résultat dépend de l'entraînement et du contrat. Les abstentions pour ratios négatifs ou interactions restent publiées sans troncature ni changement de règle. Tous les signes moyens R primaires sont négatifs ; aucune réparation utile n'est revendiquée.

L'audit gelé passe sans correctif : 42 240 recompositions, 576 reprises explicites, 19 200 pertes scalaires, 72 contrôles natifs et 3 840 conditions de chemin bloqué. Les anciens fichiers et critères restent inchangés. La rédaction des rapports, de la note sur les estimands et du complément de papier est postérieure aux résultats ; elle n'introduit pas de nouveau calcul expérimental.

Un script éditorial `build_paper.py`, **ajouté après le gel expérimental**, assemble `PAPER_MATCHED_TASK.md`, version 3, depuis la version 2 et les sections nouvelles. Il actualise résumé, méthodes, résultats et limites, conserve les parties théoriques et rétablit les liens vers les artefacts d'origine. Il ne modifie aucun résultat ni manuscrit source. Aucune relecture externe de cette nouvelle version n'est revendiquée.

## Clôture de la livraison

Les rapports, le manuscrit version 3 et la documentation sont finalisés. Le contrôle éditorial vérifie 52 liens locaux, les comptes centraux et les empreintes gelées ; ses résultats figurent dans `results/publication_checks.json`. Le manifeste final scelle les scripts, rapports, données, checkpoints et observations de cette campagne. Aucune nouvelle expérience de traitement de la limite de calibration n'est lancée. Les pistes futures mentionnées dans les textes restent non exécutées.

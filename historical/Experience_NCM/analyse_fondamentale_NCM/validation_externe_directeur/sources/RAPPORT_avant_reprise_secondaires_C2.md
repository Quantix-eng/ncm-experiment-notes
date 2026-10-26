# Avancement et premiers résultats — clôture du 25 octobre 2026

La [feuille de route générale](FEUILLE_DE_ROUTE.md) relie les propositions du directeur à leurs feuilles de route et rapports. Les antécédents de self-repair et la portée des revendications ont été corrigés dans une [nouvelle copie du papier](00_positionnement/PAPER_REVISE.md). Les campagnes historiques sont préservées.

## Résultats obtenus

**GPT-2 small × MIB–IOI : évaluation primaire achevée.** Après 128 familles de développement, 512 familles réservées ont été évaluées, avec deux donneurs chacune, sans sélection sur dommage ou réussite. Le dommage moyen sous continuation figée atteint **2,7524 nats** ; la continuation adaptée récupère **1,9100 nat**. L'avantage d'une suite correctement appariée sur l'autre donneur est plus faible : **0,0380 nat**. Les IC descriptifs à 98,75 % sont respectivement [2,6572 ; 2,8461], [1,8138 ; 2,0062] et [0,0240 ; 0,0540].

Sur les 1 024 interventions corrélées, l'exactitude dans tout le vocabulaire passe de **59/1 024 avec suite native figée à 622/1 024 avec suite adaptée**, contre 952/1 024 nativement. La récupération est donc partielle. Empêcher l'adaptation des backups publiés augmente la NLL de **0,2511 nat**, contre **0,0401** pour le groupe témoin : cela corrobore leur contribution dans ce contrat, sans établir un mécanisme exclusif. L'audit primaire passe : huit rejeux officiels bit à bit et 112 vérifications de pertes/recomposition. [Rapport IOI](01_ioi/RAPPORT.md).

**Pythia-160M : test structurel achevé sur 512 séquences, 2 048 corruptions.** La récupération moyenne attribuable au lecteur vaut **−0,000601 nat**, avec un IC descriptif à 95 % [−0,001589 ; 0,000141] : aucune récupération moyenne positive n'est établie. Les contrôles distinguent ce lecteur de blocs aval absents à cette frontière. Les audits passent ; les corrections numériques et leurs versions initiales sont conservées. [Rapport Pythia](03_pythia/RAPPORT.md).

**InterpBench : développement et oracle préparatoire seulement.** L'audit officiel de précision et de replay passe. Le programme externe, son modèle Craft et le réseau concordent sur **1 536 échanges de développement**, dont 499 changent la réponse. Cependant, la compilation actuelle et les arêtes publiées ne s'alignent pas encore pour trois des quatre tâches réservées. Ces sorties réservées restent fermées ; aucun succès externe n'est annoncé. [Complément de clôture](02_interpbench/COMPLEMENT_CLOTURE.md).

## Ce qui reste à faire

- Exécuter les réplications IOI secondaires L9H6 et L10H0 ; elles ne sont pas réalisées ici. L'audit supplémentaire des blocages et du plan E est préparé, non exécuté.
- Résoudre les correspondances programme/checkpoints d'InterpBench, puis geler les prédictions et ouvrir son évaluation réservée.
- Établir la valeur prédictive supplémentaire face à des méthodes indépendantes ; la comparaison actuelle de métriques et d'ablations ne démontre pas une supériorité générale.
- Faire réaliser une implémentation et une évaluation par une personne extérieure, puis intégrer les nouveaux résultats au manuscrit.

Les intervalles IOI restent approximatifs, notamment parce que les sources servent aussi de donneurs. Les résultats ne sont pas des taux de classification mécaniste correcte. **Arrêt demandé : aucune nouvelle campagne ni réplication secondaire lancée ; sous-agents interrompus et calcul IOI courant terminé.**

# Rapport — corrections de positionnement et spécifications d'évaluation

25 octobre 2026. Réponse au rapport du directeur. Aucun calcul de réseau dans ce volet. Les documents historiques, leur gel et le PDF V3 n'ont pas été réécrits.

## Résultat de ce volet

L'omission de **Rushing et Nanda (ICML 2024)** est confirmée dans la bibliographie du manuscrit V3. Leur étude considère déjà les contributions aval figées et le facteur de normalisation finale. La continuation figée seule ne peut donc porter une revendication de nouveauté. La correction est appliquée dans les [paragraphes prêts pour le manuscrit](CORRECTIONS_MANUSCRIT.md) et dans la spécification des [comparaisons de méthodes](02_methodes/RAPPORT.md). Les références ont été vérifiées sur les [actes ICML](https://proceedings.mlr.press/v235/rushing24a.html) et le [texte primaire, §2–3](https://arxiv.org/html/2402.15390v2).

La question principale retenue est : **quelles conclusions sur une compensation sont corroborées par des interventions réservées, et que changent les contrôles de continuation figée, de spécificité et de lecture ?** Elle se teste sans étiqueter l'ensemble de GPT-2 ou du NCM par une vérité mécaniste inexistante.

## Corrections substantielles à la recommandation

Les têtes IOI recommandées sont documentées, mais le résultat historique de backup ne garantit pas une réparation pour l'intervention proposée. L'article IOI étudie notamment un knockout conjoint des trois Name Movers ; le remplacement d'une seule tête par une activation native d'un autre exemple est une expérience différente. Les têtes d'une même couche ne sont pas en aval causal de leurs sorties respectives. Ces distinctions sont détaillées dans le [rapport des antécédents](01_antecedents/RAPPORT.md).

La source Self-Repair contient également une incohérence locale : le texte de §3.2 cite L11H10 de Pythia-160M, tandis que la légende de la figure 5 indique L11H0. Cette incohérence est transmise au volet Pythia ; elle ne justifie pas une sélection de tête après examen des effets.

## Comparaisons rendues concrètes

Le [rapport de méthodes](02_methodes/RAPPORT.md) définit séparément attribution directe, effet d'ablation, self-repair publié, récupération de NLL, spécificité et blocage des composants. Il donne le calcul exact avec activation donneuse et normalisation native, ainsi que les conditions qui permettraient une conclusion supplémentaire. Aucun score de « classification correcte » ne sera fabriqué pour comparer ces objets différents.

Le [rapport de publication](03_portee_publication/RAPPORT.md) réduit les revendications aux observations effectivement disponibles. L'effet général AR/parallèle sort des objectifs immédiats. Le système parallèle conserve son statut de proxy de génération en un coup ; les différences d'objectif d'apprentissage empêchent une conclusion causale générale sur le paradigme. Le volet commun-information reste conditionnel et ne devient pas une mesure de Wyner.

## Externalité : ce qui est acquis et ce qui manque

Des sources, données et hypothèses publiées par d'autres équipes apportent une externalité des objets. Elles ne constituent pas une réplication de notre méthode par une personne indépendante. Le [dossier de réplication](04_replication_humaine/RAPPORT.md) précise le livrable à faire produire, les données nouvelles et la séparation entre calcul des résultats et décision d'attribution. Aucun contact, engagement de chercheur, revue humaine ou validation extérieure n'est annoncé comme effectué.

## Décision scientifique actuelle

Les corrections documentaires sont réalisées. La supériorité du protocole, sa valeur ajoutée sur IOI, sa couverture d'affirmations correctes sur InterpBench et sa réplication indépendante restent des questions empiriques. Leurs rapports doivent pouvoir conclure négativement. Une étude reproductible et utile peut être présentée à TMLR sans revendiquer le statut de benchmark de référence ; les critères officiels demandent des affirmations étayées et un intérêt clairement communiqué. Cette admissibilité de principe ne prédit aucune acceptation. [Critères TMLR](https://jmlr.org/tmlr/acceptance-criteria.html)

## Application au papier

Le [papier révisé](PAPER_REVISE.md) applique ces corrections à une copie complète du manuscrit V3 : titre et résumé recentrés, antécédents directs insérés, distinction des estimands, conclusion réduite et programme extérieur explicitement prospectif. Aucun résultat des campagnes actuellement préparées ou exécutées dans d'autres volets n'est intégré dans cette copie.

La vérification compare littéralement les méthodes et résultats historiques des sections 4–5, ainsi que la théorie de la section 7, après neutralisation des cibles de liens déplacées : ils sont inchangés. Tous les liens locaux de la copie ont été recalculés et vérifiés. Le SHA-256 de la source V3 reste `c5cd61ab808b96db50b4dce38625076a2a7220b6e2833f53ce586efe529e0120`. Il s'agit d'un contrôle éditorial et logiciel, pas d'une validation scientifique indépendante.

## Relecture supplémentaire du pilote IOI

Un [audit interne des scripts IOI](AUDIT_IOI_INTERNE.md) examine le site d'intervention, les suites figées, les backups, l'effet direct et les inférences statistiques. Il confirme la prédiction analytique de self-repair nul sous suite et facteur LN figés, à l'arrondi près, et propose un audit distinct du lecteur. Il précise aussi que le contraste différentiel entre blocages ne suffit pas seul à montrer une perte de récupération, et que des donneurs partagés créent une dépendance entre familles. L'échec d'export non fini communiqué par le responsable de campagne y est consigné, sans diagnostic causal anticipé.

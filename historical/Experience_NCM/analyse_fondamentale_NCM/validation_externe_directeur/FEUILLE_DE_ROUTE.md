# Feuille de route d'application de l'avis du directeur

25 octobre 2026. Source conservée : [avis intégral](sources/AVIS_DIRECTEUR.md). Les feuilles de route ci-dessous sont écrites pour cette nouvelle campagne ; elles ne constituent pas une prérégistration rétroactive des résultats historiques.

## Objectif et ordre des décisions

La revendication principale devient : déterminer quels contrôles sont nécessaires pour attribuer une compensation, et quand les évaluations existantes conduisent à des conclusions différentes qui peuvent être éprouvées. Le statut de benchmark de référence et un effet causal général AR/parallèle ne sont plus des objectifs immédiats. Une récupération absente ou une absence de valeur ajoutée sont des issues recevables.

1. Corriger dès maintenant les antécédents et la portée du papier, dans une nouvelle copie éditoriale ; conserver les versions et campagnes historiques.
2. Intégrer GPT-2–MIB–IOI sur développement extérieur ; passer la porte de dommage avant toute sortie réservée. Les donneurs au même site, les mesures publiées et le blocage des backups sont fixés avant le pilote.
3. Exécuter le test IOI seulement si cette porte et les audits logiciels passent. Garder les têtes secondaires distinctes, sans remplacer le résultat primaire.
4. Éprouver séparément l'oracle InterpBench et la frontière lecteur/continuation de Pythia. Chaque branche possède son propre gel et ses propres arrêts ; un succès ne compense pas l'échec d'une autre.
5. Préparer une spécification et un paquet de réplication par une personne extérieure. Une relecture ou implémentation par un sous-agent reste interne.
6. Rendre les sorties réutilisables, les échecs et coûts visibles, puis évaluer si une conclusion supplémentaire est réellement corroborée. Ne pas fabriquer un classement de méthodes dont les sorties ont des sens différents.

## Correspondance exhaustive des propositions avec les livrables

- **D01 — Antécédents directs, nouveauté et priorité.** [Feuille de route](00_positionnement/01_antecedents/FEUILLE_DE_ROUTE.md), [rapport](00_positionnement/01_antecedents/RAPPORT.md).
- **D02 — DLA, self-repair, NLL et comparateurs ablatés.** [Feuille de route méthodologique](00_positionnement/02_methodes/FEUILLE_DE_ROUTE.md), [rapport](00_positionnement/02_methodes/RAPPORT.md), plus [comparaison IOI](01_ioi/02_comparaison/FEUILLE_DE_ROUTE.md) et son [rapport](01_ioi/02_comparaison/RAPPORT.md).
- **D03 — Dommage utile comme porte, abstention, aucun choix sur test.** [Feuille de route IOI](01_ioi/FEUILLE_DE_ROUTE.md), [rapport IOI](01_ioi/RAPPORT.md), [protocole de développement](01_ioi/PROTOCOLE_DEV.md).
- **D04 — GPT-2 small, MIB–IOI, 128/512 familles, trois variantes et trois têtes.** Même [feuille de route](01_ioi/FEUILLE_DE_ROUTE.md) et [rapport](01_ioi/RAPPORT.md). Le point de coupure et les canaux retenus sont déclarés, sans assimiler une tête à toute l'architecture.
- **D05 — Spécificité avec plusieurs donneurs au même site.** [Feuille de route](01_ioi/03_specificite/FEUILLE_DE_ROUTE.md), [rapport](01_ioi/03_specificite/RAPPORT.md).
- **D06 — Blocage de l'adaptation des backups, témoin de même taille.** [Feuille de route](01_ioi/04_blocage/FEUILLE_DE_ROUTE.md), [rapport](01_ioi/04_blocage/RAPPORT.md).
- **D07 — Oracle externe InterpBench, granularité, prédictions réservées, IIA et couverture.** [Feuille de route](02_interpbench/FEUILLE_DE_ROUTE.md), [rapport](02_interpbench/RAPPORT.md). La version et les identifiants de tâches doivent être contrôlés avant emploi ; les six classes construites ne sont pas une vérité imposée.
- **D08 — Pythia-160M, facteur LN natif figé et falsification d'une fausse adaptation aval.** [Feuille de route](03_pythia/FEUILLE_DE_ROUTE.md), [rapport](03_pythia/RAPPORT.md).
- **D09 — Retrait de la revendication générale AR/parallèle ; portée, publication et échecs.** [Feuille de route](00_positionnement/03_portee_publication/FEUILLE_DE_ROUTE.md), [rapport](00_positionnement/03_portee_publication/RAPPORT.md), [copie révisée du papier](00_positionnement/PAPER_REVISE.md).
- **D10 — Implémentation et nouvelles familles par une personne indépendante.** [Feuille de route](00_positionnement/04_replication_humaine/FEUILLE_DE_ROUTE.md), [rapport](00_positionnement/04_replication_humaine/RAPPORT.md). L'externalité des ressources ne suffit pas à réaliser cette proposition humaine.
- **D11 — Extension compatible MIB, plusieurs méthodes, coûts et cas positifs/négatifs/ambigus.** [Feuille de route](04_reutilisation/FEUILLE_DE_ROUTE.md), [rapport](04_reutilisation/RAPPORT.md).
- **D12 — Traçabilité, coûts réels, intégrité et absence de changement silencieux.** [Journal](JOURNAL.md), [rapport global](RAPPORT.md) et manifestes propres à chaque branche. Tous les rapports distinguent préparation, résultat acquis et travail restant.

## Critères de livraison

Pour chaque proposition : une feuille de route, un rapport complet de l'état effectivement atteint, sources vérifiées et liens vers les preuves. Un rapport de blocage ou un résultat négatif est complet s'il explique la condition non satisfaite et ce qu'il faudrait pour reprendre ; il ne transforme pas le travail restant en travail exécuté. Aucun entraînement nouveau, aucune retouche des campagnes gelées. Les calculs CPU sont chronométrés ; les estimations du directeur ne sont pas rapportées comme mesures locales.

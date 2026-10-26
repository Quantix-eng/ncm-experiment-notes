# Registre des résultats intégrés au manuscrit

25 octobre 2026. Ce registre relie les nouvelles sections de [PAPER_REVISE.md](PAPER_REVISE.md) aux expériences conservées. Il ne constitue pas une nouvelle expérience ni une validation extérieure.

## IOI — sections4.6,5.8et annexeA.1

- [Protocole réservé](../01_ioi/PROTOCOLE_TEST.md), [gel antérieur aux sorties](../01_ioi/results/freeze_test.json).
- Résultats [L9H9](../01_ioi/results/test_summary.json), [L9H6](../01_ioi/results/test_L9H6_summary.json), [L10H0](../01_ioi/results/test_L10H0_summary.json) : D/R/I/S/B, effets absolus de blocage, mesures de logits, justesse et E.
- [Observations primaires](../01_ioi/results/test_observations.json) : justesse native A/B/Aplus et contrôles individuels. Les fichiers secondaires de même nom suffixé conservent leurs observations.
- Audits officiels [primaire](../01_ioi/results/test_supplementary_audit.json), [L9H6](../01_ioi/results/test_L9H6_supplementary_audit.json), [L10H0](../01_ioi/results/test_L10H0_supplementary_audit.json). Les E de couche9sont identiques par construction de la frontière, pas trois nouveaux corpus.
- [Amendements numériques](../01_ioi/AMENDEMENTS.md) : échecs de développement conservés ; réseau et interventions FP32, mesure DE passée enFP64.

## Pythia — sections4.7,5.9et annexeA.2

- [Protocole](../03_pythia/protocol.json), [gel version1.2](../03_pythia/results/protocol_frozen.json).
- [Résumé test](../03_pythia/results/test_summary.json), [observations](../03_pythia/results/test_observations.json) : effets de lecteur, self-repair, intervalles et justesse. La comparaison appariée Y10/Y11 est calculable directement depuis les booléens conservés ; aucun gain ni perte d'exactitude n'y apparaît.
- [Audit FP64 supplémentaire](../03_pythia/results/test_memmap_verification.json), [statut de clôture](../03_pythia/results/execution_status.json). Le nouvel audit de test est post-hoc, distinct du script initial gelé.

## InterpBench — sections4.8,5.10et annexeA.3

- [Protocole C2](../02_interpbench/PROTOCOLE_C2.md), [gel de l'expérience](../02_interpbench/results/c2_frozen.json), [gel des prédictions avant falsification](../02_interpbench/results/c2_predictions_frozen.json).
- [Résultats par tâche et seuil](../02_interpbench/results/c2_summary.json), [requêtes de falsification](../02_interpbench/results/c2_falsification.json), [audit](../02_interpbench/results/c2_verification.json). Le risque contre réseau utilise tous les tokens utiles ; le risque contre programme utilise son masque défini et peut inclure un défaut natif de conformité.
- [Audit de développement B1](../02_interpbench/results/craft_development_b1.json) : les1 536accords incluent499changements de réponse sur les deux tâches de développement simples.
- [Provenance historique du cas13](../02_interpbench/sources/history_audit/history_comparison.json), [plan et auxiliaires exclus](../02_interpbench/data/c2_plan.json).
- [Protocole post-hoc C3](../02_interpbench/PROTOCOLE_C3_PADDING.md), [résultats padding](../02_interpbench/results/c3_padding.json), [audit des masques et de C3](../02_interpbench/results/c2_mask_and_c3_audit.json). Le cas18n'est pas remplacé ou retiré.

## Statut éditorial

L'[ancienne copie](archives/avant_integration_resultats_20261025/PAPER_REVISE.md) est conservée intégralement. `integrate_results.py` assemble la nouvelle rédaction depuis cette copie et les résumés conservés. [CLAIMS_INTEGRATION.json](CLAIMS_INTEGRATION.json) épingle les insertions numériques IOI générées ; `verify_integration.py` vérifie ces insertions, les principaux comptes statiques, les liens et l'invariance des artefacts scientifiques.

Les versions historiques, protocoles, données et seuils ne sont pas réécrits. Le manuscrit n'annonce aucune réplication humaine indépendante, aucun résultat de diffusion, aucune supériorité prédictive démontrée, aucune soumission ou intégration officielle MIB. Les références Pythia, WikiText et InterpBench ont été vérifiées sur leurs sources primaires pour cette intégration.

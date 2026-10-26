# Réplications IOI L9H6 et L10H0 — résultats réservés

25 octobre 2026. Les deux cibles figuraient au [protocole test](PROTOCOLE_TEST.md), gelé avant le résultat primaire. Aucun seuil, donneur ou exemple n'a été changé. Chaque tête est évaluée sur les mêmes 512 familles et deux donneurs par famille, soit 1024 interventions. Ces résultats sont secondaires et descriptifs ; mêmes poids et sources ne constituent pas des réplications scientifiques indépendantes.

## L9H6 : récupération et appariement retrouvés

- Dommage sous suite native figée D : **1,20491 nat**, IC descriptif 98,75 %[1,12483 ;1,28689].
- Récupération R : **1,04936 nat**, IC[0,97471 ;1,12523].
- Spécificité S : **0,02458 nat**, IC[0,01705 ;0,03485], au-delà de la marge 0,01 annoncée.
- Blocage des backups moins reprise adaptée : **0,11940 nat** ; effet différentiel contre groupe témoin B : **0,05446 nat**, IC[0,04154 ;0,06732].
- Justesse vocabulaire : **428/1024 figée →904/1024 adaptée**, contre 952/1024 nativement. La suite mal appariée donne 883/1024.

La récupération est partielle et l'appariement apporte un effet moyen bien plus faible que R. Le groupe de backups bloqué est le même ensemble strictement aval 4+2 que pourL9H9. [Résumé complet](results/test_L9H6_summary.json).

## L10H0 : récupération, mais critère de spécificité non satisfait

- D : **0,70582 nat**, IC descriptif 98,75 %[0,65041 ;0,76190].
- R : **0,42477 nat**, IC[0,38854 ;0,46131].
- S : **0,00817 nat**, IC[0,00595 ;0,01091]. Sa borne inférieure ne dépasse pas 0,01 : **le critère annoncé échoue**. Il ne faut pas déplacer la marge ou se contenter du signe positif.
- Blocage backups moins reprise adaptée : **0,13329 nat** ; B : **0,12296 nat**, IC[0,10780 ;0,13881].
- Justesse vocabulaire : **654/1024 figée →867/1024 adaptée**, contre 952/1024 nativement. Suite mal appariée :851/1024.

La frontière se situe plus tard, après attention 10. Seuls les deux backups de couche 11 et deux témoins restent strictement aval. Les différences de R ou de B entre têtes ne sont donc pas des effets architecturaux à budget de continuation égal. [Résumé complet](results/test_L10H0_summary.json).

## Ce que ces deux essais ajoutent

Les trois cibles L9H9, L9H6, L10H0 présentent une récupération NLL et un effet du blocage annoncé, mais la spécificité au-delà de la marge n'est pas retrouvée pour la dernière. Une récupération importante n'implique donc pas automatiquement le succès de tous les critères du benchmark. Les comptes locaux du diagnostic complet restent 192/512 pourL9H9,200/512 pourL9H6 et 100/512 pourL10H0 ; ce sont des comptes de seuils d'effet, pas des précisions mécanistes.

Les intervalles regroupent les deux donneurs par source mais ne couvrent pas entièrement la dépendance créée par leur réemploi. Les intervalles secondaires ne forment pas une nouvelle famille confirmatoire simultanée avec le primaire.

## Vérifications et coût

Pour chacune des trois têtes, l'[audit supplémentaire](audit_supplementaire.py) vérifie 40 forwards officiels complets sur quatre sources fixées : reprise adaptée, deux blocages et échanges E. Tous retrouvent les logits stockés **bit à bit**. Il vérifie aussi 24 identités scalaires : projection DE indépendante, zéro-self-repair sous lecteur et suite figés à l'arrondi près, décomposition des effets du lecteur. [AuditL9H9](results/test_supplementary_audit.json), [auditL9H6](results/test_L9H6_supplementary_audit.json), [auditL10H0](results/test_L10H0_supplementary_audit.json).

Ces audits ont été ajoutés après le gel expérimental, comme contrôles logiciels distincts, sans changement des décisions. Ils restent internes. Les runners secondaires ont pris **422,27 s** et **436,06 s**, chargement et sauvegarde compris ; les audits sont séparés et les durées ne sont pas des benchmarks matériels isolés.

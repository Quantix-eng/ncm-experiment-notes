# Journal de l'étape 3a

## Avant les nouvelles inférences

- Instruction : tester O, C et E sur les 16 familles contre-factuelles déjà construites, sans forcer de classe ni déplacer les seuils. Rapports Markdown uniquement.
- Lecture du protocole analytique, de sa note d'identifiabilité, des sources NCM et du runner croisé historique. Aucune source apprise n'est étiquetée à partir de la bibliothèque jouet.
- Point de coupure 9 fixé avant sorties ; quatre perturbations locales signées par ancrage A/B, norme locale 0,5 fois le RMS de position ; suites mal appariées sur l'autre support de même norme.
- Plans E fixés : A/B et A/Aplus, chacun à 12 conditions, avec cache exclusivement lié au préfixe récepteur. Marge C 0,01 nat, IC ajustés sur D/R/I/S ; conditions d'éligibilité et seuils E écrits avant exécution.
- Compilation et contrôles de scores sur logits synthétiques uniformes. Vérification des 16 sources et masques sans appel au modèle.
- Gel local à **13:57:39 UTC le 25 octobre 2026** : protocole, paramètres, scripts, poids/configuration, données et imports. Aucun réglage sur les nouvelles sorties.

## Exécution et audit principaux

- Runner terminé à 13:58:57 UTC, 48,3 secondes de calcul et sauvegardes. 48 entrées natives, 128 essais C, 384 sorties E. Aucun incident ni modification des scripts gelés.
- Les 48 chemins natifs et reprises sans perturbation passent la parité bit à bit des codes, états et logits. Les trois variantes de chacune des 16 familles sont résolues exactement au niveau final.
- C : dommage moyen de NLL équivalent dans la marge préfixée ; récupération utile et spécifique non établie. Aucune escalade de dose ni changement de point de coupure après ce constat.
- E : 15 familles de préférence A/B dominée par l'entrée selon les critères fixés ; une famille (`sudoku-012`) conserve le libellé non monotone/non identifiable car u/g dépasse 1. Aucun arrondi pour la faire entrer dans la classe majoritaire.
- Audit séparé passé : deux solveurs avec cap 3 ; 1 024 reconstructions de décodeur ; 32 suites rejouées par une boucle explicite distincte ; 6 nouveaux appels natifs de vérification ; scores, intervalles et décisions recalculés. Même agent, pas réplication scientifique indépendante.

## Export descriptif ajouté après lecture des résultats principaux

- `audit_sorties.py` exporte les critères de justesse/indices/format déjà annoncés, à partir des logits sauvegardés, sans appel neural. Ce script n'appartenait pas au gel initial ; son ajout n'affecte ni les contrastes, ni les seuils, ni les décisions. Son code et ses sorties sont inclus dans le manifeste final.
- Le plan E conserve les deux suites figées. Leur réalignement selon receveur/donneur utilise les conditions déjà exécutées : il ne sélectionne pas de nouvelles interventions.
- Sur les 32 échanges A↔B hors diagonale, suite recalculée et suite native du receveur figée donnent chacune 32 solutions du receveur. La suite native du donneur figée donne 32 solutions du donneur et aucune du receveur. Le succès après reprise ne suffit donc pas à établir la nécessité d'une adaptation à l'état échangé.
- Les cinq conditions C donnent chacune 128/128 solutions exactes. Les différences de logits et de codes ne sont pas des réparations de changements d'argmax.
- `COMPLEMENT_SORTIES.md` fournit cette interprétation et des nombres en notation scientifique pour éviter les arrondis à zéro. `RAPPORT.md` reçoit une note de lecture générée par l'exporteur ; le protocole, les scripts principaux gelés, les résultats bruts et décisions restent inchangés.
- Les résultats n'identifient pas un recalcul intégral, un mélange causal ni une complexité intrinsèque supérieure aux témoins. Arrêt après 3a ; pas d'étape Wyner, de modification du manuscrit ni des campagnes précédentes.

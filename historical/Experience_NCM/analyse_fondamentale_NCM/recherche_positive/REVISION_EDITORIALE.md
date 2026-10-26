# Révision critique du 24 octobre 2026

Cette révision répond à la demande d'évaluer honnêtement la portée des résultats et le risque de p-hacking. Elle corrige la présentation ; elle ne modifie ni les observations, ni la sélection des puzzles, ni les graines, ni les métriques, ni les critères annoncés. Aucun nouveau jeu ou nouvelle condition expérimentale n'est introduit. Les vérifications et rejeux utilisent les entrées déjà conservées.

## Corrections de portée

- L'ouverture ne présente plus « deux critères satisfaits » comme un verdict de découverte scientifique forte. L'effet d'édition et la couverture sont rapportés, avec une nouveauté et une importance scientifique non établies.
- La sélection d'une piste favorable est explicite. Le minimum de 30 % a été choisi après une couverture validation de 201/400 = 50,25 %, sans justification indépendante d'utilité minimale. Le protocole et ses décisions historiques restent conservés, sans les durcir ou les redéfinir rétrospectivement.
- Le gel initial concerne protocole, tables et pilote. Les scripts n'étaient pas épinglés à ce moment. Un gel local et des logs réinscriptibles ne remplacent ni une prérégistration indépendante ni un registre externe exhaustif. L'audit ne prétend pas démontrer l'absence absolue de p-hacking.
- Le contrôle gaussien apparie les normes et le support effectif, pas la géométrie du codebook. Il ne suffit pas à établir une spécificité conceptuelle, un circuit de raisonnement ou un mécanisme propre à NCM.
- L'accord prédictif et le succès d'intervention utilisent des populations et opérations différentes. Leur différence n'est pas une estimation contrôlée de l'effet du contexte. Les échecs d'écriture sont décomposés descriptivement, après coup : 237 demandes inchangées, trois donnant un autre chiffre. Aucune nouvelle inférence confirmatoire n'est attachée à cette décomposition.
- La garantie de non-dégradation relève de l'unicité, du vérificateur externe et du repli exact. Elle ne démontre pas un mécanisme interne du NCM. Les sorties en repli peuvent rester incorrectes.
- La figure temporelle montre désormais les moyennes arithmétiques en millisecondes. Le rapport géométrique de lecture de 9,2117 reste dans le texte comme statistique complémentaire, sans être présenté comme une division par neuf du coût moyen ou du pipeline complet.
- Les compteurs de vérification sont décrits comme des assertions logicielles, dont 4 096 portent sur quatre propriétés de 1 024 puzzles. Ils ne constituent pas des milliers de réplications scientifiques ou une revue indépendante.
- La distinction du pilote entre 0/32 sorties exactes sur 89 tokens et 10/32 grilles exactes sur 81 cases reste explicite.

## Historique et contrôle de l'invariance

`sources/avant_revision_editoriale_20261024.zip` préserve les rapports, le PDF, les scripts, la vérification et l'archive de sources avant cette révision. `results/editorial_revision_baseline.json` enregistre l'ancien manifeste et 31 fichiers scientifiques à garder identiques, dont les données, résultats numériques, règles gelées et scripts d'expérience.

`scripts/editorial_audit.py` vérifie ces empreintes et la cohérence des corrections. `results/editorial_revision_audit.json` annonce les contrôles effectivement passés. Les documents générés, la figure de chronométrage, les sources de publication et le manifeste final sont actualisés. Les sept campagnes antérieures ne sont pas modifiées.

`run.sh` reconstruit désormais la publication et vérifie les observations conservées, sans les remplacer. `scripts/reproduce.py` permet un recalcul complet dans un nouveau dossier frère à partir des sources historiques et des mêmes entrées, avec une notice distinguant l'ancienne rédaction de l'interprétation révisée. Cela évite de réécrire les mesures initiales, notamment les chronométrages. Ce recalcul n'est pas une confirmation indépendante.

Ce contrôle apporte de la traçabilité à la révision éditoriale. Il n'est pas une réplication indépendante ni une validation de l'originalité scientifique. Les observations restent des résultats circonscrits au protocole et aux checkpoints étudiés.

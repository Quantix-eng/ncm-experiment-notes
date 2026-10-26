# Généralisation architecturale NCM : résultats et portée

La campagne est terminée. L'[article complet en anglais](PAPER.md), sa [version PDF](PAPER.pdf) et sa source LaTeX `PAPER.tex` présentent méthode, preuve conditionnelle, résultats et limites. Le [protocole](OBJECTIFS_ET_PROTOCOLE.md), l'[amendement mémoire](AMENDEMENT_MEMOIRE.md) et le [journal](JOURNAL.md) rendent les décisions inspectables.

## Réponse à la question principale

**Les résultats ne démontrent pas un défaut général du NCM.** La propriété mathématique concerne l'agrégation de représentations additives par moyennes. Sudoku impose les histogrammes fixes qui la rendent particulièrement saillante à neuf positions. Le tokenizer complet utilise aussi normalisation, sous-échantillonnage appris et résidus : l'invariance d'une moyenne ne force donc pas celle des codes natifs.

**Les deux cas texte ne reproduisent pas le niveau constant.** Chaque tokenizer est testé sur 256 contextes validation tirés au hasard, puis 128 contextes longs. Les deux strates peuvent se recouvrir et ne représentent pas deux réplications indépendantes. LM1B small/medium/large partagent le même tokenizer. Aucun générateur texte n'a été évalué.

- **LM1B** : aucun niveau constant dans les deux strates ; perplexité marginale des niveaux entre 226.02 et 2717.00 sur l'échantillon aléatoire ; reconstruction hors padding 99.86%.

- **TINYSTORIES** : aucun niveau constant dans les deux strates ; perplexité marginale des niveaux entre 217.98 et 1844.50 sur l'échantillon aléatoire ; reconstruction hors padding 99.98%.

Les permutations dans des blocs de quatre tokens préservent les moyennes brutes à 32 positions, mais modifient les codes du tokenizer natif. Il existe donc bien une invariance locale de l'opérateur, sans effondrement correspondant de toute la représentation texte. Cela ne prouve pas que tous les niveaux texte sont utiles, ni que tous les autres checkpoints sont exempts de problèmes.

## Peut-on faire neuf prédictions au lieu de douze sur Sudoku ?

Oui, une variante calculable a été testée, avec une perte observée. Les templates viennent des 1 000 puzzles train. Les niveaux **[5, 2, 11]** sont sélectionnés sur 96 puzzles validation, puis gelés avant les nouvelles interventions test. Leur contribution résiduelle est conservée ; leur prédiction Transformer est évitée.

- **Natif** : 516/600 succès sur test, 219/256 sur externe.
- **Dix prédictions**, niveaux [5, 2] fixés : 515/600 ; 1 erreurs introduites, 0 corrigées.
- **Neuf prédictions**, niveaux [5, 2, 11] fixés : 510/600 ; variation -1.00 point(s), IC 95 % [-2.17, 0.17] points ; 9 erreurs introduites, 3 corrigées ; 72 grilles modifiées.
- **Neuf prédictions sur externe** : 219/256 ; variation 0.00 point(s), IC [-1.95, 1.56] ; 3 erreurs introduites, 3 corrigées.

L'IC recouvrant zéro ne prouve pas l'équivalence. Le seul raccourci exactement fidèle sur ces corpus est le niveau 5 seul. Neuf prédictions ne signifient pas 25 % de temps gagné : aucun benchmark de vitesse n'est revendiqué. Les niveaux 4 et 6 ont été audités, mais n'ont pas été retenus par le critère de fidélité sur validation.

## Ce que l'article peut défendre

Une interaction précise entre données structurées et architecture est identifiée ; son analogue élémentaire de pooling existe sur le texte, mais pas le même niveau constant dans les deux cas testés. Des niveaux très diversifiés peuvent aussi être partiellement compensés après remplacement. La bonne distinction est entre **invariance de l'opérateur**, **faible diversité empirique** et **coût causal d'une prédiction**.

L'article ne reconstitue pas l'entraînement et n'établit pas une pathologie universelle. Les tests Sudoku restent exploratoires puisque ces corpus ont été consultés dans les campagnes précédentes. Une étude causale de généralité demanderait un plan croisant données et architectures avec plusieurs entraînements contrôlés.

## Preuves et reproduction

Les quatre figures sont exportées en PNG/SVG/PDF. Les résultats par contexte et puzzle, histogrammes, matrices de codes et prédictions sont dans `results/`. Les poids et corpus épinglés sont copiés dans `artifacts/` et `data/`. L'archive `sources_generalisation.zip` contient tous les scripts utilisés, les sources des modèles et les configurations ; elle exclut les gros poids. `results/verification.json` donne les contrôles effectivement passés et `results/final_manifest.json` les empreintes finales.

Depuis ce dossier : `./run.sh` pour tout reproduire ; `PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" ../.venv/bin/python scripts/verify.py` pour vérifier les artefacts. Les campagnes précédentes et leurs manifestes restent inchangés.

# Lecture des résultats réels : justesse, contrôles et limites

Complément du 25 octobre 2026 au [rapport principal](RAPPORT.md). Aucun nouveau calcul de réseau : uniquement les argmax des logits conservés. Les mesures de justesse, d'indices et de format étaient prévues au protocole. Leur export séparé par `audit_sorties.py` a été ajouté après lecture des résultats principaux ; il n'appartenait pas au gel initial. Les seuils, analyses principales et diagnostics E restent inchangés. Les comptes ci-dessous sont descriptifs.

## Conclusion de portée

Les interventions ont pu être exécutées sur les vrais tenseurs FP32 avec parité native exacte. Elles apportent une observation fonctionnelle, mais ne départagent pas un calcul adaptatif nécessaire d'une restitution par une suite déjà suffisante dans les cas testés.

Sur les 32 échanges A↔B, **la suite recalculée et la suite native du receveur figée donnent toutes deux 32/32 solutions du receveur**. La suite native du donneur figée donne 32/32 solutions du donneur, donc zéro solution du receveur. Les états avant reprise sont pourtant les mêmes dans chaque comparaison de suites. L'adaptation à l'état échangé n'apporte donc aucun gain d'exactitude finale par rapport au contrôle natif du receveur sur ces échanges.

Le contrôle utilise une suite obtenue auparavant par une génération native complète du receveur : ce n'est pas une réponse gratuite, ni un gain de calcul démontré. Ce résultat ne prouve pas non plus que n'importe quelle suite imposerait sa solution, que l'état serait ignoré ou qu'une régénération serait inutile sur d'autres perturbations. Les différences de confiance existent, comme le montrent les effets NLL du rapport principal.

Les 15/16 diagnostics de préférence dominée par l'entrée restent descriptifs. Dans `sudoku-012`, u/g=1,0069949, hors [0,1] : ce cas reste explicitement non identifiable selon le seuil fixé, sans arrondi ou reclassement opportuniste. **Aucun des 16 cas n'obtient une vérité de mécanisme interne.**

## C : aucun dommage de justesse à réparer à cette dose

- Y00 : 128/128 grilles exactes, 0 violations d'indices, 0 formats de séparateurs incorrects.
- Y10 : 128/128 grilles exactes, 0 violations d'indices, 0 formats de séparateurs incorrects.
- Y11 : 128/128 grilles exactes, 0 violations d'indices, 0 formats de séparateurs incorrects.
- Y01 : 128/128 grilles exactes, 0 violations d'indices, 0 formats de séparateurs incorrects.
- Y1wrong : 128/128 grilles exactes, 0 violations d'indices, 0 formats de séparateurs incorrects.

Les 128 essais proviennent de 16 familles et deux ancrages chacun ; ce ne sont pas 128 unités indépendantes. Toutes les conditions conservent les 89 tokens de leur solution propre, y compris les séparateurs. Des logits et des codes peuvent changer sans modifier l'argmax final. Le test C n'établit donc pas une réparation après dégradation importante ; il établit une robustesse de justesse pour cette dose et ces supports, et une équivalence du dommage moyen de NLL dans la marge préfixée. Il ne faut pas augmenter la dose dans ce même pilote pour chercher une récupération favorable.

Les valeurs suivantes évitent les arrondis à zéro des petits nombres dans le rapport principal :

- D : 1.7471847e-06 nat, IC 98,75 % [-7.4452658e-07 ; 6.316694e-06].
- R : -7.4587439e-06 nat, IC 98,75 % [-2.9707048e-05 ; 1.2351563e-07].
- I : 9.4746686e-07 nat, IC 98,75 % [-2.8300379e-07 ; 4.07239e-06].
- S : 9.6795144e-07 nat, IC 98,75 % [-2.9255971e-07 ; 4.1789085e-06].

Toutes sont très inférieures à la marge de 0,01 nat fixée avant inférence. L'équivalence porte sur la moyenne et cette marge, pas sur une invariance exacte des logits ou de chaque entrée.

## E : toutes les suites figées et recalculées

Les comptes portent sur les deux échanges hors diagonale de chacune des 16 familles. `fixed_left` fige la suite A, `fixed_right` la suite B ; leur résultat agrégé ne doit pas être confondu avec un appariement systématique au receveur. Les deux lignes `frozen_recipient_native` et `frozen_donor_native` réalignent ces mêmes conditions déjà prévues dans le plan factoriel, sans sélectionner de nouvelles sorties.

- `fixed_left` : solution du receveur 16/32 ; solution du donneur 16/32 ; violation d'au moins un indice du receveur 16/32.
- `fixed_right` : solution du receveur 16/32 ; solution du donneur 16/32 ; violation d'au moins un indice du receveur 16/32.
- `regenerated` : solution du receveur 32/32 ; solution du donneur 0/32 ; violation d'au moins un indice du receveur 0/32.
- `frozen_recipient_native` : solution du receveur 32/32 ; solution du donneur 0/32 ; violation d'au moins un indice du receveur 0/32.
- `frozen_donor_native` : solution du receveur 0/32 ; solution du donneur 32/32 ; violation d'au moins un indice du receveur 32/32.

Tous ces résultats restent des Sudoku complets valides au sens des 27 unités et du format. Une sortie de donneur peut néanmoins violer l'indice discriminant du receveur : la validité d'une grille seule n'est pas la résolution de son puzzle d'entrée.

Dans les échanges A↔Aplus, la référence reste identique : les 32 sorties sont exactes pour chacune des trois politiques de suite (native du receveur, native du donneur, recalculée), sans violation des indices, chiffres ou séparateurs. Les variations de probabilités et NLL du rapport principal ne sont donc pas des changements de solution.

## Ce qui reste non identifié

Les données restent compatibles avec plusieurs réalisations internes utilisant les indices et les niveaux tardifs. Elles ne prouvent ni un recalcul depuis zéro, ni une compensation spécifique nécessaire, ni un mélange causal à coefficients constants. L'effet descriptif moyen de l'état sur la préférence A/B n'est pas nul ; il ne doit pas être effacé par la réussite des contrôles figés.

Un prochain test devrait être une **nouvelle campagne** préalablement gelée : autre point de coupure ou corruption produisant un dommage utile sur développement indépendant, puis évaluation sans sélection sur les cas effectivement abîmés. Aucun de ces essais supplémentaires n'a été exécuté ici. L'étape 3a s'arrête avec les résultats présents.

# Traçabilité des affirmations du manuscrit

25 octobre 2026. Ce registre distingue les observations antérieures, la consolidation post-hoc et les déductions mathématiques. Il accompagne [PAPER.md](PAPER.md) ; ce n'est pas une nouvelle prérégistration. Les fichiers historiques restent inchangés.

## Calibration analytique — sections 3, 4.1 et 5.1

Sources : [protocole](../benchmark_causal_revision/PROTOCOLE.md), [programmes](../benchmark_causal_revision/controls.py), [accès et méthodes](../benchmark_causal_revision/methods.py), [résumé numérique](../benchmark_causal_revision/results/summary.json), [audit historique](../benchmark_causal_revision/results/verification.json).

- `n_families=52`, `n_cases=344` ; 120 instances de développement et 224 d'évaluation. Les mélanges représentent 84 instances, les cinq autres programmes 52 chacun.
- `scores.trajectory_naive.all.singleton_correct=52` sur 344 ; précision macro par programme 1/6. Le taux de 15,12 % est celui de cette règle sur cette collection, pas une borne générale pour toutes les méthodes observationnelles.
- `scores.trajectory_cautious.all` : couverture 344/344, taille moyenne 6, aucune décision unique.
- `scores.crossed_only.all` : couverture 344/344, taille moyenne 2,3953488, 52 décisions uniques correctes.
- `scores.complete_intervention` : 344/344, dont 224/224 dans `split:evaluation`. La correspondance entre équations et diagnostic est une calibration interne attendue, sans évaluation indépendante de nouvelles méthodes.

L'équivalence observationnelle repose sur la construction des trajectoires, et non sur une différence de taux. Les ensembles compatibles dépendent des champs effectivement exposés par `methods.project`. Si les résidus internes sont rendus accessibles, certaines équivalences changent : relecture et tolérance ont des suites différentes. Compensation depuis l'état, recalcul et mélanges restent identiques à entrée fixe lorsque leur ancre est la même réponse.

## Sudoku et exécution réelle — sections 4 et 5.2

Sources : [familles complètes](../benchmark_causal_revision/data/families.json), [protocole NCM](../benchmark_ncm_reel/PROTOCOLE.md), [adaptateur](../benchmark_ncm_reel/runtime.py), [gel](../benchmark_ncm_reel/results/protocol_frozen.json), [traces natives](../benchmark_ncm_reel/results/native.npz), [audit NCM](../benchmark_ncm_reel/results/verification.json).

Les bases ont exactement deux solutions, vérifiées avec une limite de trois par deux solveurs. Les variantes A et B sont uniques, de même cardinalité, avec une seule case d'indice différente ; Aplus préserve la solution A en ajoutant un indice. Le changement de cardinalité de Aplus reste un facteur de confusion. Les 16 familles sont les unités d'analyse, pas leurs variantes et requêtes individuelles. Elles ne constituent pas un échantillon uniforme de Sudoku.

Dans `summary.json`, les trois branches `O.A`, `O.B`, `O.Aplus` ont chacune 16/16 solutions finales exactes ; les douze lectures incluent les séparateurs dans la condition d'exactitude. La parité native et les shams sont vérifiés au niveau des vrais tenseurs FP32. Ces assertions sont des audits logiciels internes, pas une réplication externe.

## Corruptions C — section 5.3

Sources : [contrastes et intervalles](../benchmark_ncm_reel/results/summary.json), [traces croisées](../benchmark_ncm_reel/results/crossed.npz), [sorties discrètes](../benchmark_ncm_reel/results/output_quality.json), [complément](../benchmark_ncm_reel/COMPLEMENT_SORTIES.md).

Les branches `C.D`, `C.R`, `C.I`, `C.S` du résumé contiennent les moyennes, IC à 95 % et IC ajustés à 98,75 %. Le papier utilise ces derniers pour les quatre critères coprimaires, une marge de 0,01 nat par case vide et 20 000 bootstrap de familles entières. La décision est `mean_frozen_damage_equivalent_within_prespecified_margin`. Le script auxiliaire ne modifie aucun contraste.

Dans `output_quality.C`, chacune des cinq conditions a `n=128`, `exact=128`, sans violation d'indice ni de séparateur. Le dénominateur 128 correspond à 16 familles × 2 ancrages × 4 perturbations ; il ne fournit pas 128 observations indépendantes. L'absence de dommage d'exactitude retire l'occasion de démontrer sa réparation. L'équivalence dans une marge sur la moyenne n'est pas l'identité des logits.

## Échanges E et interprétation — sections 4.3 et 5.4

Sources : [résumé E](../benchmark_ncm_reel/results/summary.json), [12 croisements conservés par comparaison](../benchmark_ncm_reel/results/exchanges.npz), [comptes alignés receveur/donneur](../benchmark_ncm_reel/results/output_quality.json).

`E.label_counts` contient 15 `input_dominant_response_preference_in_this_test` et un `non_monotone_non_identifiable`. Les seuils sont ceux du protocole antérieur : rapport d'entrée au moins 0,8, domaine [0,1], interaction relative au plus 0,2, écart natif supérieur à 0,1 nat, deux sorties natives exactes. `sudoku-012` reste hors domaine avec 1,0069949021. Aucun de ces diagnostics ne dispose d'une vérité de mécanisme interne.

Les moyennes u, v et w ainsi que leurs IC descriptifs sont les champs `input_effect`, `state_effect`, `interaction`. L'identité u+v=g est algébrique ; ces rapports ne sont pas des fractions causales de calcul ou d'information.

Dans `output_quality.E.answer_change` :

- `regenerated.recipient.exact=32` ; `donor.exact=0`.
- `frozen_recipient_native.recipient.exact=32` ; `donor.exact=0`.
- `frozen_donor_native.recipient.exact=0` ; `donor.exact=32`, et 32 violations des indices du receveur.

Ces comptes réalignent les suites déjà prévues, sans nouvelles sorties de réseau. La réussite d'une suite recalculée ne démontre pas ici sa nécessité pour l'exactitude. La suite native du receveur a néanmoins coûté une génération antérieure complète. Les changements de confiance et l'effet d'état ne sont pas niés. Les politiques de suite des échanges A/Aplus produisent chacune 32/32 solutions communes.

L'exporteur de justesse était absent du gel initial, bien que les métriques soient prévues : cette addition post-hoc est déclarée dans le [journal historique](../benchmark_ncm_reel/JOURNAL.md). Elle ne doit pas être décrite comme une nouvelle expérience confirmatoire.

## Consolidation auxiliaire — section 5.5

Sources : [plan post-hoc](PROTOCOLE_AUXILIAIRE.md), [script gelé](auxiliary.py), [empreintes d'entrée](results/auxiliary_frozen.json), [résultats](results/auxiliary.json).

Les nombres de familles distinguées par vecteur complet de codes sont, du niveau 1 au 12 : **6, 1, 5, 6, 0, 11, 11, 16, 16, 16, 16, 16**. Le niveau 5 contient uniquement le code 11. Les préfixes 1..9 et suffixes 10..12 distinguent tous deux les deux entrées dans chaque famille.

L'information mutuelle d'un bit est exacte sur une loi construite à deux atomes équiprobables par famille. Elle n'est ni une estimation populationnelle, ni une mesure du coût d'information commune consommé, ni un résultat sur les codes du tokenizer de solution. Les reconstructions latentes peuvent être non injectives. L'information conditionnelle sachant l'entrée complète est nulle analytiquement pour le chemin argmax déterministe ; ce zéro n'est pas estimé à partir d'un seul exemple par entrée.

Les quatre seuils post-hoc annoncés sont tous rapportés : 0,70 → 15/16 ; 0,80 → 15/16 ; 0,90 → 8/16 ; 0,95 → 4/16. Les conditions d'éligibilité restent inchangées ; aucun seuil nouveau ne remplace 0,80 dans le résultat principal.

## Théorie — section 7

La [note théorique](THEORIE_WYNER.md) donne les hypothèses et preuves. Les valeurs C_W=n−1 pour la parité uniforme et C_W=1 pour le mélange uniforme de deux solutions sont des déductions sur des lois cibles spécifiées. L'énumération n=2..10 vérifie des cas finis, sans remplacer la preuve générale et sans optimiser numériquement une information de Wyner.

La génération argmax sur puzzles uniques n'échantillonne aucune de ces lois non dégénérées à entrée fixée. Le lecteur global interdit de transférer directement une borne de producteurs finaux locaux indépendants. Aucun traitement du pooling n'a été randomisé ou modifié dans l'étape 3a. Le lien « pooling → perte de coordination → compensation par l'entrée » reste donc une hypothèse, pas un résultat.

Le résultat historique de pooling concerne le niveau de longueur 9 (niveau 5), distinct du niveau 9 à 37 positions. Voir le [rapport de correction](../correction_pooling/RAPPORT.md) et le [manuscrit historique](../manuscrit_unifie/MANUSCRIPT.md). L'encodeur de solution n'est pas dans le chemin de génération native.

Les sources primaires consultées le 25 octobre 2026 sont celles citées dans le papier : [MIB](https://proceedings.mlr.press/v267/mueller25a.html), [Tyen](https://aclanthology.org/2024.findings-acl.826/), [NCM v1](https://arxiv.org/html/2610.14162v1), [extension multivariée](https://arxiv.org/abs/1010.3613), [extension conditionnelle](https://www.isiweb.ee.ethz.ch/papers/docu/rgra-alap-mwig-2022-2.pdf), [synthèse de canal](https://arxiv.org/abs/1208.4415), [synthèse exacte](https://arxiv.org/abs/1402.0062). La référence originale de [Wyner](https://doi.org/10.1109/TIT.1975.1055346) est fournie, mais son texte intégral éditeur n'a pas été directement consulté. La recherche bibliographique reste ciblée, sans garantie de nouveauté ni revue systématique.

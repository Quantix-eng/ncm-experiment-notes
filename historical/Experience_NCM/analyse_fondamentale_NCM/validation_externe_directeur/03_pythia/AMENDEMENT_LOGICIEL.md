# Amendement logiciel 1.1 : préserver le noyau LayerNorm natif

25 octobre 2026, après la première exécution de développement, avant toute inférence de test.

Le premier développement respecte son arrêt technique : l'écart maximal entre le lecteur manuel FP32 et le lecteur natif atteint `2.44140625e-4` logit, au-dessus du seuil gelé `2e-4`. Le résidu natif recomposé est exact ; le MLP final est inchangé. Les scripts, protocole, empreintes et observations de cette exécution sont conservés dans `archives/v1_native_parity_failure_20261025T174632Z/`.

La correction ne relève aucun seuil. Elle remplace le calcul manuel `(r-mu)/s*gamma+beta` par le noyau `nn.LayerNorm` effectivement utilisé par le modèle, avec correction du seul dénominateur : `LN(r)+(LN(r)-beta)*(s_r/s-1)`. À facteur natif, le terme ajouté est zéro ; la lecture reprend le noyau natif. En arithmétique réelle, les deux expressions sont égales. L'audit distinct calcule désormais la formule directe en FP64 et tolère les écarts FP32 dans la marge technique déjà annoncée de `2e-4`.

Poids, données, donneurs, têtes, positions, métriques, absence de filtrage, critères scientifiques et seuils techniques restent identiques. Les quatre scripts historiques, les résultats initiaux et le gel original sont archivés. Un nouveau gel local épingle la correction avant de rejouer les 128 séquences de développement. Le test ne s'ouvre qu'après réussite des mêmes contrôles techniques. Le développement est donc connu avant l'amendement ; cette correction logicielle ne constitue pas une prérégistration aveugle du développement.

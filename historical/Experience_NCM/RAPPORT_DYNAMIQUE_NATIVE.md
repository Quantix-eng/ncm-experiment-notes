# Dynamique native du NCM : prédiction et compensation

**Étude locale du 23 octobre 2026 — terminée et relue.** Référence : Sudoku hard, générateur 25,46 M + VQ-VAE 14,90 M, 12 niveaux, CPU FP32 déterministe, batch 8 et quatre threads. Poids, chemin public, code et données des études gelées inchangés ; documents de synthèse actualisés.


**Résultat : aucun gain prédictif établi pour la lecture dynamique locale ; réponse latente directionnellement compensatrice, sans gain fonctionnel établi sous les perturbations testées.** Le contrôle révèle surtout que le décodage final à suite fixe est peu sensible aux perturbations appliquées au niveau 8. Une preuve de surveillance métacognitive native n'est pas obtenue. Le pilote de renforcement s'arrête selon sa règle, sans nouveau corpus de stress.

## Questions et portée

Deux questions sont séparées avant leurs mesures externes :

1. La direction de la contribution latente du niveau 9 aide-t-elle à classer les erreurs qui persisteront jusqu'au niveau 12, au-delà des observables, de l'état latent courant et des amplitudes ?
2. Après remplacement d'un code au niveau 8, régénérer les codes futurs améliore-t-il la résolution par rapport au maintien des codes futurs de la trajectoire propre ?

Une réponse positive à la première question démontre une information accessible à une sonde. Une réponse positive à la seconde démontre une compensation fonctionnelle sous intervention. Ni l'une ni l'autre ne démontre une représentation de second ordre évaluant la fiabilité, ou son utilisation par un mécanisme métacognitif natif.

## Protocole et protection de l'évaluation

[Protocole numérique](experiments/metacognition/dynamics/protocol-v1.json), [commandes de reproduction](experiments/metacognition/dynamics/README.md), [gel des sources, modèles et corpus](audit/metacognition/native-dynamics-v1/lock.json). Le gel est antérieur à la première inférence NCM sur le nouveau corpus. La spécification est locale et versionnée ; ce n'est pas un pré-enregistrement public. Tous les critères sont fixés après connaissance des études antérieures : la recherche globale reste exploratoire, avec évaluation externe réservée pour ces nouvelles comparaisons.

**Données.** Ajustements sur les 1 000 puzzles de l'apprentissage original, sans utiliser ses partitions calibration/test pour choisir les sondes. Nouveau corpus de 2 000 solutions, générées avec graine 20261201 par backtracking randomisé puis retrait glouton jusqu'à 30 indices. Chaque puzzle a une solution unique vérifiée par un solveur de couverture exacte distinct. Exclusion des solutions officielles, des 1 000 solutions de l'étude de correction précédente, du pilote générateur de 32 puzzles et des doublons nouveaux sous relabellisation des chiffres et symétries du carré. Aucune sélection selon les sorties NCM, aucun ajout d'exemples après observation de l'effet. La distribution n'est ni uniforme ni une réplication IID garantie de l'officielle ; tous les isomorphismes et la contamination du préentraînement ne sont pas audités.

**Lecture native.** Après incorporation des codes du niveau 9, `z9` et `z8` ont tous deux 89 positions × 256 canaux. La contribution observée est `delta9 = z9 − z8`. Les coordonnées correspondent à l'entrée du décodeur ; une correspondance sémantique strictement locale avec les cellules n'est pas présumée. Le delta est une transformation déterministe de deux états ; une meilleure prédiction ne serait pas une nouvelle information théorique.

Cinq sondes logistiques régularisées :

- **Visible** : les 388 variables de l'étude précédente, dont probabilités et historique des grilles jusqu'au niveau 9.
- **Statique** : visible + PCA16 du latent courant + normes du latent et du delta, et leur cosinus.
- **Dynamique** : statique + PCA16 du delta.
- **Bruit** : statique + 16 variables normales indépendantes.
- **Permutation** : statique + PCA16 du delta permuté entre les cellules vides d'un même puzzle, avant sélection de la cohorte d'erreurs.

La PCA utilise toutes les cellules initialement vides des seuls puzzles d'ajustement ; les erreurs courantes et leur persistance sont construites séparément avec la solution. Normalisation et PCA sont refaites dans chaque pli. Sélection de λ dans {0,0001 ; 0,001 ; 0,01 ; 0,1 ; 1 ; 10} par moyenne des AUROC au sein des puzzles dans cinq plis groupés ; égalité départagée en faveur de la pénalisation supérieure. Perte logistique avec poids total égal par puzzle contenant des erreurs courantes, intercept non pénalisé. Toutes les variantes ont retenu λ = 0,1. Les scores de sélection sont des résultats de développement, pas une validation indépendante.

**Critère prédictif principal.** Différence dynamique − statique de la moyenne des AUROC au sein des puzzles comprenant des erreurs corrigées et persistantes. La cohorte utilise la référence pour identifier les erreurs au niveau 9 : ce n'est pas un détecteur déployable de toutes les erreurs. Les puzzles uniformes restent dans l'ajustement et les métriques regroupées ; ils ne définissent pas une AUROC au sein du puzzle. Gain utile de cadrage : 0,02, convention pratique.

**Interventions.** Sur les 1 500 premiers nouveaux puzzles, position de code uniforme à graine fixée au niveau 8. Deux remplacements : deuxième code le plus probable dans la distribution propre ; autre code distinct, choisi pour minimiser l'écart de norme de perturbation latente avec le premier. L'appariement discret n'est pas exact : ses écarts et les probabilités propres sont rapportés. Aucun choix n'utilise le succès futur. Un passage fictif force le code déjà choisi par argmax et doit conserver exactement la trajectoire.

Pour chaque remplacement : génération adaptative normale des niveaux 9–12, puis témoin gardant les codes propres des niveaux 9–12 avec le même code modifié au niveau 8. Ce témoin est une intervention mécaniste, pas un système concurrent déployable. Le critère causal principal est la différence appariée de résolution stricte entre adaptation et suite fixe, pour le remplacement par le deuxième code. Vérification du format, des contraintes, des indices et de l'égalité à la solution. Gain utile de cadrage : trois points de pourcentage.

La compensation géométrique est définie par `−⟨E,R⟩ / ||E||²`, où `E` est le changement du latent cumulé au niveau 8 et `R` le changement de reconstruction finale entre adaptation et suite fixe. Une valeur positive mesure une réponse opposée à la perturbation ; elle ne garantit ni une restauration complète ni une amélioration des grilles.

**Incertitude.** 10 000 bootstraps appariés par puzzle. Intervalles à 97,5 % pour les deux critères principaux, correction de Bonferroni nominale de cette famille de deux comparaisons. Les autres contrastes restent secondaires. Intervalles prédictifs conditionnels aux modèles ajustés ; la variance complète de réentraînement n'est pas intégrée.

Les scénarios normaux de dimensionnement sont enregistrés avant les sorties externes dans [le rapport d'ajustement](audit/metacognition/native-dynamics-v1/fit.json). Avec 290 puzzles mixtes supposés et gain 0,02, la puissance conditionnelle varie de 29,5 % à presque 100 % pour des écarts-types appariés de 0,20 à 0,05. Pour un gain causal de 0,03 sur 1 500 puzzles, elle varie de 64,2 % à 99,9 % selon une discordance supposée de 20 % à 5 %. L'effectif relève d'un budget fixé, sans garantie universelle de puissance de 80 %.

## Contrôles réalisés avant évaluation

- Pilote de 32 puzzles d'apprentissage : reconstruction native et référence identiques bit à bit, argmax fictif identique, codes antérieurs inchangés et un seul remplacement au niveau ciblé.
- Refus des codes futurs dans le constructeur natif. Réutilisation du constructeur observable dont la frontière temporelle est déjà testée ; aucun état après le niveau 9 n'entre dans la sonde.
- Tests de non-influence des données tenues à l'écart et des étiquettes futures sur les transformations apprises ; relecture des transformations sauvegardées.
- 29 tests de la suite passent avant l'inférence externe ; poids inchangés pendant les étapes vérifiées. Les hooks temporaires sont retirés et le cache nettoyé.

## Résultats prédictifs externes

Sur les 2 000 puzzles, 39 686 cellules initialement vides sont erronées au niveau 9 ; 2 247 de ces erreurs persistent. Les 258 puzzles mixtes définissent la métrique principale. Au total, 37 439 erreurs courantes sont corrigées (94,34 %) et 303 erreurs nouvelles apparaissent dans les cellules initialement vides qui étaient correctes au niveau 9. Ces nouvelles erreurs ne font pas partie de la cible étudiée.

- Visible : AUROC au sein des puzzles **0,64534**.
- Statique, avec amplitudes : **0,64193**.
- Dynamique : **0,63877**.
- Bruit : **0,64550** ; permutation du delta : **0,64018**.

**Critère principal dynamique − statique : −0,00316, intervalle à 97,5 % [−0,00863 ; +0,00161].** Aucun apport positif de cette lecture dynamique n'est établi ; la borne supérieure exclut le gain de cadrage de 0,02 pour ces prédicteurs et cette population. Elle laisse possible un très petit gain, et ne limite pas toutes les représentations ou familles de sondes.

Secondaires : statique − visible −0,00341 [−0,00828 ; +0,00167] ; dynamique − visible −0,00657 [−0,01426 ; +0,00058]. Les témoins ne soutiennent pas un avantage spécifique du delta. Le fait que le témoin bruit soit légèrement meilleur en estimation ponctuelle n'est pas une preuve que le bruit aide : bruit − statique +0,00357 [−0,00039 ; +0,00761].

Les coefficients de la référence visible sont exactement identiques à ceux de l'étude précédente. Son AUROC différente sur ce nouveau lot (0,64534 contre 0,67325 dans l'ancien corpus externe) ne vient donc pas d'un changement d'ajustement. Ne pas confondre cette variation entre cohortes avec le contraste apparié entre représentations.

Les cinq modèles ont un Brier brut pondéré par puzzle proche de 0,0409. La forte prévalence des corrections rend ce score peu discriminant à lui seul ; aucune calibration ajoutée n'a été ajustée sur le test. Le critère principal reste l'AUROC au sein du puzzle.

Les intervalles restent conditionnels aux coefficients ajustés. La précision observée permet de limiter un gain important pour cette lecture, mais ne remplace pas une étude complète de variabilité d'apprentissage. [Scores et résultats détaillés](audit/metacognition/native-dynamics-v1/external/analysis/report.json).

## Résultats causaux externes

Sur les **1 500 puzzles distincts** fixés avant leurs sorties, la trajectoire propre résout strictement **1 293 puzzles (86,20 %)**. Ces puzzles sont un sous-ensemble des 2 000 précédents, pas un second corpus indépendant de 1 500 solutions.

**Remplacement par le deuxième code le plus probable — critère principal :**

- Suite fixe : **1 294/1 500** ; suite adaptative : **1 296/1 500**.
- Quatre puzzles résolus seulement avec adaptation, deux seulement avec suite fixe.
- Gain **+0,13 point de pourcentage**, intervalle bootstrap à 97,5 % **[−0,20 ; +0,53] point**.
- Exactitude des cellules initialement vides : gain moyen de **+0,020 point**, intervalle à 95 % [−0,026 ; +0,063] point.

Aucun bénéfice fonctionnel de l'adaptation n'est établi. Le gain de trois points fixé comme utile est exclu par cet intervalle pour cette intervention. Cela ne démontre pas une incapacité générale à réparer : même avec suite fixe, le remplacement ne dégrade pratiquement pas le nombre de résolutions par rapport à la trajectoire propre.

Avec seulement six discordances, le bootstrap ne doit pas être surinterprété. Une sensibilité exacte ajoutée après l'analyse donne un test apparié bilatéral à **p = 0,6875** et un intervalle conservateur d'au moins 97,5 % **[−0,55 ; +0,81] point**, obtenu par différence de bornes marginales de Clopper–Pearson avec Bonferroni. Elle conserve la même conclusion. Ce complément post-hoc ne remplace pas le critère préspécifié.

**Géométrie :** coefficient de compensation moyen **0,254**, intervalle à 95 % [0,235 ; 0,272] ; cosinus compensateur moyen 0,157. La réponse a une composante opposée à la perturbation. Mais le rapport norme de l'écart final / norme de la perturbation initiale vaut **1,55 en moyenne, 1,36 en médiane** : la distance totale n'est pas contractée en moyenne. Il serait incorrect de traduire le coefficient en « 25 % des erreurs corrigées ». En moyenne, 3,00 des 243 codes futurs changent ; pour 435 puzzles, aucun ne change.

**Remplacement distinct apparié approximativement en norme — secondaire :** suite fixe 1 292 résolutions, adaptation 1 294 ; gain +0,13 point [−0,13 ; +0,47] à 97,5 %. Compensation moyenne 0,051 [0,043 ; 0,060] ; rapport des normes moyen 1,40. Ces résultats ne constituent pas une réplication indépendante du premier traitement.

L'écart relatif d'appariement des normes a une médiane de 1,32 %, un quantile 90 % de 8,13 % et un maximum de 22,71 %. Les probabilités propres médianes des codes forcés sont respectivement 0,00584 et 7,86e−8. Être le deuxième candidat n'assure donc pas une perturbation dans la distribution habituelle ; la norme proche ne rend pas les contenus équivalents. On ne peut attribuer la différence de réponse géométrique à une évaluation interne de fiabilité.

Les interventions peuvent aussi améliorer une trajectoire : le premier traitement récupère cinq échecs propres et dégrade deux succès propres. Ce constat ne prouve ni une bonne politique d'exploration ni un bénéfice au même budget face à d'autres candidats.

[Résultats et comptes](audit/metacognition/native-dynamics-v1/causal/report.json), [descriptifs et sensibilité exacte post-hoc](audit/metacognition/native-dynamics-v1/causal/descriptive-diagnostics.json).

## Renforcement et diagnostic : arrêt justifié sur apprentissage

Le faible effet fonctionnel motive un **suivi exploratoire distinct**, défini après les résultats précédents. Sur les 128 premiers puzzles de l'apprentissage archivé, on remplace 1, 4, 8, 13 ou 26 codes du niveau 8 par des alternatives uniformes. Positions et valeurs sont emboîtées à graine fixée. La règle est de retenir la plus petite dose qui fait baisser la résolution de la **suite fixe** d'au moins dix points. Aucun résultat de génération adaptative n'est calculé pour sélectionner la dose.

Le témoin propre résout 106/128 puzzles. Les suites fixes en résolvent respectivement **105, 105, 105, 104 et 105**. Aucune dose n'atteint la baisse prévue ; **arrêt du suivi, aucune génération de corpus externe de stress, aucune mesure adaptative de ce suivi**. Le code conditionnel non exécuté n'est pas présenté comme une expérience validée. Cette règle est un critère pratique de faisabilité sur un petit lot, pas une preuve d'invariance générale.

Un diagnostic supplémentaire, post-hoc sur les mêmes 128 puzzles, décode les contributions natives archivées sans relancer le générateur :

- Reconstruction complète : 106 résolutions, 98,23 % de cellules correctes.
- Contribution du niveau 8 retirée : 106 résolutions ; grille entière identique à la référence pour 126/128 puzzles.
- Dernier niveau retiré : zéro résolution ; 82,02 % de cellules correctes.
- Dernier niveau seul : zéro résolution ; 96,44 % de cellules correctes.
- Niveaux 9–12 seuls : zéro résolution ; 97,44 % de cellules correctes.

La contribution du niveau 8 n'est pas un bon levier fonctionnel pour ce stress. Le dernier niveau est essentiel et ne suffit pas seul à reconstruire exactement ces puzzles. **Aucune conclusion d'inutilité des premiers niveaux pour générer les codes finaux** : ce diagnostic garde les codes générés avec leur conditionnement initial et ne teste que leur contribution directe au décodage.

[Pilote et règle d'arrêt](audit/metacognition/native-stress-v1/pilot.json), [diagnostic de décodage](audit/metacognition/native-stress-v1/codec-training.json), [relecture des comptes](audit/metacognition/native-stress-v1/verification.json), [description du suivi](experiments/metacognition/dynamics_stress/README.md). Tous ces résultats supplémentaires portent sur l'apprentissage, sans statut confirmatoire externe.

## Vérification, anomalies et ressources

[Relecture arithmétique indépendante](audit/metacognition/native-dynamics-v1/verification.json) : **52 fichiers contrôlés**, scores reproduits avec écart maximal nul, AUROC recomputées par comparaisons directes des paires, intervalles recomputés, résolutions strictes et géométrie du sous-ensemble archivé vérifiées. L'écart maximal entre la perturbation initiale et l'effet final à suite fixe est inférieur à 7,4e−7 sur l'échantillon de vérification, compatible avec l'accumulation FP32. C'est une vérification dans le même environnement, pas une réplication par une autre équipe.

- **Pas de divergence numérique de l'instrumentation** sur les 250 lots externes. Les 1 500 passages fictifs causaux conservent la référence bit à bit ; poids inchangés en fin d'exécution.
- **Traçabilité du pilote :** l'archivage des vecteurs causaux a été ajouté après le premier pilote, avant le gel externe. Le pilote initial est conservé ; un [replay de 32 puzzles avec les sources finales](audit/metacognition/native-dynamics-v1/technical-replication-final-source/technical.json) passe aussi. Les états et les poids des études précédentes restent inchangés.
- **Premier audit interrompu par une comparaison trop stricte** : les intervalles recomputés à partir d'AUROC par paires diffèrent de ceux obtenus par rangs d'au plus **1,74e−17**. Tentative conservée dans [le journal d'audit](audit/metacognition/native-dynamics-v1/verification-attempt-01.json). La relecture indépendante utilise une tolérance absolue de 1e−12 ; aucun score, coefficient ou résultat principal n'a été modifié.
- **Pression du système** : environ 3,8 Gio libres initialement, réserve observée descendant à environ 151 Mio ; mémoire résidente du processus causal observée autour de 384 Mio, sans mesure de pic. Les artefacts nouveaux représentent environ 620 Mio. Une tentative de compression transparente n'a montré aucun gain d'allocation ; les octets et empreintes sont restés identiques. Aucun fichier de recherche antérieur ou donnée utilisateur supprimé. Les relectures ont été découpées en blocs de 1 024 cellules. L'origine de la pression globale n'est pas établie.
- **Faible portée fonctionnelle du stress** : six discordances seulement sur 1 500 et échec du seuil de dommage sur apprentissage. C'est une limite majeure du test causal, pas une démonstration d'absence de réparation native.

Les 29 tests passent. Les sources, coefficients, graines, corpus et prédictions nécessaires à la reproduction sont conservés ; les gros tableaux restent locaux, exclus de Git. La génération du corpus a pris environ 111 s, le replay natif d'apprentissage 6 s, l'ajustement environ 48 s et l'extraction externe environ 465 s. Le temps causal est consigné dans son manifeste. Ces durées sur une machine partagée sous pression ne sont pas un benchmark comparatif.

## Positionnement scientifique

La [source NCM, §3.4 et annexe F.3](https://arxiv.org/html/2610.14162v1) décrit déjà une corruption des codes de conditionnement à l'entraînement. Une résistance au bruit ne serait donc pas la découverte d'un principe inédit. L'export Sudoku ne permet pas de déterminer l'effet causal de cet entraînement sans checkpoint témoin. [AID-VAR](https://arxiv.org/abs/2605.24843) étudie une correction inter-échelles avec diagnostic auxiliaire dans la génération d'images ; notre étude porte sur le comportement natif du checkpoint Sudoku figé. Voir [les notes bibliographiques ciblées](audit/metacognition/native-dynamics-v1/literature-notes.md).

L'apport visé est une dissociation entre **direction latente prédictive**, **compensation géométrique** et **réparation fonctionnelle**, avec témoins et corpus réservé. Une revendication de métacognition demanderait encore de dissocier traitement du contenu, détection générique d'anomalie et évaluation de sa propre fiabilité.

## Conclusion et prochaine direction

Cette campagne fournit une **dissociation mesurée entre réponse latente et bénéfice fonctionnel**. La lecture dynamique locale PCA16 n'améliore pas la prédiction des erreurs persistantes. Les futurs niveaux répondent directionnellement aux perturbations, mais le gain de résolution n'est pas établi ; la robustesse du décodage à la perturbation du niveau 8 rend cette expérience peu informative sur une réparation sémantique difficile. Aucun mécanisme de surveillance métacognitive native n'est démontré, et son absence générale ne l'est pas davantage.

Les résultats sont exploitables pour **éviter une fausse interprétation métacognitive de la compensation latente** et pour choisir de meilleures interventions sur les NCM. Ils ne justifient ni d'annoncer une découverte positive de métacognition, ni de déclarer toute l'architecture incapable de surveillance.

Je recommande de conserver ce dossier comme rapport technique exploratoire. Avant une nouvelle étude, il faudra identifier sur développement une intervention qui induit effectivement des erreurs dans un circuit pertinent pour les décisions finales, puis dissocier contenu de la réponse et évaluation de sa fiabilité. Une nouvelle évaluation réservée sera nécessaire pour une nouvelle méthode sélectionnée ; les corpus examinés ici sont désormais consultés. Multiplier les couches, sondes non linéaires ou fine-tunings sur les mêmes tests ne résoudrait pas ce problème de définition.

**Arrêt de cette campagne : deux critères externes terminés et relus, suivi de stress arrêté selon sa règle, ressources système trop contraintes pour une nouvelle étude lourde.** Aucun nouveau corpus de stress, comparaison d'architectures, contrôle de calcul ou fine-tuning n'a été lancé.

La [fiche de clôture de campagne](audit/metacognition/native-dynamics-v1/campaign-summary.json) relie protocoles, résultats, relectures et documents courants par empreintes.

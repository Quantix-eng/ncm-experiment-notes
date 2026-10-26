# Dynamique temporelle NCM : corrections utiles, sans verrouillage au niveau 9

Étude exploratoire, 24 octobre 2026. Checkpoints Sudoku publiés, gelés, CPU FP32. La campagne de correction du pooling est conservée intacte. [Note anglaise](PAPER.md), [PDF](PAPER.pdf), [protocole gelé](OBJECTIFS_ET_PROTOCOLE.md).

## Résultat principal

**L'hypothèse d'un verrouillage strict des valeurs décodées au niveau 9 est réfutée par les trajectoires natives.** Sur les 600 puzzles hard test, 3088 des 12333 cases incorrectes à 8 puis corrigées à 9 redeviennent incorrectes à 10, 11 ou 12 : **25.04 %**, IC 95 % par puzzle **[24.29, 25.83]**. Sur les 2 000 hard, le même constat concerne 10684/41620 corrections. Cela ne réfute pas toute hypothèse d'engagement dans un état interne non observé ; cela réfute le verrouillage proposé des valeurs lues dans la grille.

L'arrêt à 8 donne 0/600 grilles exactes, contre 516/600 à 12. Mais les arrêts à 9, 10 et 11 donnent eux aussi zéro. Le contraste arrêt-8 moins final vaut -86,00 points ; IC [-88.67, -83.17]. Ce test montre l'insuffisance du préfixe, sans isoler le rôle de 9. L'omission du seul résidu 9 avec régénération de la suite donne 103/128, contre 105/128 nativement : 16 réponses et 1 962 codes sont modifiés, mais la baisse de réussite de 1,56 point a un IC [-3,91 ; 0,00]. L'intervention établit une influence causale sur certaines réponses, sans démontrer un avantage populationnel de performance ni une nécessité générale du niveau 9.

## 1. Questions et critères de réfutation

Nous séparons cinq questions : continuer est-il nécessaire ; le résidu 9 contribue-t-il à la réponse ; le débit de correction s'accélère-t-il ; les valeurs se verrouillent-elles ; la réponse devient-elle moins sensible aux perturbations ? Aucune réponse isolée n'établit une transition de phase.

Une trajectoire déterministe à douze niveaux, dont les longueurs et les poids diffèrent, ne fournit pas une famille de tailles et un paramètre de contrôle permettant une analyse de taille finie. Une cassure ajustée à douze points est donc décrite comme une courbure, pas comme une transition fondamentale. Plus de puzzles augmente la précision sur la population d'entrées ; cela n'augmente pas la taille du système.

Le pic de changements argmax peut combiner ajout d'information et franchissement de seuils du décodeur. « Décision fonctionnelle » et « artefact du décodeur » ne forment pas une alternative exhaustive. Nous n'attribuons ni intention ni métacognition aux changements de grille.

## 2. Mécanisme observé et données

Le runner explicite suit la récurrence `c_k = argmax G_k(pool_k(z_(k-1)), puzzle)`, puis `z_k = z_(k-1) + R_k(c_k)` et `grille_k = argmax D(z_k)` ; le premier niveau utilise une requête initiale dédiée. `R_k` inclut l'embedding, le rééchantillonnage et la transformation résiduelle du niveau. Les prédictions de codes anciens ne sont jamais révisées sur la trajectoire native : leur fixation est une propriété du protocole à **tous** les niveaux, pas une découverte au seul niveau 9. En revanche, chaque ajout peut modifier la lecture de toute la grille. Il faut donc distinguer fixation des codes déjà émis et verrouillage des valeurs Sudoku décodées.

L'observable principal est le ratio `nombre(c8 faux, c9 correct, au moins un c10..12 faux) / nombre(c8 faux, c9 correct)`, où c désigne ici la valeur de case décodée, distincte des codes VQ de la récurrence. Ce ratio mesure une propriété falsifiable des trajectoires, sans supposer une décision latente.

### Données, exécution et incertitude

Réutilisation explicite des traces natives vérifiées de la campagne initiale : hard/easy/medium, 2 000 puzzles chacun, test hard fixé de 600, et 256 puzzles externes. Ce ne sont pas de nouvelles données aveugles : ces jeux ont déjà été examinés. Les difficultés officielles partagent leurs familles de solutions ; les réplications ne sont pas indépendantes à ce niveau. Les interventions utilisent les 128 premiers puzzles du test hard fixé ; les bruits les 64 premiers, sans sélection selon le résultat.

À l'arrêt k, nous décodons l'accumulation des contributions 1..k, à 89 positions. Il ne s'agit ni du seul bloc k ni d'un remplissage des niveaux manquants par le code d'indice zéro, dont l'embedding n'est pas nul. Le runner explicite est identique au chemin natif sur 32 puzzles pour les codes, logits finaux et accumulations 8/9 ; les 128 trajectoires causales sont aussi comparées aux traces natives sauvegardées. Le décodage au niveau 8 est contrôlé directement.

Les taux de revirement portent sur les 81 cases, indices compris ; les transitions sur cases initialement vides sont également conservées dans le JSON. La réussite exacte porte sur les 89 tokens et est doublée d'un contrôle sur les 81 cases. À tous les arrêts 1..11, **aucun** des 600 puzzles n'a ses 81 cases entièrement exactes : l'échec précoce ne provient pas des séparateurs. Les sorties intermédiaires sont celles du décodeur de solutions complètes, pas celles d'un lecteur optimal entraîné pour chaque niveau ; leur pauvreté ne prouve pas l'absence d'une solution décodable autrement.

Bootstrap percentile apparié, 2 000 tirages par puzzle, poids et modèles fixes. Pour un ratio de corrections, numérateur et dénominateur sont rééchantillonnés ensemble. Pour le bruit, les trois graines sont moyennées dans chaque puzzle avant le bootstrap. Les contrastes sont exploratoires, sans correction de multiplicité. Le protocole est gelé avant cette campagne, après les observations des campagnes antérieures : pas de revendication confirmatoire indépendante.

## 3. Arrêts temporels et trajectoires natives

- Niveau 1 : 0/600 grilles ; 14.53 % de cases exactes ; médiane 69 erreurs.
- Niveau 2 : 0/600 grilles ; 14.44 % de cases exactes ; médiane 69 erreurs.
- Niveau 3 : 0/600 grilles ; 16.79 % de cases exactes ; médiane 67 erreurs.
- Niveau 4 : 0/600 grilles ; 20.29 % de cases exactes ; médiane 65 erreurs.
- Niveau 5 : 0/600 grilles ; 20.30 % de cases exactes ; médiane 65 erreurs.
- Niveau 6 : 0/600 grilles ; 27.82 % de cases exactes ; médiane 58 erreurs.
- Niveau 7 : 0/600 grilles ; 33.26 % de cases exactes ; médiane 54 erreurs.
- Niveau 8 : 0/600 grilles ; 45.31 % de cases exactes ; médiane 44 erreurs.
- Niveau 9 : 0/600 grilles ; 61.51 % de cases exactes ; médiane 31 erreurs.
- Niveau 10 : 0/600 grilles ; 75.93 % de cases exactes ; médiane 19 erreurs.
- Niveau 11 : 0/600 grilles ; 81.68 % de cases exactes ; médiane 14 erreurs.
- Niveau 12 : 516/600 grilles ; 98.10 % de cases exactes ; médiane 0 erreurs.

![Arrêts natifs](figures/01_native_stops.png)

Une métrique exigeant 81 cases correctes est conjonctive : une amélioration continue des cases peut produire une augmentation tardive abrupte des grilles exactes. Le saut final ne localise donc pas à lui seul une décision discrète.

Exemple vérifiable : puzzle source **938**, ligne **2**, colonne **4**, référence **7**. Les valeurs aux niveaux 8..12 sont **6 → 7 → 7 → 4 → 7**. La case est corrigée à 9, dégradée à 11, puis récupérée à 12. Le puzzle et onze autres contre-exemples figurent dans `results/observations.json`.

Parmi les corrections de 9 du test, 55 restent incorrectes au dernier niveau, soit 0.45 %, IC [0.30, 0.62]. Les revirements transitoires sont donc beaucoup plus fréquents que les erreurs persistantes. 50.50 % de toutes les valeurs lues à 9 changent encore au moins une fois ; aucune des 600 grilles lues à 9 n'est identique à sa sortie finale. Le nombre de grilles déjà résolues à 9 puis perdues est zéro, mais ce contrôle est vide puisque aucune grille n'était résolue à 9.

- hard_all, n=2000 : 1726 grilles exactes finales ; 10684/41620 corrections à 9 redeviennent incorrectes, soit 25.67 %, IC [25.26, 26.09].
- hard_test, n=600 : 516 grilles exactes finales ; 3088/12333 corrections à 9 redeviennent incorrectes, soit 25.04 %, IC [24.29, 25.83].
- easy_all, n=2000 : 1988 grilles exactes finales ; 7874/39790 corrections à 9 redeviennent incorrectes, soit 19.79 %, IC [19.38, 20.19].
- medium_all, n=2000 : 1928 grilles exactes finales ; 11006/40290 corrections à 9 redeviennent incorrectes, soit 27.32 %, IC [26.90, 27.73].
- external, n=256 : 219 grilles exactes finales ; 1362/5253 corrections à 9 redeviennent incorrectes, soit 25.93 %, IC [24.78, 27.06].

## 4. Un pic de corrections réel, sans preuve de transition de phase

Le pic historique de 41 620 événements sur hard est un nombre brut de passages incorrect vers correct entre 8 et 9, et non 41 620 décisions irréversibles ni 41 620 puzzles. Le nombre net doit soustraire les dégradations simultanées. Les comptes bruts dépendent également du nombre de cases, de puzzles et du budget de codes par niveau ; une version par code ajouté est conservée.

Le gain net de cases exactes entre 8 et 9 dépasse la moyenne des gains 7→8 et 9→10 de **2.96 points**, IC [2.30, 3.58], sur test. Cette accélération locale ne s'accompagne pas du verrouillage proposé.

![Corrections et contre-exemple](figures/02_corrections_reversals.png)

Une droite segmentée continue, sélectionnée parmi huit nœuds possibles sur les mêmes douze points, place sa meilleure cassure au niveau **6** en fonction du niveau, et **10** en fonction du nombre cumulé de codes. Les préférences bootstrap décrivent la stabilité de cet ajustement conditionnel ; elles ne constituent pas un test de transition. La comparaison favorise par construction un modèle plus flexible et ne valide pas une singularité.

![Courbures descriptives](figures/06_descriptive_bends.png)

## 5. Ablations temporelles : contribution utile, intervention imparfaite

Nous calculons normalement la prédiction du niveau ciblé, omettons uniquement sa contribution résiduelle, puis régénérons les niveaux suivants à partir de l'accumulation modifiée. L'horaire, les autres poids et le nombre de têtes exécutées sont conservés. L'omission n'est donc pas une mesure de gain de temps d'un modèle compressé. Baseline : **105/128** grilles exactes.

- Contribution 8 omise : **106/128** ; écart 0.78 points, IC 95 % [-1.56, 3.91] ; 1 échecs introduits, 2 corrigés ; 16 grilles modifiées ; norme relative moyenne retirée 0.727.
- Contribution 9 omise : **103/128** ; écart -1.56 points, IC 95 % [-3.91, 0.00] ; 2 échecs introduits, 0 corrigés ; 16 grilles modifiées ; norme relative moyenne retirée 0.814.
- Contribution 10 omise : **103/128** ; écart -1.56 points, IC 95 % [-4.69, 1.56] ; 3 échecs introduits, 1 corrigés ; 21 grilles modifiées ; norme relative moyenne retirée 0.634.
- Contribution 11 omise : **104/128** ; écart -0.78 points, IC 95 % [-2.34, 0.00] ; 1 échecs introduits, 0 corrigés ; 15 grilles modifiées ; norme relative moyenne retirée 0.493.
- Contribution 12 omise : **0/128** ; écart -82.03 points, IC 95 % [-88.28, -75.00] ; 105 échecs introduits, 0 corrigés ; 128 grilles modifiées ; norme relative moyenne retirée 1.274.

L'omission de 9 conserve l'essentiel de la réussite ; les niveaux suivants peuvent récupérer malgré le résidu absent. La contribution finale domine ce test d'omission, mais son rôle ne prouve pas non plus une transition à 12. Les normes retirées diffèrent entre niveaux. Les écarts ne sont pas un classement parfaitement contrôlé de l'importance spécifique des têtes : supprimer un résidu de grande amplitude peut être plus perturbant. Les interventions peuvent produire des états hors distribution. Elles identifient l'effet de cette suppression précise sur ces puzzles, pas une nécessité universelle ni un mécanisme de décision.

Reprise des niveaux 10..12 depuis z8 + gamma R9, où R9 est la contribution issue des codes natifs de 9 :

- Gamma 0.0 : 103/128 grilles exactes ; 16 réponses modifiées.
- Gamma 0.5 : 105/128 grilles exactes ; 12 réponses modifiées.
- Gamma 1.0 : 105/128 grilles exactes ; 0 réponses modifiées.
- Gamma 1.5 : 105/128 grilles exactes ; 6 réponses modifiées.

Gamma 1 reproduit exactement les codes et sorties natifs ; gamma 0 reproduit exactement l'omission du résidu 9. Gamma 1,5 extrapole hors de l'intervalle natif et reste diagnostique.

![Interventions temporelles](figures/04_temporal_interventions.png)

## 6. Contrôle de lecture fractionnaire

Sur les mêmes 128 puzzles, nous conservons les codes natifs de 9 et décodons 21 points z8 + gamma R9, sans continuation. De gamma 0 à 1, l'exactitude des cases passe de 45.97 % à 61.54 % ; la NLL moyenne du chiffre de référence passe de 1.3695 à 0.9982 nat. Les logits, argmax et marges sont sauvegardés pour chaque point.

![Interpolation de lecture](figures/03_reader_interpolation.png)

Cette interpolation examine comment le lecteur transforme une contribution fixe en prédictions discrètes. Les changements argmax sont des franchissements de frontières de lecture. L'interpolation ne remplace pas une trajectoire native avec des codes recalculés, ne teste pas toutes les discontinuités des têtes argmax et ne prouve pas que le niveau 9 serait sans fonction. Le test causal précédent répond séparément à cette dernière question.

## 7. Susceptibilité avant/après 9

Une perturbation gaussienne isotrope normalisée est ajoutée juste avant ou après la contribution 9, puis la suite est régénérée. Pour un puzzle, une dose et une graine, la direction et la norme absolue sont exactement appariées ; la norme est dose × norme(z8) dans les deux cas. Le ratio à norme(z9) n'est donc pas apparié. Avant 9, la tête 9 est recalculée ; après 9, ses codes sont conservés. C'est une comparaison de sensibilité du calcul restant, pas une recherche exhaustive d'attracteurs.

- Dose 1.00 % de norme(z8) : réponses modifiées avant/après 1.04 %/0.52 % ; différence après moins avant -0.52 points, IC [-1.56, 0.00]. Réussite avant/après 84.38 %/84.38 % ; contraste de réussite 0.00 points, IC [0.00, 0.00].
- Dose 5.00 % de norme(z8) : réponses modifiées avant/après 3.12 %/2.60 % ; différence après moins avant -0.52 points, IC [-1.56, 0.00]. Réussite avant/après 84.38 %/84.38 % ; contraste de réussite 0.00 points, IC [0.00, 0.00].

![Susceptibilité appariée](figures/05_susceptibility.png)

Les IC [0,0] issus d'aucun changement de statut sont dégénérés et ne prouvent pas une équivalence populationnelle. Le taux de réussite natif des 64 puzzles est 54/64 ; aucun bruit testé ne change ce statut. Deux doses, trois graines et 64 puzzles ne suffisent pas à caractériser toutes les perturbations ou un bassin global. Une invariance à ce bruit ne prouverait pas l'irréversibilité ; une réponse modifiée ne serait pas à elle seule une inversion de décision. Les revirements natifs constituent le test direct du verrouillage des valeurs.

## 8. Conclusion et suite justifiée

Le résultat défendable est une **correction distribuée, non monotone, avec accélération locale autour de 9 et corrections substantielles ultérieures**. Le niveau 9 contribue au calcul, mais les données réfutent son interprétation comme verrouillage strict de la grille. Aucun résultat de cette campagne n'établit une transition de phase architecturale, ni une décision interne consciente ou métacognitive. L'étude ne généralise pas au texte, où ces interventions temporelles n'ont pas été exécutées.

Une suite pertinente serait de comparer des modèles réentraînés à plusieurs tailles et budgets de niveaux, avec un paramètre de contrôle annoncé, puis de chercher un ordre macroscopique et une mise à l'échelle reproductible. Sans réentraînement, on peut chercher un engagement latent plus faible par patching de candidats cohérents et tests de récupération, avec une définition préalable distincte des argmax de grille. Ces pistes restent des propositions, pas des résultats acquis.

## 9. Preuves et reproduction

Depuis ce dossier, `./run.sh` rejoue les observations, les interventions, la publication et la vérification. Les poids, sources et jeux de données antérieurs sont lus dans le dossier parent autorisé ; aucune écriture ne les modifie. Le runner et le code de campagne sont dans `scripts/`. Les traces sources sont les NPZ du parent, explicitement réutilisées ; les nouvelles réponses par intervention et les logits de lecture sont dans `results/interventions.npz` et `results/reader.npz`.

`results/verification.json` précise les contrôles réellement exécutés ; `results/final_manifest.json` fige les artefacts livrés. `sources/previous_manifests.json` lie cette campagne aux quatre états antérieurs vérifiés. `sources_temporelles.zip` archive les scripts utilisés, le code des modèles, les configurations, les protocoles et les archives de sources des quatre campagnes antérieures ; les gros poids et traces restent dans le parent avec leurs empreintes. Le rapport ne remplace pas les données par des exemples sélectionnés : tous les comptes et toutes les réponses des sous-échantillons fixés sont conservés.

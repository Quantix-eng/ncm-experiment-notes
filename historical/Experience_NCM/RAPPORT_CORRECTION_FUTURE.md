# NCM — anticipation des corrections, 23 octobre 2026

État consolidé : [synthèse des deux études](SYNTHESE_METACOGNITION.md).

Priorité choisie par l'utilisateur : déterminer quelles erreurs le NCM corrigera ensuite. Protocole exploratoire [correction-v0.1](experiments/metacognition/correction/protocol-v0.1.json), fixé après l'audit des labels d'apprentissage et avant tout entraînement de cette sonde. Le précédent test H1 n'est pas utilisé pour choisir ou évaluer cette nouvelle méthode.

## Cible et pièges détectés

Sur les **1 000 puzzles d'apprentissage**, parmi les 51 cellules initialement vides de chaque puzzle :

- **19 691 erreurs au niveau 9**, dont **18 494 corrigées** au niveau 12 et **1 197 persistantes**.
- Répondre toujours « sera corrigée » donnerait **93,9 % d'exactitude**, sans distinguer les erreurs.
- **136 puzzles** ont simultanément des erreurs corrigées et persistantes ; 864 voient toutes leurs erreurs courantes corrigées.
- **142 erreurs nouvelles** apparaissent au niveau final dans des cellules correctes au niveau 9. Elles ne font pas partie de la cible actuelle. Un puzzle finalement faux peut donc avoir corrigé toutes ses erreurs courantes.

La mesure principale est la **moyenne des AUROC au sein de chaque puzzle mixte**, avec un poids égal par puzzle. Étiquette 1 : erreur persistante ; le score de correction est son complément. Le taux de réussite global du puzzle ne suffit pas à classer ses cellules : un score constant par puzzle vaut 0,5 dans cette mesure.

**Limite essentielle :** on utilise la solution pour sélectionner les cellules actuellement erronées et construire les labels, jamais comme entrée. L'évaluation porte donc sur une cohorte définie par un oracle ; elle ne constitue pas un détecteur utilisable en pratique pour reconnaître toutes les erreurs présentes. Les puzzles mixtes sont sélectionnés uniquement pour la métrique ; tous les puzzles d'apprentissage avec erreurs participent à l'ajustement.

## Représentation et apprentissage

Ajouter un vecteur global identique à toutes les cellules d'un puzzle dans une régression linéaire ne peut pas changer leur classement à coefficients locaux fixés : il ajoute le même décalage. Nous utilisons donc un état local : les **256 entrées de `vqvae.decoder.output_layer`, après `ln_f`**, aux positions fixes des cellules, au niveau 9. **Ce sont des états du décodeur VQ-VAE, distincts des résumés du générateur utilisés dans H1.** Un résultat négatif ne limiterait pas toutes les représentations du générateur.

La reconstruction est recalculée à partir des **116 premiers codes seulement**, sans codes futurs. Les probabilités du décodeur reproduisent bit à bit les traces archivées sur les 1 000 puzzles, avec et sans hook ; poids et buffers inchangés. Les variables ne consultent pas les niveaux 10–12.

Référence visible : **388 variables par cellule** — coordonnées, probabilités et tokens locaux des niveaux 1–9, conflits locaux, changements, indices visibles, probabilités des cellules voisines, logits actuels et résumés globaux définis. Normalisation observable avec échelle minimale 0,1. Ajout interne : **16 composantes PCA**, ajustées sur les 51 cellules vides des seuls puzzles d'ajustement, avec standardisation des canaux et des composantes.

Quatre modèles : visible, visible + états, visible + 16 variables normales, visible + états permutés entre les 51 cellules du même puzzle. La permutation conserve le contexte du puzzle mais ne constitue pas un test formel d'indépendance conditionnelle.

Régression logistique régularisée, perte moyenne donnant le même poids total à chaque puzzle ; aucune pondération de classe. Grille λ fixée à 1e-4/1e-3/1e-2/0,1/1/10, choix par AUROC moyenne au sein des puzzles en cinq plis groupés. Tous les prétraitements sont refaits dans l'ajustement de chaque pli. Aucun mélange de cellules d'un même puzzle entre ajustement et validation.

## Premier résultat de développement

799 anciens puzzles d'apprentissage pour l'ajustement et la sélection ; **201 du pli 0 pour la validation**, dont 28 puzzles mixtes. Aucun usage de l'ancien test/calibration H1.

- Référence visible : **0,62075** d'AUROC au sein des puzzles.
- Avec états locaux : **0,61103**.
- Avec bruit : **0,62272** ; états permutés : **0,62331**.
- Différence principale **−0,00972**, intervalle apparié à 95 % **[−0,02522 ; +0,00824]**, 5 000 bootstraps de puzzles mixtes, conditionnels aux modèles ajustés.

Aucun apport interne positif établi sur ce développement. Les 3 982 erreurs de validation, dont 238 persistantes, ne constituent pas autant d'observations indépendantes : l'intervalle rééchantillonne les puzzles. λ=0,1 sélectionné pour les quatre modèles. Relecture numérique sans ajustement réussie, écart maximal 8,89e-16. Extraction locale : 7,12 s ; entraînement et contrôles : 27,01 s. **26 tests passent**, dont invariance aux états futurs, pondération par puzzle et impossibilité pour un décalage global d'améliorer le classement interne.

Un audit supplémentaire par couverture exacte confirme une solution unique pour **1 000/1 000 puzzles officiels d'apprentissage**. Le premier essai d'audit lisait les positions d'une grille de 89 tokens dans un prompt de 91 ; il est conservé sous un nom explicitement invalide et ne constitue pas un résultat sur la solvabilité. La version corrigée passe par le parseur de prompts vérifié.

## Évaluation externe indépendante

Effectif **1 000**, graine **20261106**, fixés dans [le design externe](experiments/metacognition/correction/external-design-v0.1.json) avant génération et inférence. Les 1 000 premiers puzzles acceptés ont chacun 30 indices et une solution unique vérifiée par backtracking et couverture exacte. Exclusion des solutions connues selon relabellisation des chiffres et huit symétries du carré, y compris le précédent pilote de 32 ; pas d'audit exhaustif des isomorphismes ou de contamination du préentraînement.

Génération : **53,79 s**, 1 000 tentatives acceptées, aucun rejet. Corpus haché avant toute inférence. Les quatre prédicteurs finaux sont ajustés sur les 1 000 anciens puzzles d'apprentissage, λ choisi dans les cinq plis d'origine, puis coefficients, transformations, sources et corpus sont gelés dans [le manifeste](audit/metacognition/correction-v0.1/external/lock.json).

**Incident d'implémentation conservé :** le premier lancement construisait des grilles de 89 tokens sans les marqueurs attendus par l'API (91 tokens). Le contrôle de format a arrêté l'exécution avant toute prédiction externe. Source et gel antérieurs archivés ; seule la construction du prompt a été corrigée, modèles et corpus inchangés, puis nouveau gel avant reprise. Le script de re-gel a aussi nécessité la correction de son chemin d'import ; aucune de ces tentatives n'a produit de score externe.

**Évaluation terminée et vérifiée, effectif maintenu à 1 000.** Le corpus neuf contient **19 941 erreurs courantes**, dont **18 617 corrigées** et **1 324 persistantes** ; **160 nouvelles erreurs** apparaissent dans des cellules auparavant correctes. La métrique principale porte sur **145 puzzles mixtes**, et non sur 19 941 observations indépendantes.

- Référence visible : **0,67325** d'AUROC au sein des puzzles.
- Avec états locaux : **0,67265**.
- Avec bruit : **0,67483** ; états permutés : **0,67517**.
- Différence principale **−0,00060**, intervalle apparié à 95 % **[−0,00428 ; +0,00320]**, conditionnel aux prédicteurs ajustés.

**Aucun apport interne établi pour ces 16 composantes et cette famille linéaire** ; la borne supérieure reste nettement sous 0,03. La référence visible permet un classement partiel des corrections futures, mais ce résultat ne démontre pas que le NCM lui-même surveille ses erreurs. Les métriques secondaires ne renversent pas ce constat : Brier pondéré 0,04855 contre 0,04854 avec états, log-loss pondérée 0,19842 contre 0,19835, probabilités non calibrées.

Parité cache exacte sur les 1 000 inférences nouvelles, contrôle eager sur 112, probabilités du replay local identiques aux traces ; empreinte du modèle inchangée. Extraction : **179,88 s**. Maximum absolu des logits des sondes inférieur à 4,20, sans instabilité numérique. Relecture sans ajustement des scores et des 5 000 bootstraps passée. Aucune extension opportuniste de l'effectif. Les probabilités restent brutes ; aucune calibration sur les labels externes. Cette distribution générée n'est pas garantie identique à l'officielle : validation externe exploratoire, pas réplication IID ni garantie de puissance à 80 %.

Preuves : [audit des cibles](audit/metacognition/correction-v0.1/training-label-audit.json), [extraction locale](audit/metacognition/correction-v0.1/training/report.json), [développement](audit/metacognition/correction-v0.1/development/report.json), [relecture](audit/metacognition/correction-v0.1/development/verification.json), [corpus externe](audit/metacognition/correction-v0.1/external/corpus-manifest.json). Archives numériques locales exclues de Git, empreintes conservées ; sources/poids de référence et export public inchangés.

## Portée architecturale et décision suivante

La génération dans `src/ncm_mac/cached.py` utilise le préfixe, les codes et leur reconstruction latente cumulative ; **elle n'appelle pas le décodeur VQ-VAE pour consulter nos grilles intermédiaires**. Leur décodage a été ajouté hors trajectoire pour l'analyse. « Correction » désigne ici le passage d'une cellule erronée dans cette projection au bon chiffre final, pas une décision native d'éditer cette cellule après inspection d'une grille. Les variables dites visibles sont observables par l'expérimentateur ; elles ne sont pas toutes des entrées du générateur.

Cette sonde est donc un test de décodabilité prédictive locale, pas un test direct d'une surveillance native. Son résultat négatif limite la représentation PCA16 choisie du décodeur et la famille linéaire, pas tous les états du générateur, ni toute information des 256 canaux du décodeur. La PCA peut perdre des directions informatives et des interactions peuvent échapper à la sonde.

**Jalon achevé ; décision utilisateur : consolider avant une nouvelle expérience.** Aucun mécanisme suivant n'est sélectionné et aucun nouveau calcul expérimental n'est engagé. Les possibilités précédemment proposées (états du générateur, interventions sur les codes) restent des hypothèses de reprise, pas des résultats ni des actions autorisées par ce choix. Le corpus externe maintenant consulté ne sera pas présenté comme intact pour sélectionner de nouvelles variables. Voir [la synthèse consolidée](SYNTHESE_METACOGNITION.md).

Preuves finales : [résultats externes](audit/metacognition/correction-v0.1/external/analysis/report.json), [vérification indépendante](audit/metacognition/correction-v0.1/external/analysis/verification.json), [audit d'unicité valide](audit/metacognition/correction-v0.1/training-uniqueness.json), [échec de format conservé](audit/metacognition/correction-v0.1/external/format-failure-01/failure.json), [commandes de reproduction](experiments/metacognition/correction/README.md). Aucun calcul ne reste actif.

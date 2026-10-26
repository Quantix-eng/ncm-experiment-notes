# Métacognition NCM — développement du 23 octobre 2026

Rapport de l'essai v0.2 conservé. Suite : [amendement v0.3, stabilité et évaluation tenue à l'écart](RAPPORT_RESULTATS_METACOGNITION.md).

**Extraction des 1 000 puzzles d'apprentissage validée. Premier entraînement des sondes terminé, mais une instabilité du prétraitement empêche une conclusion fiable sur leur apport interne. Calibration et test restent réservés.**

## Travail réalisé

- 1 000 trajectoires hard, 863 succès et 137 échecs : codes, reconstruction, logits et réponses identiques bit à bit au chemin CPU local de référence avec cache ; comparaison eager supplémentaire sur 112 puzzles. Réponses identiques aux archives ; empreinte des poids et buffers inchangée.
- Extraction complète en **195,06 s**, pic RSS du processus **656 Mo**, **215,89 Mo** de traces compressées en 125 lots. Mesures d'une exécution, sans benchmark répété.
- Variables construites au niveau 9 conformément à la spécification conservée. Régressions logistiques FP64, cinq plis externes et cinq plis internes ; sélection de λ, normalisation et PCA refaites dans chaque ensemble d'ajustement. Les observations des plis externes ne participent pas à la sélection de leur prédicteur. Chaque solution hard est unique dans le partage, donc aucun groupe n'est divisé par les plis internes.
- Référence enrichie, ajout des 16 composantes internes, témoin de 16 variables aléatoires, puis témoins puzzle seul, résumés et états permutés. Ces derniers utilisent les mêmes plis et grille. Aucun ajustement ni nouvelle inférence sur les 500 exemples de calibration ou les 500 de test.
- Transformations et coefficients externes sauvegardés ; relecture numérique indépendante des prédictions et vérification des empreintes. **17 tests passent**, dont vérification de l'objectif logistique, des ex æquo et de la reconstruction des transformations sauvegardées.

Le premier ensemble de trois modèles demande **167,68 s**, pic RSS **1,05 Go**. Les ajustements internes s'arrêtent avant le plafond de 500 itérations (maximum observé 131). L'arrêt LBFGS peut dépendre de la variation de l'objectif ; tous les gradients ne passent pas sous 1e-7, maximum enregistré 1,70e-5. Aucun réglage d'optimiseur ajouté après observation.

## Résultat du premier essai, à interpréter comme diagnostic

AUROC regroupant les prédictions externes des 1 000 exemples :

- Référence enrichie : **0,8642**.
- Référence + états internes : **0,8578**, différence **−0,0064**.
- Référence + bruit indépendant : **0,8549**.
- Puzzle seul : **0,5173** ; résumés observables : **0,9012** ; référence enrichie avec états permutés : **0,8644**.

La référence résumée obtient une log-loss de **0,2754**, contre 0,3994 pour le puzzle seul. Les sorties et leur dynamique rendent donc l'échec nettement plus accessible à cet observateur que notre encodage du puzzle seul, en développement. L'ajout de toutes les probabilités brutes détériore ici cette référence simple, ce qui renforce le besoin de traiter son conditionnement. Cela ne démontre aucune utilisation native d'un signal de fiabilité et ne justifie pas de remplacer rétrospectivement le comparateur principal. Le témoin permuté conserve la fragilité de normalisation (log-loss 2 074,91), comme les autres modèles enrichis. Ces trois témoins ont demandé 66,45 s supplémentaires.

Le bootstrap descriptif des scores fixés donne pour la différence un intervalle [−0,0208 ; +0,0059]. **Ce n'est pas l'intervalle du test principal** : cinq modèles différents produisent ces scores et leurs ensembles d'apprentissage se recouvrent ; ce bootstrap ne réentraîne pas les modèles. Surtout, l'anomalie ci-dessous interdit d'utiliser cet intervalle pour exclure un apport interne utile.

Par pli, les différences d'AUROC interne−référence sont +0,0006, +0,0037, +0,0062, +0,0037 et 0,0000. Leur signe diffère donc de la différence regroupée. Une AUROC regroupée compare également des scores issus de modèles distincts : ce résultat n'est pas une contradiction mathématique, mais avertit de problèmes de comparabilité des échelles. Les AUROC d'ajustement de la référence atteignent 1 dans tous les plis, contre environ 0,799–0,914 en validation : écart important avec cette représentation très dimensionnelle, pas preuve de généralisation parfaite.

## Anomalie principale : amplification de probabilités rares

Le protocole v0.2 supprime les colonnes de variance exactement nulle puis divise les autres par leur écart-type d'apprentissage. Il laisse ainsi **421 à 442 colonnes d'écart-type inférieur à 1e-8** selon le pli. Ce ne sont pas des colonnes exactement constantes retenues par erreur : leurs probabilités varient réellement, mais à une échelle minuscule.

Exemple du pli externe 3, colonne 2293 : probabilité du token 4 à la position 79, normalement un séparateur de ligne, au niveau 9. Dans l'ajustement, elle varie entre 5,69e-21 et 1,40e-14, avec écart-type 7,75e-16. Sur le puzzle de validation 89, elle vaut 6,61e-6 : après normalisation, **8,53 milliards**. Une contribution individuelle au logit atteint 23,96 millions ; le logit total maximal de ce pli atteint 1,68 million en valeur absolue. D'autres plis présentent le même phénomène ; il ne s'agit pas d'un cas isolé.

Conséquence : log-loss regroupée **1 712,44** pour la référence et **1 895,43** pour la sonde, malgré des Brier bruts proches de 0,09. Des probabilités saturées donnent un Brier borné mais peuvent subir une pénalité logarithmique arbitrairement grande. Les corrélations très élevées entre scores sont aussi dominées par ces valeurs extrêmes : **ne pas les réutiliser directement pour dimensionner la confirmation**.

Il s'agit d'une fragilité du prétraitement spécifié face à l'extrapolation, sans divergence des sorties NCM ni fuite calibration/test détectée. L'essai, ses coefficients et son protocole sont conservés. La relecture FP64 reproduit les logits dans la tolérance relative documentée ; son écart absolu maximal de 9,31e-8 apparaît sur ces très grandes valeurs et ne constitue pas une parité bit à bit des produits matriciels.

## Autres observations et limites

- **Aucune grille valide ou exactement correcte aux niveaux 1 à 11 sur les 1 000 exemples**, contre 863 au dernier niveau. L'arrêt anticipé direct n'a aucun bénéfice de résolution observé ici. Cela ne préjuge pas d'une exploration supplémentaire guidée par le risque.
- **804 trajectoires** comportent au moins une baisse du nombre de cellules correctes, soit **1 232 transitions régressives**. Le raffinement n'est pas monotone ; ce n'est pas une preuve de métacognition.
- La sortie finale mal formée déjà connue reste présente ; aucun token supprimé ni réponse réparée.
- Une erreur d'indentation du nouveau lanceur a été corrigée avant tout entraînement ; compilation et tests ensuite réussis. Aucun résultat n'a été perdu ou remplacé.
- H1 demeure ouverte ; H2/H3, correction future des cellules, niveaux secondaires, calibration de Platt et confirmation indépendante ne sont pas évalués ici. Une validation croisée imbriquée n'est pas à elle seule une analyse de puissance intégrant le réentraînement.

## Décision

**Pause de rapport avant modification du protocole.** Prochaine priorité : un amendement de développement versionné pour stabiliser les probabilités rares, appliqué identiquement à la référence et à la sonde. Deux options à examiner avant de nouveaux ajustements : conserver les probabilités bornées sans standardisation individuelle, ou imposer une borne inférieure explicite à l'échelle de normalisation. Éviter de choisir un seuil en maximisant l'AUROC des plis externes déjà vus ; si plusieurs variantes sont comparées, leur sélection doit rester interne et la confirmation indépendante doit porter cette limite exploratoire.

Ensuite : rejouer les témoins, examiner discrimination et log-loss ensemble, mesurer la variabilité liée au réentraînement et dimensionner la confirmation. **Ne pas ouvrir le test pour résoudre ce problème de développement.** Aucun changement de couche, nombre de composantes ou famille de sonde n'est justifié par ce premier essai.

Preuves : [extraction](audit/metacognition/train-v0.2/report.json), [grilles intermédiaires](audit/metacognition/train-v0.2/descriptive-evaluation.json), [premiers ajustements](audit/metacognition/development-primary-v0.2/report.json), [témoins](audit/metacognition/development-controls-v0.2/report.json), [relecture et intervalle descriptif](audit/metacognition/development-primary-v0.2/verification.json), [diagnostic de normalisation](audit/metacognition/development-primary-v0.2/scaling-diagnostic.json), [commandes](experiments/metacognition/README.md). Les tableaux et poids de sondes restent locaux, exclus de Git avec manifestes d'empreintes.

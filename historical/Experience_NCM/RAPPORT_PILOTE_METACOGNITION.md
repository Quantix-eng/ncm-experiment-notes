# Pilote métacognition NCM — 23 octobre 2026

Rapport du premier jalon, conservé comme état historique. État actuel : [synthèse consolidée](SYNTHESE_METACOGNITION.md). Suite exécutée : [développement initial](RAPPORT_DEVELOPPEMENT_METACOGNITION.md), puis [correction et test principal](RAPPORT_RESULTATS_METACOGNITION.md).

**Premier jalon validé : partitions matérialisées et 32 trajectoires extraites sans divergence finale. Aucune sonde entraînée ; aucune conclusion métacognitive à ce stade.**

## Exécution et contrôles

Checkpoint Sudoku hard, CPU FP32, batch 8, quatre threads, décodage déterministe. Le modèle et le chemin public restent inchangés. L'extracteur observe le bloc final et les têtes de codes par des hooks temporaires ; les reconstructions des niveaux sont décodées après la génération, sans influencer les niveaux suivants.

Partage figé par solution :

- Apprentissage : **1 000 puzzles**, 863 succès et 137 échecs historiques.
- Calibration : **500 puzzles**, 431 succès et 69 échecs.
- Test : **500 puzzles**, 432 succès et 68 échecs.

Aucun groupe commun ; les versions easy/medium d'une même solution héritent de cette partition. Pas d'audit exhaustif des isomorphismes. Les résultats historiques ont servi à la stratification, comme prévu ; aucune nouvelle inférence n'a été faite sur calibration/test.

Le pilote contient **32 puzzles d'apprentissage, 24 succès et 8 échecs sélectionnés à l'avance**. Il est volontairement enrichi en échecs : ne pas utiliser ses proportions pour estimer la performance naturelle.

**Contrôles réussis :**

- 32/32 : codes, reconstruction, logits et réponses identiques bit à bit aux chemins CPU local avec cache et eager, dans les mêmes lots. Réponses également identiques aux archives exhaustives. Il ne s'agit pas d'une nouvelle exécution du dépôt amont original.
- 12 instantanés cumulatifs indépendants par exemple ; chaque niveau vérifié par accumulation explicite, reconstruction finale identique à l'accumulateur de génération.
- Logits de codes alignés sur les codes effectivement générés ; hooks supprimés, cache nettoyé et empreinte des paramètres/buffers persistants identique avant/après.
- Relecture sans modèle des traces brutes de deux exemples : probabilités et résumés d'activations reproduits exactement.
- Variables aux niveaux 3/6/9/11 inchangées après corruption volontaire de tous les états futurs dans une copie des traces.
- **12 tests automatisés passent**, dont quatre nouveaux sur partitions, fuite temporelle, format des tokens et nettoyage après exception.

## Traces et coût mesuré

Les fichiers comprennent les codes, probabilités, sorties décodées et résumés des 12 niveaux ; deux exemples disposent aussi des activations, reconstructions et logits bruts. Les cibles et corrections des cellules sont dans un fichier d'évaluation séparé. Le constructeur des variables ne reçoit pas les solutions.

Au niveau 9 : 846 variables pour le puzzle seul, 1 341 pour la référence résumée, **13 877 pour la référence enrichie**, et 1 024 résumés internes bruts. La PCA à 16 composantes sera ajustée ultérieurement sur apprentissage uniquement. Cette forte dimension de la référence confirme un risque de surapprentissage à examiner ; aucun surapprentissage n'a encore été mesuré.

- Run complet : **28,98 s**, incluant chargement, chauffe, comparaisons, contrôles et écritures.
- Inférence de référence cumulée : 6,03 s ; extraction instrumentée et replay cumulé : 7,00 s. Une seule série, sans benchmark apparié répété : aucun ratio d'accélération ou de surcoût stable revendiqué.
- Pic RSS du processus : **509 Mo décimaux**, incluant modèle, traces, références et vérifications ; pas une mesure isolée du modèle.
- Traces principales compressées : **6,90 Mo** ; traces détaillées des deux exemples : **3,38 Mo**. Tableaux principaux non compressés : 7,89 Mo, soit une projection linéaire de **493 Mo pour 2 000 puzzles**, hors matrices d'apprentissage et traces détaillées. Cette projection n'est pas un pic mémoire d'une extraction complète.

## Anomalies et observations

1. **Zéro final du puzzle hard 934, position 82**, déjà présent dans l'audit historique, reproduit et compté comme sortie mal formée. La position n'est pas supprimée ni déplacée ; aucune réponse réparée.
2. **Aucune grille entièrement correcte ni valide avec ses indices aux niveaux 1 à 11 dans ce lot**, contre 24 au niveau 12 par construction. Un arrêt anticipé ne présente donc pas encore de bénéfice évident sur ces exemples ; cela n'invalide pas l'anticipation des échecs finaux.
3. La progression n'est pas monotone : **25 exemples ont au moins une baisse du nombre de cellules correctes**, soit 37 transitions régressives entre niveaux. Comportement compatible avec le raffinement déjà décrit par les auteurs, pas preuve de métacognition ni défaut d'instrumentation.
4. La variation relative de reconstruction est indéfinie au premier niveau, sans état précédent non nul : le code utilise explicitement un zéro sentinelle à ce niveau. Les valeurs suivantes sont calculées normalement. Cette variable diagnostique n'entre pas dans la référence observable.
5. La spécification ne fixe pas une procédure exécutable complète pour la correction future des cellules, les perturbations et le corpus indépendant : ces extensions restent à définir avant leurs évaluations. La confirmation n'est pas encore générée et le dimensionnement intégrant l'apprentissage n'est pas réalisé.

## Décision et prochaine étape

**Faisabilité technique démontrée pour le pilote ; pause de rapport à ce jalon.** Les traces de calibration/test restent à extraire. La prochaine étape utile est une extraction limitée aux **1 000 puzzles d'apprentissage**, puis l'étude de la référence et de la sonde en validation croisée, afin d'estimer la dépendance des scores et la variabilité du réentraînement. Cela résout le besoin de données de développement pour le dimensionnement sans consulter les résultats du test. Aucun réglage scientifique n'est choisi à partir du lot enrichi de 32.

Avant confirmation : vérifier le conditionnement de la référence à 13 877 variables, implémenter la PCA/sonde sans fuite, préciser le budget et figer la taille du corpus indépendant. La performance prédictive et les hypothèses H1/H2/H3 restent non testées.

Preuves : [partitions](audit/metacognition/split-v0.2.json), [rapport brut et empreintes](audit/metacognition/pilot-v0.2/report.json), [vérification indépendante des fichiers](audit/metacognition/pilot-v0.2/verification.json), [évaluation descriptive du lot d'apprentissage](audit/metacognition/pilot-v0.2/evaluation-training-only.json), [commandes de reproduction](experiments/metacognition/README.md). Les archives numériques sont locales et exclues de Git ; leurs empreintes sont conservées.

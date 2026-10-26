# Plan d'action — métacognition du NCM

**Mise à jour du 24 octobre 2026 :** pilote de politique d'abandon au niveau 9 exécuté sur développement uniquement, après critique du cadrage confiance. Parité exacte, mais économie optimiste de 2,29–2,60 % sous le repère de 5 % ; arrêt avant nouveau corpus ou dimensionnement confirmatoire. Les chronométrages complets sont instables. Voir [le rapport](RAPPORT_CONTROLE_PRECOCE.md) et [la spécification séparée](experiments/metacognition/early_control/design-v1.json). Le transfert de calibration reste différé.

**État après exécution autorisée :** l'étude de compensation latente native est terminée et relue ; voir [le rapport](RAPPORT_DYNAMIQUE_NATIVE.md) et [la synthèse](SYNTHESE_METACOGNITION.md). Pas de gain prédictif dynamique établi ; compensation directionnelle sans bénéfice de résolution établi sous ce stress, puis arrêt du pilote de renforcement faute de dommage suffisant. Les spécifications v0.2 et les étapes ci-dessous restent le cadrage historique ; les nouveaux protocoles sont séparés et n'en modifient pas les gels.

Version **0.2 du 23 octobre 2026, complétée par l'amendement d'exécution v0.3**. **État : prétraitement stabilisé, 20 réentraînements audités, modèles gelés, calibration et test principal exécutés. Gain interne au test : +0,00426 d'AUROC [0,00201 ; 0,00681], inférieur au repère 0,03 ; les résumés observables seuls font mieux.** Voir [les résultats](RAPPORT_RESULTATS_METACOGNITION.md). Les essais antérieurs sont conservés dans [le développement v0.2](RAPPORT_DEVELOPPEMENT_METACOGNITION.md) et [le pilote](RAPPORT_PILOTE_METACOGNITION.md). Le [JSON v0.2](experiments/metacognition/protocol-v0.2.json) reste inchangé ; [l'amendement v0.3](experiments/metacognition/amendment-v0.3.json) documente la correction fixée avant ses ajustements. Ce n'est pas un pré-enregistrement public. Le test demeure exploratoire ; le dimensionnement confirmatoire intégrant l'apprentissage n'est pas achevé.

**Décision historique du 23 octobre : consolider les deux premières évaluations avant de poursuivre.** Cette consolidation a précédé l'autorisation explicite de l'étude native, maintenant terminée. La [synthèse](SYNTHESE_METACOGNITION.md) distingue ces étapes ; les extensions non exécutées restent conditionnelles.

## Objectif et niveau de preuve

**Question principale : à niveau donné et à famille de sondes fixée, les états internes améliorent-ils la prédiction de la réussite finale par rapport aux variables observables précisément définies ?**

Le calcul est déterministe : les états sont eux-mêmes une fonction du puzzle. « Au-delà du puzzle » signifie donc un gain d'accessibilité prédictive avec notre observateur limité, pas une information au sens théorique qui serait absente du puzzle, ni une supériorité sur tout prédicteur possible.

Question secondaire : peut-on anticiper quelles erreurs seront corrigées par les niveaux restants ? Si un signal est confirmé, examiner son transfert après perturbation puis son utilité pour allouer le calcul.

Trois conclusions à distinguer :

1. Une sonde prédit l'erreur : **information interne décodable**.
2. Une politique ajoutée utilise ce signal avec bénéfice : **contrôle du système augmenté**.
3. Un mécanisme du NCM lui-même évalue sa fiabilité et influence son comportement : **preuve plus forte d'une fonction métacognitive native**, nécessitant des expériences causales supplémentaires.

Un résultat au premier niveau ne suffit pas pour conclure au troisième. La petite taille n'interdit pas le pilote ; elle limite ce que nous pouvons supposer et généraliser.

## Base disponible et points déjà résolus

- Checkpoint **Sudoku hard** : générateur 25,46 M + VQ-VAE 14,90 M, soit 40,35 M de paramètres ; 12 niveaux. Un seul modèle au départ évite de confondre difficulté et checkpoint.
- Référence CPU FP32, batch 8, quatre threads, argmax déterministe. Parité exhaustive antérieure documentée sur les trois difficultés ; celle de l'instrumentation est validée sur les 1 000 puzzles d'apprentissage (chemin cache ; eager également vérifié sur 112).
- 2 000 puzzles hard, 1 726 succès et 274 échecs archivés. Les trois difficultés partagent les mêmes 2 000 solutions : aucune séparation indépendante par difficulté.
- Aliasing des instantanés corrigé. Deux zéros finaux détectés par le contrôle strict de format ; les scores historiques les comptaient déjà comme échecs.
- Bibliographie et catalogue disponibles dans [la revue d'antériorité](REVUE_ANTERIORITE_METACOGNITION.md) et [l'inventaire](MODELES_NCM.md). Aucun précédent de sonde entraînée pour notre cible sur ce NCM identifié ; aucune priorité scientifique garantie.

## Phase 1 — Figer le protocole et protéger les données

**Actions réalisées.** Configuration et empreintes du code/des actifs enregistrées ; les 2 000 groupes de solutions sont répartis en **50 % apprentissage / 25 % calibration / 25 % test**, avec graine fixée, stratification des succès historiques et regroupement de toutes les versions d'une solution. Le [manifeste figé](audit/metacognition/split-v0.2.json) comprend les indices du pilote et les plis d'apprentissage. Le générateur et le solveur indépendant sont testés sur 32 puzzles nouveaux non évalués par le NCM ; le corpus statistique de confirmation reste à dimensionner et constituer.

La calibration sert à ajuster les probabilités et les seuils ; les hyperparamètres sont sélectionnés par validation croisée groupée dans l'apprentissage. Le test est consulté une fois, après gel des choix. Les transformations d'un puzzle restent dans son groupe. Contrôler doublons exacts et liens de transformation connus ; ne pas revendiquer une déduplication exhaustive de toutes les symétries sans l'avoir réalisée.

**Cibles fixées.** Principal : échec exact final (étiquette 1), prédit au **niveau 9 sur 12** ; prédire la réussite est équivalent en inversant les probabilités. Le choix du niveau 9 est une convention de design conservée avant toute trajectoire : 37 positions de codes au niveau courant et trois niveaux de raffinement encore disponibles. Il représente 116 codes cumulés sur 322 ; il n'a donc pas consommé nécessairement 75 % du calcul. Cette justification structurelle est ajoutée après notre choix initial, avant les résultats, sans prétendre à un niveau optimal ou à une performance intermédiaire connue. Analyses de sensibilité fixées aux niveaux 3, 6 et 11 : elles ne peuvent remplacer un résultat principal défavorable. Les niveaux sont numérotés de 1 à 12 et ne désignent pas les huit couches Transformer.

La correction future des cellules est secondaire. **Priorité suivante choisie par l'utilisateur après H1 : étude dédiée [correction-v0.1](experiments/metacognition/correction/protocol-v0.1.json), résultats dans [le rapport de correction](RAPPORT_CORRECTION_FUTURE.md).** Étude terminée : référence 0,67325, avec PCA16 locale 0,67265, différence −0,00060 [−0,00428 ; +0,00320] sur 145 puzzles mixtes d'un corpus neuf de 1 000. Elle utilise des états locaux du décodeur (pas les résumés globaux du générateur), une AUROC au sein des puzzles et un corpus externe neuf ; le test H1 déjà consulté n'est pas réutilisé. Cette étude ne modifie pas rétrospectivement le critère principal H1. L'identification des cellules erronées utilise la référence uniquement pour construire les cibles et analyser les résultats ; elle n'est pas une entrée de la sonde.

Les scores historiques ont déjà été vus : ce sera une étude exploratoire avec test tenu à l'écart de l'ajustement, pas un corpus prospectif vierge. **Concevoir dès cette phase le corpus de confirmation** : nouvelles solutions par backtracking randomisé, retrait d'indices jusqu'à 30 en conservant une solution unique vérifiée par un solveur indépendant, journal des graines/rejets, exclusion des solutions déjà utilisées et des transformations connues. Geler et hacher puzzles et règle de sélection avant de mesurer leurs sorties NCM ; ne pas sélectionner selon leur réussite. Générateur et solveur indépendant sont désormais implémentés et vérifiés sur un pilote de 32 ; exclusion des relabellisations de chiffres et huit symétries du carré, sans prétendre à une déduplication exhaustive par isomorphisme. Ce pilote ne constitue pas la confirmation statistique. Cette distribution générée différera possiblement de l'officielle : validation externe, pas réplication IID garantie. L'ensemble d'apprentissage des auteurs n'ayant pas été audité, ne pas affirmer l'absence de contamination du préentraînement.

**Dimensionnement avant l'expérience complète.** La simulation binormale appariée [simulate_auc_design.py](experiments/metacognition/simulate_auc_design.py) explore 72 scénarios, 1 000 répétitions chacun : n=500/1 000/2 000/5 000, AUROC de référence 0,70/0,85, corrélations conditionnelles aux classes 0,30/0,70/0,95 et gains 0/0,03/0,06. [Résultats synthétiques](audit/metacognition/design-power-sensitivity.json). Avec n=500 et gain réel 0,03, la détection d'un gain positif varie de 13,2 % à 93,9 %. À référence 0,85, elle vaut respectivement 19,6 %, 30,4 % et 93,9 % pour les trois corrélations. Les scores appariés comptent donc autant que le seul nombre d'échecs.

Ce calcul emploie la variance de placement de DeLong et des intervalles normaux, conditionnellement à deux prédicteurs fixés ; il ne simule ni leur apprentissage ni le bootstrap du protocole final. Compléter sur l'apprentissage seulement par une analyse de précision/puissance tenant compte du réentraînement, sans consulter le test. Dimensionner la confirmation pour une puissance cible de 80 % sous des hypothèses explicites et un budget accepté ; aucune taille définitive n'est justifiée aujourd'hui. Si ce budget est insuffisant, conserver le statut exploratoire. Ni le taux d'échec du futur générateur ni les corrélations réelles ne sont connus. [Référence méthodologique sur le dimensionnement apparié](https://pubmed.ncbi.nlm.nih.gov/35429054/).

**Livrables :** configuration, manifeste de partage, définition des cibles et métriques, hypothèses H1/H2/H3. **Condition de passage :** aucun groupe partagé entre partitions ; règles figées avant extraction analytique.

## Phase 2 — Instrumenter et valider un pilote technique

**Actions.** Sur **32 puzzles pris seulement dans l'apprentissage** — 24 succès et 8 échecs historiques pour solliciter les deux cas — enregistrer par niveau :

- codes, grille décodée à partir de la reconstruction cumulative à pleine longueur et probabilités du décodeur ;
- entropies et marges des distributions de codes, agrégées par moyenne et quantiles sans favoriser les niveaux plus longs ;
- sortie de `net.transformer.blocks[7]`, avant `transformer.ln_f`, uniquement lors de l'appel de génération courant : moyenne et écart-type population sur toutes ses positions cibles, hors appel de préremplissage et hors préfixe ; 512 canaux donnent 1 024 valeurs ;
- changements de cellules, format, conflits Sudoku, indices violés et variation normalisée de la reconstruction ;
- temps d'inférence, coût du décodage de chaque niveau, mémoire et volume des traces.

**Contrôles.** L'instrumentation doit restituer exactement les codes, la reconstruction et les logits finaux de la référence au même batch. Vérifier indépendance des instantanés, nettoyage des hooks/caches et absence de modification des poids. Utiliser les positions de grille attendues pour les cellules ; ne pas déplacer les cellules en filtrant les tokens mal formés. Conserver les échecs de format comme des échecs, sans réparation.

Le pilote technique ne sert pas à annoncer des performances statistiques. Aucun tri des variables selon leurs corrélations sur le test. Les activations sont copiées en FP32 sans modifier l'inférence, puis agrégées en FP64. La sonde emploiera 16 composantes PCA des 1 024 valeurs, ajustées uniquement sur l'apprentissage et réajustées à l'intérieur de chaque pli. Une absence d'effet restera limitée à cette représentation, qui peut perdre un signal de faible variance. Mesurer le stockage avant passage à 2 000 puzzles ; conserver principalement des résumés et les sorties utiles, avec traces complètes seulement pour un petit sous-ensemble identifié.

**Livrables :** extracteur isolé, traces du pilote et rapport de parité/coût. **Condition de passage :** zéro divergence finale ; coût et stockage compatibles avec le Mac. Sinon, corriger l'instrumentation avant toute interprétation scientifique.

## Phase 3 — Tester l'information prédictive sur le modèle original

**Avancement du 23 octobre.** Essai v0.2 conservé après amplification de probabilités rares jusqu'à 8,53e9. Amendement v0.3 fixé avant de nouveaux ajustements : échelle observable ≥0,1, mêmes modèles et réglages. CV imbriquée sur 1 000 puzzles, relecture indépendante puis 20 bootstraps de réentraînement. Gel final et calibration/test 500/500 exécutés : référence 0,88351, ajout interne 0,88777, résumés seuls 0,91023. Intervalle principal [0,00201 ; 0,00681], conditionnel aux prédicteurs ajustés. Petit gain discriminatif dans cette comparaison, sans bénéfice pratique établi ni métacognition native démontrée. Les sensibilités 3/6/11 sont terminées sous les mêmes règles : gains +0,01191/+0,00453/+0,00361, secondaires sans correction de multiplicité, ne remplaçant pas le principal. Le test principal a été maintenu exploratoire à effectif fixé ; la sensibilité de réentraînement ne constitue pas une puissance complète de confirmation.

**Actions.** Après validation technique, extraire d'abord les seuls 1 000 puzzles d'apprentissage pour implémenter les sondes et compléter le dimensionnement avec variabilité du réentraînement. L'extraction de calibration/test suit le gel des choix et la décision de poursuite ; aucune performance sur ces partitions ne guide le dimensionnement. Comparer au niveau principal les modèles suivants :

1. Taux de réussite constant ; puis caractéristiques du puzzle seul, sans solveur externe.
2. Probabilités et marges seules.
3. Contraintes visibles de la grille intermédiaire ; puis ensemble des caractéristiques précédentes.
4. Ajout de l'historique visible jusqu'au niveau courant.
5. Référence enrichie : ajout des probabilités complètes du décodeur courant (89×12), des probabilités de codes de tous les niveaux disponibles et des tokens décodés de leur historique en one-hot, en plus du puzzle encodé en one-hot et des résumés définis dans la configuration.
6. Ajout de 16 composantes des résumés internes à cette référence enrichie : **sonde principale, régression logistique régularisée**. Un contrôle ajoute plutôt 16 variables normales indépendantes pour examiner l'effet de dimension/capacité.

La configuration précise chaque résumé, position, quantile et normalisation. Normalisation et PCA sont apprises dans chaque pli d'apprentissage, jamais sur calibration/test. La référence enrichie peut comporter beaucoup de variables : examiner son surapprentissage sur développement et l'expliciter, sans la qualifier de maximale ou universelle. Un gain sur des résumés seuls ne justifiera pas un gain sur toute l'information observable.

**Apprentissage fixé.** Régression logistique binaire sans pondération de classe, objectif `moyenne(log-loss) + λ||w||²/2`, intercept non pénalisé ; λ∈{1e-4, 1e-3, 1e-2, 1e-1, 1, 10}. Validation croisée à cinq plis stratifiés et groupés dans l'apprentissage, sélection par log-loss moyenne ; égalité à 1e-8 près départagée en faveur du λ supérieur. Chaque modèle reçoit la même grille et les mêmes plis. Optimiseur prévu : LBFGS PyTorch CPU FP64, maximum 500 itérations, tolérances et graines dans le JSON. Aucun package d'apprentissage supplémentaire installé. Une non-convergence est signalée avant analyse, jamais corrigée par réglage sur le test. Calibration de Platt sur les seuls exemples de calibration ; AUROC calculée sur le logit brut. Ne pas passer directement à un réseau profond.

**Contrôles de validité.** Permuter les états entre puzzles du même niveau pour vérifier que leur association avec l'exemple est utile ; comparer les caractéristiques du puzzle et des sorties à la sonde ; vérifier que les variables ne contiennent ni solution, ni état futur, ni label historique. Des cellules et niveaux d'un même puzzle ne doivent jamais traverser des partitions différentes.

**Évaluation.** Principal : différence appariée d'AUROC entre la sonde avec états internes et la référence enrichie sans ces états au niveau 9. Secondaires : aire précision-rappel pour l'échec, Brier et calibration, risque selon couverture. Référence de prévalence hard : 13,7 % d'échecs sur le corpus entier, à recalculer pour chaque partition. Intervalles percentile à 95 % par 5 000 bootstraps appariés des groupes de solutions, stratifiés sur succès/échec ; les cellules et niveaux restent regroupés. Ces intervalles sont conditionnels aux prédicteurs ajustés : rapporter séparément la variabilité liée à leur apprentissage. Les résultats secondaires sont exploratoires et ne remplacent pas le critère principal après observation.

**Décision : séparer deux questions.** Un intervalle entièrement positif soutient un apport discriminatif ; **0,03 reste un repère pratique** pour l'importance du gain, pas une norme scientifique. L'ancien filtre « estimation≥0,03 et intervalle>0 » ne peut pas avoir une probabilité de réussite de 80 % au gain réel exactement égal à 0,03 : la première condition seule passe environ une fois sur deux dans une estimation centrée. La simulation retrouve environ 49–50 % pour le filtre complet dans les scénarios très corrélés à n=500, malgré une meilleure détection d'un simple gain positif.

Avec 500 puzzles test, environ 68–69 échecs sont attendus. Rapporter estimation et intervalle : borne supérieure<0,03 limite le gain utile pour cette sonde ; intervalle recouvrant 0,03 laisse son importance pratique indéterminée. Une preuve de gain minimal de 0,03 demanderait une borne inférieure>0,03 et un dimensionnement sous un effet réel supérieur à ce seuil. La confirmation est dimensionnée et gelée avant ses résultats, sans extension opportuniste au vu de son effet.

**Livrable :** résultats reproductibles, positifs, négatifs ou indéterminés, incluant les erreurs stables à forte confiance. **Condition de passage :** apport interne reproductible ou hypothèse nouvelle suffisamment précise pour justifier une confirmation.

## Phase 4 — Perturbations et spécificité au modèle

**Actions prioritaires, sans fine-tuning :**

- Modifier un petit nombre de codes à des niveaux fixés, avec taux et graines prédéfinis ; comparer les mêmes puzzles avec/sans perturbation et des témoins adaptés.
- Tester une concentration différente des probabilités par échelle positive des logits, avec argmax conservé : témoin de calibration, pas modèle doté d'une personnalité « arrogante ».
- Tester un petit ensemble fixé de symétries/permutations valides, en ramenant les réponses au repère initial : mesure de robustesse candidate.

**Comparateurs appariés à définir sur développement avant l'évaluation finale.** Pour les remplacements de codes, comparer un candidat plausible (parmi les alternatives du générateur) à un remplacement du même codebook ; contrôler niveau, nombre et positions des codes, entropie avant intervention et norme de la variation de reconstruction induite. Ajouter un passage fictif sans modification pour vérifier le mécanisme d'injection et une perturbation de direction aléatoire appariée en norme pour tester la sensibilité à une anomalie générique. Cette dernière n'est pas supposée sans effet. Mesurer l'écart à la distribution des états propres. Le choix des paires ne doit pas utiliser la réussite future ; éviter un appariement systématique sur l'entropie après intervention, qui peut être une conséquence du mécanisme étudié. Une égalité de norme ou d'entropie ne garantit pas l'égalité des effets fonctionnels.

Évaluer d'abord **la sonde gelée**, puis, séparément, une sonde réentraînée sur les variantes. Cela distingue transfert d'un signal et facilité d'apprendre un détecteur spécifique. Vérifier si la sonde suit les changements de fiabilité d'un même puzzle, au-delà de la difficulté globale. Une perturbation peut être bénéfique ; rapporter corrections comme dégradations. La détection d'un état hors distribution ne suffit pas à démontrer une surveillance d'erreur.

Développer ces choix sur les partitions de développement ; réserver une évaluation dédiée indépendante plutôt que réutiliser le test principal pour sélectionner les variantes.

**Livrable :** test H2 de transfert/spécificité et limites des variables. Si une expérience modifie la génération, l'étiqueter comme variante expérimentale, distincte de la référence de parité.

## Phase 5 — Tester un contrôle utile

**Actions.** Choisir une seule politique sur les données de développement : arrêt à un niveau intermédiaire si la qualité le permet, ou exploration supplémentaire réservée aux trajectoires jugées à risque. Le choix dépendra du taux réel de réussite intermédiaire et du coût mesuré ; aucune économie n'est présumée.

**Utilité principale : coût pour un taux de résolution vérifiée et une couverture comparables**, avec marges de comparaison fixées sur développement. Comparer à une politique fixe, une allocation aléatoire et une politique fondée sur les contraintes visibles, coût de la sonde, des décodages, des vérifications et de toute recherche de candidats inclus. Rapporter séparément les grilles correctes retournées et les abstentions, pour éviter une économie artificielle obtenue en répondant moins.

Inclure le vérificateur final format/contraintes/indices : il détectait déjà tous les échecs finaux archivés. Une abstention après résultat final doit donc être comparée à cette référence forte. **Ce vérificateur ne corrige pas les réponses** : une politique d'exploration peut augmenter le nombre de grilles résolues si elle trouve d'autres candidats valides. Ce gain reste à démontrer face à une exploration simple avec le même vérificateur et au même budget. À candidat final inchangé, améliorer la sélection/abstention ne crée aucun succès supplémentaire.

**Livrable :** test H3, compromis coût/réussite/couverture. Un bénéfice démontrerait un intérêt du système augmenté ; un contrôle natif demanderait de localiser et intervenir sur un mécanisme interne d'évaluation, avec témoins de capacité et de contenu. Cette dernière expérience est une extension, pas un résultat supposé.

## Phase 6 — Fine-tuning et changement d'échelle, sous conditions

**Variantes apprises.** Si la mesure précédente est informative : comparer original, adaptation sur données propres et adaptation sur données bruitées, avec budget et données de départ contrôlés, plusieurs graines et VQ-VAE figé. Mesurer simultanément compétence, calibration et prédiction d'erreur. LoRA est une option technique à prototyper ; ni pipeline prêt ni faisabilité mémoire ne sont établis. Ne pas entraîner sur les puzzles de test.

Une variante explicitement entraînée à évaluer sa fiabilité sera étudiée séparément des variantes de compétence. Le terme « sous-entraîné » est réservé à de vrais checkpoints précoces d'une même trajectoire, indisponibles dans notre inventaire actuel. Sans ces checkpoints, renoncer à cette comparaison ou organiser un entraînement contrôlé supplémentaire.

**Échelle et autres modèles.** Après confirmation, étudier une référence sur la même tâche pour savoir ce qui est spécifique au NCM. Pour une extension textuelle, LM1B small/medium/large partagent leur VQ-VAE ; vérifier aussi les conditions d'entraînement et définir une cible textuelle fiable avant de télécharger les poids. XSum est une autre possibilité pour la fidélité au document, nécessitant une mesure adaptée. Une différence Sudoku/LM1B ne peut pas être attribuée à la seule taille.

**Condition de passage :** intérêt scientifique démontré par les premières phases et coût matériel mesuré. Ne pas lancer simultanément variantes, nouvelles architectures et nouvelles tâches.

## Organisation et premier jalon

Conserver les expériences dans `experiments/metacognition/`, les configurations sous ce dossier, les données dérivées dans `audit/metacognition/` et les résultats synthétiques dans un rapport dédié. Ces dossiers contiennent désormais la spécification v0.2, le dimensionnement synthétique, les partitions, le pilote de 32, les 1 000 trajectoires d'apprentissage et les ajustements externes des six comparateurs avec leurs contrôles et diagnostic de normalisation, l'amendement v0.3, les 20 réentraînements, le gel et l'évaluation principale, ainsi que le pilote technique du générateur indépendant. Empreintes, graines, environnement et sorties des contrôles accompagnent chaque exécution. Exclure les gros tenseurs de Git tout en conservant leur manifeste. Éviter de dupliquer les rapports à chaque run : un journal, des artefacts identifiés et un rapport de synthèse.

Les modifications du chemin public, des poids de référence et de `optimisation-ncm/` ne font pas partie de ce plan. Le travail de portage multi-modèles déjà signalé dans la feuille de route reste distinct.

**Jalons atteints : pilote, développement, correction versionnée, sensibilité au réentraînement et test principal. Suite choisie exécutée : correction future par états locaux du décodeur, validation externe sans gain établi. Suite native autorisée et terminée : lecture dynamique sans gain établi, stress causal peu dommageable, suivi de faisabilité arrêté selon sa règle. Décision actuelle : rapport clos, nouvelles méthodes à évaluer sur un autre corpus après cadrage et marge de ressources.** Le coût du pilote conditionne l'étude complète. L'objectif est une expérience interprétable et falsifiable, pas la garantie d'un résultat positif ou d'une métacognition native.

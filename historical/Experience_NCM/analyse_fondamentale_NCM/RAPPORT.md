# Analyse fondamentale des checkpoints NCM Sudoku

Rapport expérimental — 24 octobre 2026. Analyse locale CPU FP32. Périmètre : trois checkpoints publiés Sudoku, avec analyse approfondie de hard. Les modèles de texte ne sont pas évalués.

## Résultat principal

Au niveau 6, sur 600 puzzles hard réservés au test, la baseline obtient une AUROC de **0.890**, contre **0.908** après ajout des états internes. Le gain est **0.0183**, IC 95 % apparié **[-0.008, 0.042]**. L'intervalle recouvre zéro : cette expérience n'établit pas une valeur ajoutée des états au-delà de la baseline.

Ce contraste évalue ces sondes linéaires et leur procédure de régularisation ; une absence de gain n'établit pas l'absence d'information dans les états. Il s'agit de la prédiction externe de la réussite d'un modèle gelé. Aucun de ces résultats ne démontre que NCM évalue lui-même sa fiabilité ou utilise une évaluation pour contrôler sa génération.

## 1. Questions, méthode et indépendance

Le protocole révisé couvre les 13 items de la TODO : audit du tokenizer ; stabilité ; symétries ; généralisation et difficulté ; discrimination/calibration ; sondes de contraintes, ablations et progression de la représentation. Le protocole original gelé est conservé dans [sources/TODO_protocol_v1.md](sources/TODO_protocol_v1.md), avec ses paramètres dans [protocol.json](protocol.json), le gel horodaté dans [results/protocol_frozen.json](results/protocol_frozen.json) et les amendements dans [JOURNAL.md](JOURNAL.md).

L'audit des state_dict confirme que les trois tokenizers et les trois générateurs ont des poids distincts ; leurs analyses ne sont donc pas mutualisées sur la seule base de leurs architectures identiques. Les entrées, poids, sources et environnements sont copiés ou installés dans ce dossier. Aucun résultat antérieur du projet n'est utilisé comme résultat de cette campagne. Le runner CPU provient de la copie locale ncm_mac ; le dépôt amont intact est aussi conservé et exécuté pour le contrôle de parité. Les dépendances système préexistantes sont accessibles en lecture à l'environnement local ; les installations et caches de cette campagne sont locaux.

Les corpus des trois difficultés partagent les mêmes 2 000 familles de solutions (confirmé par audit croisé), avec des puzzles différents. Les réplications entre difficultés sont donc corrélées ; elles ne sont pas des réplications indépendantes de familles de solutions. Les 2 000 puzzles de chaque difficulté ont été validés quant au format, aux indices et à la validité des références. Chaque corpus possède 2 000 puzzles et 2 000 solutions distincts, même après normalisation des chiffres et les huit symétries du carré testées. Cela n'exclut ni toutes les isomorphies Sudoku ni une présence dans l'entraînement.

**Partition hard** : 1 000 train, 400 validation, 600 test ; graine 20261024. Les groupes de puzzles équivalents sous les transformations testées restent ensemble. La standardisation et les poids sont ajustés sur train seulement ; C est choisi dans [0.001, 0.01, 0.1, 1.0] par log-loss de validation, sans réentraînement sur le test. Les variantes géométriques utilisent 128 puzzles de test, sans entraîner de nouvelle sonde. Les contrôles de stabilité et le croisement modèle/corpus sont descriptifs et n'entraînent aucune sonde.

**États** : sortie de la couche Transformer 8 avant la normalisation finale ; moyenne et écart-type population sur les positions du niveau courant, soit 1 024 valeurs. Au niveau 1, l'écart-type est nul car il n'y a qu'une position. Les états sont extraits pendant une vraie génération à partir du puzzle seul. Les niveaux de génération, de longueurs 1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67 et 89, sont distincts des huit couches Transformer.

**Baseline** : 6 proxies de difficulté issus du puzzle, 8 statistiques des distributions de codes et du décodeur, 5 mesures de validité/indices/conflits et 1 taux de changement depuis le décodage précédent. Ces 20 valeurs sont calculées sans solution de référence. Les distributions du décodeur au niveau courant nécessitent un décodage intermédiaire supplémentaire ; il s'agit d'une baseline d'observation post-hoc, pas d'une évaluation du coût d'un système déployé. Les sondes logistiques prédisent la réussite exacte finale. Les analyses par niveau sont exploratoires, à l'exception du contraste annoncé au niveau 6.

**Incertitude** : Wilson pour les taux de réussite ; bootstrap percentile apparié, 2000 tirages par puzzle, pour le contraste principal et les interventions. Les sondes ne sont pas réentraînées dans le bootstrap : les IC sont conditionnels à ce découpage et à ces ajustements. Pas de correction pour comparaisons multiples exploratoires ; aucune découverte de circuit n'est affirmée sur ces seules comparaisons.

## 2. Reproduction et fidélité du runner

- **easy** : 1988/2000 grilles exactes, soit 99.40 %, IC Wilson [98.95 %, 99.66 %] ; cases correctes 99.97 % ; validité avec indices 99.40 %. Parité avec le code amont intact sur 16 puzzles : codes=True, réponses=True, écart maximal de logits=0.

- **medium** : 1928/2000 grilles exactes, soit 96.40 %, IC Wilson [95.49 %, 97.13 %] ; cases correctes 99.79 % ; validité avec indices 96.40 %. Parité avec le code amont intact sur 16 puzzles : codes=True, réponses=True, écart maximal de logits=0.

- **hard** : 1726/2000 grilles exactes, soit 86.30 %, IC Wilson [84.72 %, 87.74 %] ; cases correctes 98.35 % ; validité avec indices 86.30 %. Parité avec le code amont intact sur 16 puzzles : codes=True, réponses=True, écart maximal de logits=0.

L'article rapporte 99,5 %, 96,5 % et 86,1 % de grilles résolues pour easy, medium et hard. Les résultats ci-dessus concernent les fichiers publiés effectivement disponibles et une comparaison de séquences complète, séparateurs compris. Un petit écart avec le tableau de l'article n'est pas à lui seul une preuve d'erreur : version des poids, protocole d'évaluation et arrondis sont des explications à vérifier. Les contrôles de parité ne portent que sur 16 puzzles par checkpoint ; ils ne prouvent pas l'identité numérique pour toutes les entrées. [Article, §4.4](https://arxiv.org/html/2610.14162v1#S4.SS4)

L'instrumentation vérifie à chaque batch l'alignement argmax/logits/codes, le nettoyage du cache et des hooks, l'absence d'alias entre accumulateurs intermédiaires, leur égalité à une accumulation explicite et l'égalité du dernier décodage avec la sortie normale. L'incident de sérialisation ListConfig et sa récupération depuis les NPZ intacts sont documentés dans JOURNAL.md ; les contrôles de parité ont été rejoués. Le code amont brut présente un aliasing dans idx_to_zhats ; la copie CPU utilise clone() pour conserver un véritable historique. Ce correctif concerne l'observation des historiques ; le calcul final est vérifié séparément contre l'amont.

![Progression des décodages](figures/02_generation.png)

## 3. Unicité et définition de la réussite

- **hard** : 0 puzzle sans solution, 2000 à solution unique, 0 avec au moins deux solutions selon Algorithm X (comptage plafonné à 2).

- **easy** : 0 puzzle sans solution, 2000 à solution unique, 0 avec au moins deux solutions selon Algorithm X (comptage plafonné à 2).

- **medium** : 0 puzzle sans solution, 2000 à solution unique, 0 avec au moins deux solutions selon Algorithm X (comptage plafonné à 2).

La métrique principale reste la fidélité stricte à la référence, et la validité Sudoku avec respect des indices est mesurée séparément. Une solution valide alternative doit être interprétée comme telle ; elle n'est pas assimilable à une violation des contraintes. L'unicité des corpus officiels n'était pas supposée.

## 4. Audit du tokenizer : utilisation, complémentarité et redondance — items 1–3

Cet audit encode les solutions connues. Il mesure la capacité de représentation et de reconstruction ; il ne mesure pas une résolution à partir des indices.

- **easy** : utilisation moyenne encodée 80.73 %, générée 80.73 % ; entrées non observées dans les codebooks encodés 37/192. Reconstruction finale : 100.00 % des tokens, 100.00 % des grilles. Perplexité marginale par niveau : 11.23, 2.25, 11.83, 13.62, 1.74, 14.07, 14.83, 14.58, 13.88, 13.07, 14.84, 12.52.

- **medium** : utilisation moyenne encodée 83.85 %, générée 83.85 % ; entrées non observées dans les codebooks encodés 31/192. Reconstruction finale : 100.00 % des tokens, 99.95 % des grilles. Perplexité marginale par niveau : 11.11, 1.18, 14.17, 14.77, 1.01, 14.05, 14.59, 14.54, 13.93, 13.54, 14.47, 12.61.

- **hard** : utilisation moyenne encodée 80.73 %, générée 80.21 % ; entrées non observées dans les codebooks encodés 37/192. Reconstruction finale : 100.00 % des tokens, 100.00 % des grilles. Perplexité marginale par niveau : 10.65, 1.35, 12.89, 13.95, 1.00, 14.00, 14.52, 14.55, 13.86, 13.13, 13.70, 12.55.

Audit exact des erreurs de reconstruction **easy** : 0 token(s) erroné(s) sur 178000, touchant 0 grille(s) sur 2000. Les cas non nuls ont été rejoués dans le tokenizer amont intact avec codes et sorties identiques ; détails dans results/tokenizer_errors.json. L'arrondi du taux token à deux décimales peut masquer une erreur isolée ; il ne faut pas le lire comme une garantie d'exactitude de toutes les grilles.

Audit exact des erreurs de reconstruction **medium** : 1 token(s) erroné(s) sur 178000, touchant 1 grille(s) sur 2000. Les cas non nuls ont été rejoués dans le tokenizer amont intact avec codes et sorties identiques ; détails dans results/tokenizer_errors.json. L'arrondi du taux token à deux décimales peut masquer une erreur isolée ; il ne faut pas le lire comme une garantie d'exactitude de toutes les grilles.

Audit exact des erreurs de reconstruction **hard** : 0 token(s) erroné(s) sur 178000, touchant 0 grille(s) sur 2000. Les cas non nuls ont été rejoués dans le tokenizer amont intact avec codes et sorties identiques ; détails dans results/tokenizer_errors.json. L'arrondi du taux token à deux décimales peut masquer une erreur isolée ; il ne faut pas le lire comme une garantie d'exactitude de toutes les grilles.

**Niveaux peu diversifiés (easy)** : niveau 5, 2/16 codes observés, H=0.5552 nat.

**Niveaux peu diversifiés (hard)** : niveau 2, 2/16 codes observés, H=0.3037 nat, niveau 5, 1/16 codes observés, H=0.0000 nat.

Un niveau entièrement constant n'encode aucune variation entre les solutions observées ; son résidu appris peut cependant agir comme un décalage commun utile au décodeur. Ce constat ne démontre ni inutilité causale du niveau ni effondrement global du tokenizer. Il motive un test dédié de suppression ou de remplacement de ce niveau, distinct de l'ablation de confiance réalisée ici.

Les histogrammes, entropies en nats, perplexités, indices non observés et couvertures à 16/64/256/1 000/2 000 solutions sont disponibles dans results/tokenizer.json. L'entropie conditionnelle à la position distingue la variabilité entre puzzles d'une simple hétérogénéité entre positions. Les estimations restent empiriques ; l'absence d'un code dans ce corpus n'établit pas qu'il serait impossible à activer ailleurs.

La reconstruction cumulative quantifie ce qu'apportent les niveaux au décodage observé. Une entropie élevée n'est pas une preuve d'utilité, une faible entropie n'est pas une preuve d'inutilité, et une reconstruction partielle qui se dégrade après ajout d'un niveau ne suffit pas à conclure à une hiérarchie défectueuse.

![Audit tokenizer](figures/01_tokenizer.png)

Les distances d'embeddings ont été complétées par un test fonctionnel : pour 32 solutions hard de train, remplacer le premier emplacement de chaque niveau par chacun des 16 codes, puis comparer les 120 paires dans les mêmes contextes, les autres codes restant fixés. Le test compare les cases argmax et la variation totale des distributions finales. Il ne teste qu'un emplacement par niveau et peut créer des combinaisons hors distribution.

- Niveau 1 : désaccord de cases min/médian/max = 0.00 % / 0.39 % / 0.58 % ; 7/120 paires sans désaccord argmax dans ces contextes ; corrélation de Spearman distance/effet = 0.759.

- Niveau 2 : désaccord de cases min/médian/max = 0.00 % / 0.00 % / 0.04 % ; 105/120 paires sans désaccord argmax dans ces contextes ; corrélation de Spearman distance/effet = 0.573.

- Niveau 3 : désaccord de cases min/médian/max = 0.00 % / 0.00 % / 0.00 % ; 120/120 paires sans désaccord argmax dans ces contextes ; corrélation de Spearman distance/effet = indéfinie (effet constant).

- Niveau 4 : désaccord de cases min/médian/max = 0.00 % / 0.00 % / 0.00 % ; 120/120 paires sans désaccord argmax dans ces contextes ; corrélation de Spearman distance/effet = indéfinie (effet constant).

- Niveau 5 : désaccord de cases min/médian/max = 0.00 % / 0.00 % / 0.00 % ; 120/120 paires sans désaccord argmax dans ces contextes ; corrélation de Spearman distance/effet = indéfinie (effet constant).

- Niveau 6 : désaccord de cases min/médian/max = 0.00 % / 0.00 % / 0.00 % ; 120/120 paires sans désaccord argmax dans ces contextes ; corrélation de Spearman distance/effet = indéfinie (effet constant).

- Niveau 7 : désaccord de cases min/médian/max = 0.00 % / 0.00 % / 0.00 % ; 120/120 paires sans désaccord argmax dans ces contextes ; corrélation de Spearman distance/effet = indéfinie (effet constant).

- Niveau 8 : désaccord de cases min/médian/max = 0.00 % / 0.00 % / 0.00 % ; 120/120 paires sans désaccord argmax dans ces contextes ; corrélation de Spearman distance/effet = indéfinie (effet constant).

- Niveau 9 : désaccord de cases min/médian/max = 0.00 % / 0.04 % / 0.12 % ; 20/120 paires sans désaccord argmax dans ces contextes ; corrélation de Spearman distance/effet = 0.187.

- Niveau 10 : désaccord de cases min/médian/max = 0.00 % / 0.00 % / 0.08 % ; 67/120 paires sans désaccord argmax dans ces contextes ; corrélation de Spearman distance/effet = 0.242.

- Niveau 11 : désaccord de cases min/médian/max = 0.00 % / 0.04 % / 0.08 % ; 38/120 paires sans désaccord argmax dans ces contextes ; corrélation de Spearman distance/effet = 0.169.

- Niveau 12 : désaccord de cases min/médian/max = 0.04 % / 0.58 % / 1.04 % ; 0/120 paires sans désaccord argmax dans ces contextes ; corrélation de Spearman distance/effet = 0.802.

Les codes fins restent ici ceux de la solution connue : ce test ne régénère pas les niveaux suivants après substitution. Il peut donc masquer un rôle des codes précoces dans le conditionnement du générateur. Une paire sans changement argmax peut conserver des probabilités différentes. Ni ce test ni une petite distance géométrique ne prouvent une interchangeabilité globale. Les matrices complètes sont conservées dans results/functional_substitutions.json.

**Complément post-hoc sur le niveau constant** : suppression de la contribution résiduelle du niveau 5 hard, puis régénération des niveaux suivants, sur 128 puzzles test. Réussite 82.03 %, variation 0.00 %, IC [0.00 %, 0.00 %], 4 grilles modifiées ; sham identique à la génération native. Le niveau constant peut agir comme un décalage commun : ce test n'établit pas qu'il encode des différences entre puzzles. Il a été ajouté après observation de la faible diversité et reste exploratoire. Si aucun succès ne change de statut, l'IC bootstrap nul est dégénéré et ne prouve pas un effet populationnel nul.

Aucun changement de statut succès/échec sur 128 puzzles : borne supérieure Wilson à 95 % du taux de changements de statut 2.91 %. Cette borne complète l'IC bootstrap dégénéré [0,0] ; elle n'établit pas une équivalence de performance à toute précision.

## 5. Déterminisme et dépendance au batch — items 5–6

- repeat_0 : écart maximal d'état 0, écart relatif L2 0, codes divergents 0, tokens finaux divergents 0.

- repeat_1 : écart maximal d'état 0, écart relatif L2 0, codes divergents 0, tokens finaux divergents 0.

- repeat_2 : écart maximal d'état 0, écart relatif L2 0, codes divergents 0, tokens finaux divergents 0.

- batch_1 : écart maximal d'état 0.000122547, écart relatif L2 1.6e-06, codes divergents 0, tokens finaux divergents 0.

- batch_8 : écart maximal d'état 0.000122547, écart relatif L2 1.6e-06, codes divergents 0, tokens finaux divergents 0.

- reversed : écart maximal d'état 0, écart relatif L2 0, codes divergents 0, tokens finaux divergents 0.

- composition_64 : écart maximal d'état 0, écart relatif L2 0, codes divergents 0, tokens finaux divergents 0.

Les répétitions testent 32 puzzles hard, trois fois, à poids et entrées constants. Les comparaisons de tailles 1/8/32, ordre inversé et composition élargie à 64 séparent l'arrondi FP32 d'un changement discret de décision. Ces résultats sont locaux au CPU et ne garantissent pas un comportement identique sur CUDA ou MPS.

## 6. Symétries et stabilité du score — item 4

- **identity**, n=128 : réussite 82.03 %.

- **digits**, n=128 : réussite 85.16 %, accord de grille après réalignement 79.69 %, variation absolue moyenne de confiance 0.041, IC [0.031, 0.051]; différence de réussite vs identité 3.12 %, IC [-2.36 %, 9.38 %].

- **rotation**, n=128 : réussite 85.16 %, accord de grille après réalignement 75.00 %, variation absolue moyenne de confiance 0.056, IC [0.044, 0.069]; différence de réussite vs identité 3.12 %, IC [-3.91 %, 10.16 %].

- **reflection**, n=128 : réussite 80.47 %, accord de grille après réalignement 79.69 %, variation absolue moyenne de confiance 0.045, IC [0.034, 0.056]; différence de réussite vs identité -1.56 %, IC [-7.03 %, 4.69 %].

Les ratios de variance intra/inter-puzzles des résumés, niveaux 1 à 12, valent : 1.30, 1.20, 0.90, 0.73, 0.32, 0.35, 0.62, 1.03, 1.70, 2.00, 2.15, 1.72. Ces coordonnées ne sont pas garanties invariantes aux transformations. Une forte variance ne constitue donc pas une preuve de bruit ; l'équivalence des réponses réalignées et le comportement de la sonde gelée sont plus directement interprétables. Une seule permutation des chiffres et deux transformations spatiales sont testées ; les 128 variantes partagent leurs puzzles sources, et les IC les rééchantillonnent par source.

## 7. Généralisation et difficulté — items 7–8

Le croisement checkpoint/corpus utilise les mêmes 256 puzzles par corpus pour tous les checkpoints. Il sépare ainsi le changement de difficulté évaluée du changement des poids, dans les limites de cet échantillon.

![Modèles et corpus](figures/05_cross_models.png)

- **easy** : easy 99.61 %, medium 88.67 %, hard 44.53 %.

- **medium** : easy 98.44 %, medium 96.09 %, hard 66.41 %.

- **hard** : easy 100.00 %, medium 98.83 %, hard 86.33 %.

Sur le corpus hard commun, l'écart hard–easy est de 41.80 %, IC bootstrap apparié [35.16 %, 48.05 %]. Ces comparaisons secondaires sont exploratoires, sans correction de multiplicité ; près du plafond sur easy, les petits écarts demandent davantage de puzzles pour être interprétés. Le profil observé ne permet pas d'assimiler toutes les difficultés à un même modèle.

Un corpus indépendant de 256 puzzles à 30 indices a été produit par backtracking aléatoire puis retrait glouton d'indices. Chaque puzzle possède une solution unique selon backtracking MRV et Algorithm X. Les solutions équivalentes aux solutions officielles selon les symétries testées ont été exclues. Réussite hard : **219/256 = 85.55 %**, IC [80.71 %, 89.33 %], contre 86.30 % sur le corpus officiel complet.

Nombre moyen de candidats initiaux par case vide : officiel 3.307, nouveau 3.299. Proportion initiale de célibataires nus : officiel 0.042, nouveau 0.041. Différences de moyennes standardisées des six proxies : 0.000, -0.054, -0.081, -0.043, -0.010, -0.002 ; la variance nulle de la proportion de cases vides (30 indices partout) ne porte pas d'information de difficulté.

Ce corpus n'est pas uniforme sur les Sudokus et le retrait glouton modifie la distribution. Un écart de performance mesure un décalage de distribution combinant structure, difficulté et mécanisme de génération ; il ne permet pas de distinguer mémorisation et généralisation sans accès à l'entraînement. La contamination reste indéterminée.

Les proxies de difficulté sont la proportion de cases vides, moyenne et dispersion du nombre initial de candidats, proportion de célibataires nus, proportion de cases remplies par leur propagation et nombre de tours de propagation. Ils ne représentent pas toute la difficulté logique ou humaine. La baseline les inclut ; le contrôle difficulté seule et la stratification par quartiles train apportent un diagnostic supplémentaire, sans ajustement causal exhaustif.

- Strate 0, n=152, réussite 93.42 % : AUROC états 0.879, baseline 0.794, combinaison 0.879.

- Strate 1, n=143, réussite 90.91 % : AUROC états 0.941, baseline 0.936, combinaison 0.941.

- Strate 2, n=141, réussite 85.82 % : AUROC états 0.918, baseline 0.882, combinaison 0.916.

- Strate 3, n=164, réussite 75.00 % : AUROC états 0.871, baseline 0.882, combinaison 0.874.

## 8. Discrimination, calibration et potentiel de monitoring — items 9–10

Toutes les probabilités ci-dessous sont des sorties de sondes externes. Une AUROC élevée signifie que le score ordonne succès et erreurs ; elle n'implique pas calibration ni mécanisme métacognitif natif. Un prior constant peut être raisonnablement calibré et possède une AUROC de 0,5.

- **difficulty**, niveau 6 : AUROC 0.760, Brier 0.1088, ECE à 10 intervalles 0.0208, erreurs test 84/600.

- **baseline**, niveau 6 : AUROC 0.890, Brier 0.0792, ECE à 10 intervalles 0.0321, erreurs test 84/600.

- **hidden**, niveau 6 : AUROC 0.907, Brier 0.0757, ECE à 10 intervalles 0.0409, erreurs test 84/600.

- **combined**, niveau 6 : AUROC 0.908, Brier 0.0752, ECE à 10 intervalles 0.0429, erreurs test 84/600.

- **shuffled_hidden**, niveau 6 : AUROC 0.553, Brier 0.1202, ECE à 10 intervalles 0.0126, erreurs test 84/600.

- **prior**, niveau 6 : AUROC 0.500, Brier 0.1206, ECE à 10 intervalles 0.0130, erreurs test 84/600.

Exemple de dissociation : le prior constant présente une ECE de 0.013, contre 0.043 pour la combinaison, tout en ayant une AUROC de 0,5 contre 0.908. Dans cette population où les succès dominent, une calibration moyenne favorable ne garantit donc pas une capacité à distinguer les erreurs.

Le Brier combine calibration et résolution ; l'ECE dépend du découpage et de l'effectif par intervalle. Les courbes de risque/couverture conservent tous les scores ex aequo ensemble. Ici, les 6 000 puzzles officiels sont uniques : validité et respect des indices suffisent à caractériser la réussite. L'AUROC de la baseline atteint 1 au dernier niveau hard ; ce résultat est compatible avec l'accès au vérificateur externe, sans supposer un monitoring interne. Les derniers niveaux peuvent révéler la validité de la réponse déjà produite : cela ne constitue pas une anticipation de l'erreur.

![Sondes par niveau](figures/03_probes.png)

![Calibration et sélection](figures/04_calibration.png)

Les réplications suivantes ajustent séparément les sondes sur chaque checkpoint, sur des familles de solutions partagées entre difficultés, avec la même procédure et des partitions de 1 000/400/600. Le faible nombre d'erreurs, surtout sur easy, limite fortement la précision de l'AUROC et interdit d'en tirer une hiérarchie générale entre modèles.

- **easy** : erreurs test 3/600 ; AUROC baseline 0.888, états 0.954, combinaison 0.955. Gain combiné–baseline 0.0664, IC [-0.007, 0.212].

- **medium** : erreurs test 23/600 ; AUROC baseline 0.808, états 0.893, combinaison 0.894. Gain combiné–baseline 0.0866, IC [-0.007, 0.182].

Sur le corpus nouveau, les sondes hard gelées au niveau 6 donnent : AUROC baseline 0.924, états 0.963, combinaison 0.964. Les métriques complètes, calibration comprise, figurent dans results/transfer_confidence.json. Le transfert direct entre checkpoints est conservé séparément et limité aux 600 familles de solutions du test hard, exclues de l'entraînement et de la validation des sondes hard ; leurs coordonnées cachées ne sont pas alignées, aussi son échec éventuel ne prouve pas l'absence d'information décodable.

## 9. Sonde de contrainte naked single — item 11, exploratoire

Une sonde locale utilise les 512 activations des positions correspondant aux cases au dernier niveau. La cible indique si une case initialement vide possédait exactement un candidat à partir des seuls indices. Entraînement : 512 puzzles de train ; validation : 128 ; test : 256 puzzles, 13056 cases vides, dont 547 positives. Les puzzles, et non les cases, définissent les partitions et les unités bootstrap.

AUROC états : **0.877** ; labels d'entraînement mélangés : **0.496** ; contraste IC [0.352, 0.410]. Le calcul explicite du nombre de candidats atteint par construction une AUROC de 1 pour cette cible.

Ce résultat concerne la décodabilité d'une contrainte initiale dans un état final. Il ne localise pas un circuit monosémantique, ne montre pas que la règle est calculée à cet endroit ni qu'elle est utilisée causalement. Le candidat explicite est un contrôle sémantique de la cible, pas un modèle appris concurrent. Une recherche de neurones spécifiques et des patchings de règles sélectifs restent nécessaires avant toute affirmation de circuit.

## 10. Ablation causale — item 12, exploratoire

Une sonde sur les 512 moyennes d'activations du niveau 6 fournit une direction linéaire, apprise sur train et sélectionnée sur validation. L'intervention retire la projection des activations centrées sur cette direction, à la couche 8 du niveau 6 uniquement ; le reste de la génération se poursuit normalement. Le centrage utilise les moyennes train. Les 600 puzzles test sont les seules entrées utilisées pour mesurer les effets causaux.

Cinq directions aléatoires unitaires reçoivent, pour chaque token et puzzle, exactement la norme absolue de projection de l'intervention ciblée ; leur signe suit leur propre projection. Ce contrôle apparie la taille de la perturbation, sans prétendre apparier toute sa géométrie. Le sham ne modifie aucun état.

- **sham** : réussite 86.00 %, variation vs génération native 0.00 %, IC [0.00 %, 0.00 %] ; erreurs introduites 0, erreurs corrigées 0, grilles modifiées 0.

- **confidence** : réussite 86.50 %, variation vs génération native 0.50 %, IC [0.00 %, 1.17 %] ; erreurs introduites 0, erreurs corrigées 3, grilles modifiées 19.

- **random_0** : réussite 86.17 %, variation vs génération native 0.17 %, IC [0.00 %, 0.50 %] ; erreurs introduites 0, erreurs corrigées 1, grilles modifiées 13.

- **random_1** : réussite 86.00 %, variation vs génération native 0.00 %, IC [-0.50 %, 0.50 %] ; erreurs introduites 1, erreurs corrigées 1, grilles modifiées 15.

- **random_2** : réussite 86.50 %, variation vs génération native 0.50 %, IC [0.00 %, 1.17 %] ; erreurs introduites 0, erreurs corrigées 3, grilles modifiées 16.

- **random_3** : réussite 86.17 %, variation vs génération native 0.17 %, IC [-0.33 %, 0.83 %] ; erreurs introduites 1, erreurs corrigées 2, grilles modifiées 17.

- **random_4** : réussite 86.67 %, variation vs génération native 0.67 %, IC [0.17 %, 1.33 %] ; erreurs introduites 0, erreurs corrigées 4, grilles modifiées 21.

L'effet ponctuel ciblé vaut 0.50 %, contre une plage de 0.00 % à 0.67 % pour les cinq directions aléatoires. L'intervention ciblée n'apporte donc pas, à elle seule, une preuve de spécificité métacognitive ; cette comparaison est descriptive et ne transforme pas cinq contrôles en test exhaustif de directions.

![Interventions causales](figures/06_causal.png)

Ces interventions testent l'influence d'une direction sur la résolution. Une baisse de performance peut refléter une information utile à la tâche ou une perturbation générique. Elle n'établit pas un contrôle métacognitif natif. Cinq contrôles aléatoires ne permettent pas une estimation précise des queues de leur distribution. Un effet nul n'exclut pas redondance, compensation ou mauvaise localisation de la direction.

## 11. Progression de la représentation — item 13, exploratoire

Une projection gaussienne fixe réduit les résumés de 1 024 à 128 dimensions. Une régression ridge (alpha=100, sans sélection sur test) prédit les 81 distributions de chiffres à partir des états de chaque niveau ; les métriques portent sur les cases initialement vides. Un contrôle utilise les seuls indices, encodés one-hot puis projetés en 128 dimensions, avec la même ridge. Les cibles ne sont utilisées qu'à l'entraînement ou pour le calcul final des métriques.

Les états sont observés avant la tête qui prédit les codes du niveau ; les décodages intermédiaires sont calculés après ajout des codes de ce niveau. Les deux courbes ne représentent donc pas exactement le même instant du calcul.

![Décodabilité et révisions](figures/07_hierarchy.png)

Le profil est descriptif : aucun test de rupture, analyse de taille finie ou réplication par graines ne justifie de parler de transition de phase. Les révisions correct→incorrect montrent directement que l'affinement n'est pas obligatoirement monotone. Une solution décodable n'implique ni engagement irréversible ni utilisation causale de cette lecture linéaire.

## 12. Limites et portée des conclusions

- Trois checkpoints Sudoku, une version publiée et un backend CPU ; aucune conclusion sur toute la famille NCM ou sur les modèles de texte.
- Un découpage de données et une graine d'entraînement des sondes ; bootstrap conditionnel aux sondes ajustées. Les performances de train ou validation ne servent pas de preuves finales.
- Nombre d'erreurs limité ; particulièrement peu d'erreurs dans les réplications easy. Les métriques par strate et par niveau sont exploratoires.
- Symétries testées partielles ; absence de doublons dans cet audit ne prouve pas absence de contamination d'entraînement.
- Le corpus nouveau ajoute un décalage de générateur et ne définit pas un test de mémorisation.
- La baseline ne contient pas toutes les informations possibles des sorties ; un gain de la sonde combinée serait relatif à cette baseline précise.
- Entropie des codes, reconstruction et décodabilité ne prouvent pas une représentation conceptuelle humaine ni une conscience de sa fiabilité.
- Le test naked single ne caractérise pas de circuit monosémantique. L'ablation de confiance ne démontre pas le contrôle métacognitif.
- Aucun entraînement de NCM, aucune étude longitudinale de checkpoints et aucune mesure comparative de coût GPU ne sont réalisés.

## 13. Sources, preuves et reproduction

Les expériences sont exécutées depuis ce dossier. Pour reconstruire toutes les sorties (opération longue) :

```sh
./run.sh
```

Pour vérifier les artefacts existants et rejouer les sondes sauvegardées :

```sh
PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" .venv/bin/python scripts/verify.py
```

Pour recréer l'environnement local : Python 3.13, venv et installation de requirements.txt. Le fichier sources/requirements-lock.txt est un instantané de l'environnement complet, incluant des paquets système non utilisés. L'environnement de campagne utilise des paquets système en lecture ; le verrou enregistre les versions réellement résolues. Les variables de caches de run.sh doivent être conservées pour confiner les écritures. Les résultats bit à bit ne sont garantis ni entre versions BLAS ni entre architectures.

- **Rapport et méthode** : RAPPORT.md, TODO.md, sources/TODO_protocol_v1.md, protocol.json et JOURNAL.md.
- **Code exécuté** : scripts/common.py, prepare.py, extract.py, analyze.py, solvers.py, experiments.py, upstream_parity.py, report.py et verify.py ; smoke_checks.py, recover_summaries.py et continue.py assurent contrôles, récupération des seuls résumés et ordonnancement ; run.sh orchestre une reproduction séquentielle. La campagne initiale exécute certaines étapes indépendantes simultanément, avec un thread de calcul par processus.
- **Code des modèles** : vendor/ncm_mac et vendor/upstream-ncm ; ce dernier reste intact. vendor/reference_metacognition est une copie de référence non importée par les expériences ; son inclusion ne signifie pas que ses expériences ont été exécutées ici.
- **Entrées exactes** : artifacts/sudoku, data/sudoku et data/external.json. Les poids ne sont pas modifiés. Les fichiers pickle officiels sont chargés par un lecteur n'acceptant que les données simples.
- **Preuves par puzzle** : results/hard.npz, easy.npz et medium.npz ; préfixes, références, codes encodés/générés, décodages intermédiaires, états résumés et statistiques sans référence. hard.npz contient aussi les activations locales finales.
- **Sondes** : results/probe_models.pkl, naked_single_probe.pkl, probe_predictions.npz et causal_direction.npz. Ne charger les pickles de sondes qu'en provenance de ce dossier de confiance.
- **Interventions** : results/causal_predictions.npz ; résultats de symétries, substitutions, croisement et transfert dans les fichiers JSON/NPZ correspondants.
- **Inspection simple** : results/summary.csv, probes.csv et predictions.csv ; figures en PNG et SVG.
- **Intégrité** : results/input_manifest.json, final_manifest.json et verification.json ; le dernier indique les contrôles effectivement passés. Les logs sont dans logs/.
- **Archive de sources** : sources_utilisees.zip contient les scripts, les deux implémentations des modèles, les configurations et le protocole ; les gros poids et les traces restent disponibles dans les dossiers ci-dessus.

Références primaires : [article NCM](https://arxiv.org/html/2610.14162v1), [dépôt officiel](https://github.com/HazyResearch/ncm), [checkpoints publiés](https://huggingface.co/hazyresearch/ncm-checkpoints), [corpus publiés](https://huggingface.co/datasets/hazyresearch/ncm-datasets). Les copies de l'article consultées sont dans sources/ et leurs empreintes sont conservées. Les résultats numériques de ce rapport sont calculés localement, non repris de ces pages.

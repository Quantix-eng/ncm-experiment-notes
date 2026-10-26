# Étape 3a — O, C et E sur les états réels du NCM

25 octobre 2026. Pilote sur le checkpoint Sudoku hard gelé, CPU FP32, un thread, lots unitaires. Scores et statistiques calculés en FP64 à partir des logits FP32 conservés. Rapports en Markdown uniquement. Aucun ajustement des niveaux, doses, métriques ou seuils après les nouvelles inférences.

## Question et limites des étiquettes

Le protocole analytique reste-t-il techniquement applicable aux états NCM, et quelles dépendances fonctionnelles les interventions permettent-elles de mesurer ? La réussite du transport logiciel n'établit ni validité externe complète ni découverte d'un mécanisme. Le pourcentage de 15 % de la calibration concernait une règle et une composition de classes particulières ; le score de 100 % était attendu pour un diagnostic conçu à partir des équations des témoins. Ces chiffres ne sont pas des garanties sur un réseau appris.

Un résultat non identifiable est recevable. Il ne prouve pas que le réseau est plus complexe que les témoins : effets trop faibles, intervention inadaptée, manque de précision, décalage de distribution et plusieurs mécanismes compatibles restent possibles. Une signature intermédiaire ne sera pas appelée mélange causal certain. Aucune vérité mécaniste n'est attribuée au NCM.

## Données et absence de sélection

Utiliser les 16 lignes Sudoku de `benchmark_causal_revision/data/families.json`, dans leur ordre conservé. A, B et Aplus sont les 48 entrées. Les solutions et unicités ont déjà été vérifiées mais aucune de leurs nouvelles sorties NCM n'est examinée avant gel de ce protocole et du code. Les données et les décisions analytiques antérieures sont déjà connues : pas de nouveau corpus indépendant ni de test secret. Toutes les sources sont conservées, y compris les échecs natifs. Aucune nouvelle famille ne remplace une source défavorable.

Le masque primaire contient les cases vides dans **les trois variantes**, excluant ainsi l'indice discriminant A/B et l'indice supplémentaire Aplus. Pour la préférence A/B, on retient dans ce masque les cases où les solutions diffèrent. Il en existe au moins une dans chaque famille. Les références sont utilisées uniquement pour l'évaluation, jamais pour générer des codes, choisir la direction d'une perturbation selon son effet ou piloter une correction.

## Chemin natif, point de coupure et arithmétique

Le point de coupure est l'accumulation après le niveau 9 sur 12, choisie pour continuité avec les campagnes antérieures, sans recherche du niveau le plus favorable. La suite comprend les contributions 10, 11 et 12. Les 89 positions de cet état ne sont pas 89 variables symboliques indépendantes ; attention et rééchantillonnage limitent une interprétation locale.

Pour chacune des 48 entrées, comparer l'adaptateur au chemin `SudokuModel.predict` : codes, état final, logits, et reconstructions cumulatives natives. Exiger la parité bit à bit et la remise à zéro des caches. Une reprise sans perturbation depuis le niveau 9 doit reproduire la suite et le résultat natifs. Arrêt sur une défaillance ; tout correctif post-gel nécessite conservation de l'ancienne version et amendement explicite.

Les suites résiduelles sont conservées contribution par contribution. Toutes les branches figées ajoutent les résidus dans l'ordre 10→11→12, sans remplacer l'addition séquentielle par `z+sum(T)`. Les écarts de réassociation sont un diagnostic, pas un motif de modifier les observations. Les états de branches sont clonés ; le cache utilisé pour une reprise appartient exclusivement à l'entrée réceptrice. Aucun cache de préfixe donneur n'est transféré avec un état.

## O — observations

Conserver les 12 reconstructions et logits de chacune des 48 trajectoires natives, exactitudes de case et de grille, respect des indices et du format. Compter corrections et dégradations sur le masque commun, aux 11 transitions, sans sélectionner rétrospectivement un niveau spectaculaire. Aucun mécanisme n'est classé à partir de ces seules courbes. Le dernier niveau n'est pas une référence pour juger la justesse : les vraies solutions sont disponibles.

## C — perturbations et continuations croisées

Exécuter C sur A et B, soit 32 ancrages. Dans chaque famille, tirer deux positions distinctes dans le masque vide commun à partir de la graine fixée, indépendamment des sorties. Chaque direction gaussienne est portée par les 256 coordonnées d'une seule position de l'état 89×256. Les supports sont disjoints ; on teste les deux signes de chaque direction. Il s'agit d'interventions continues, potentiellement hors distribution, et non de substitutions de codes sémantiques.

La norme L2 cible de chaque direction est `0.5 * ||z9||_F / sqrt(89)` pour l'ancrage concerné. Les deux directions ont donc la même norme globale et deux signes, soit quatre perturbations fixes par ancrage. Cette dose représente environ 5,3 % de la norme globale du latent, concentrés sur une position. Elle est choisie avant outcomes pour ce pilote ; aucune dose de secours ou escalade n'est autorisée si le dommage est trop faible. Vérifier les normes effectives en FP32 à tolérance relative 2e-6 et sauvegarder les deltas réels.

Pour `z`, `delta`, suite native `T0`, suite régénérée `Tdelta` et suite adaptée à l'autre support de même signe `Tother` :

- Y00 : état natif et suite native ;
- Y10 : état perturbé et suite native figée ;
- Y11 : état perturbé et suite recalculée ;
- Y01 : état natif et suite adaptée à la perturbation ;
- Y1wrong : état perturbé et suite adaptée à l'autre perturbation de même norme.

Enregistrer logits, codes des suites, deltas et contributions de chaque suite. Le décodeur ne consulte pas le puzzle directement ; les effets du puzzle passent ici par les états et suites produits avec son préfixe. Le témoin mal apparié apparie norme et signe ; il ne garantit pas l'appariement de difficulté, de géométrie ou de densité.

La NLL est moyennée sur les cases vides communes et évaluée envers la solution de l'ancrage. Contrastes : dommage `D=L10-L00`, récupération `R=L10-L11`, interaction croisée `I=L10+L01-L11-L00`, spécificité relative `S=L1wrong-L11`. Les contrastes ne sont pas des proportions médiées ni des mesures d'intention.

Moyenner d'abord les quatre perturbations et les deux ancrages dans chaque famille. Bootstrap percentile sur les **16 familles entières**, 20 000 tirages. IC 95 % descriptifs et IC 98,75 % pour les quatre contrastes D/R/I/S (Bonferroni 5 %). Pas de multiplication artificielle de l'effectif par cases, branches ou signes.

Marge fixée : 0,01 nat par case vide, choix exploratoire sans validation indépendante d'utilité. Une récupération fonctionnelle utile et spécifique est soutenue dans ce pilote seulement si les **quatre bornes inférieures ajustées dépassent 0,01**. Si l'IC ajusté de D est entièrement contenu dans [-0,01 ; 0,01], parler d'équivalence du dommage moyen dans cette marge, pas de tolérance de chaque entrée. Autrement : conclusion insuffisante ou absence de confirmation, avec tous les effets et IC rapportés. Des codes qui changent ne suffisent pas. Les résultats par signe ou support sont descriptifs et ne remplacent pas cette décision.

## E — entrées, états et suites

Deux comparaisons : A/B (réponse modifiée) et A/Aplus (réponse conservée). Pour chaque comparaison, exécuter les **12 conditions** `2 préfixes receveurs × 2 états z9 donneurs × {suite native gauche figée, suite native droite figée, suite recalculée}`. Les états complets z9 sont échangés, sans alignement appris ni retouche par case. Vérifier les diagonales contre le natif et l'invariance au préfixe receveur des sorties avec état et suite figés.

Conserver aussi le décodage sans toute la suite (déjà observé au niveau 9). L'ablation de suite et les échanges restent des interventions artificielles ; aucune classe de témoin n'est copiée par une règle d'égalité numérique FP64.

Pour A/B, définir la préférence `m(Y)=moyenne(log p(S_B)-log p(S_A))` sur les cases de réponse différentes du masque commun. m>0 favorise B. Soit `h_ij` la préférence après suite recalculée avec entrée i et état j. Les effets sont :

- entrée : `u=0.5*((h_BA-h_AA)+(h_BB-h_AB))` ;
- état : `v=0.5*((h_AB-h_AA)+(h_BB-h_BA))` ;
- interaction : `w=h_BB-h_BA-h_AB+h_AA` ;
- étendue native : `g=h_BB-h_AA`.

Rapporter u, v et w en nats pour **toutes** les familles, avec IC descriptifs groupés. Sous diagonales natives, `u+v=g` par identité algébrique : leurs ratios ne sont pas deux ressources indépendantes ni des pourcentages causaux d'un calcul.

Le diagnostic descriptif par famille exige les **deux grilles natives exactes**, `g>0.1 nat` et une interaction normalisée `|w/g|<=0.2`. Sans ces conditions : « non identifiable » ou « interaction non additive », sans exclusion de la famille des statistiques. Sous ces conditions : `u/g>=0.8` avec ratio dans [0,1] → préférence dominée par l'entrée dans ce test ; `u/g<=0.2` → préférence dominée par l'état ; entre les deux → « deux sensibilités : mélange ou mécanisme non modélisé ». Un ratio hors [0,1] n'est pas forcé dans une classe : « non monotone/non identifiable ». Ces seuils sont heuristiques, préfixés, sans preuve d'identification d'un programme interne. Aucun diagnostic individuel ne reçoit un statut confirmatoire par le seul bootstrap global.

Pour A/Aplus, la réponse est identique : ne pas normaliser par une différence de solutions nulle. Rapporter variations de NLL, distance en variation totale des probabilités et changements d'argmax sous échanges réciproques, sur le même masque. L'indice supplémentaire peut améliorer le calcul ; tout changement n'est donc pas un faux positif. Le nombre d'indices reste un facteur non neutralisé.

## Diagnostics de portée et conservation

Conserver tous les échecs, format, indices, distances relatives entre états natifs, changements de codes de suite, normes, et fraction de coordonnées perturbées hors de l'enveloppe min/max coordonnée par coordonnée des 48 états natifs z9. Cette enveloppe décrit une amplitude ; elle n'est pas un test de densité ni un certificat sur distribution. L'échange d'un état natif peut demeurer hors distribution conditionnellement à l'autre préfixe.

Les 16 familles ne suffisent pas à établir une généralité architecturale, des effets indépendants du checkpoint ou une puissance élevée. Le générateur de puzzles est non uniforme et les nombres d'indices variables. Pas de revendication de nouveauté face à MIB/Tyen dans ce pilote, ni de preuve de métacognition ou de complexité intrinsèque.

Avant inférence, épingler ce protocole, paramètres, tous les scripts, poids/configuration, données, sources importées et documents historiques lus. Gel local, pas prérégistration indépendante. Un audit séparé recalcule les scores, les comptes et les décisions, recontrôle les puzzles par deux solveurs, puis rejoue intégralement C et E sur les deux premières familles, sans changement des entrées ou de la dose. Rapporter les contrôles logiciels comme tels.

Arrêt au rapport de l'étape 3a ; pas de nouvelle calibration opportuniste ni de travail Wyner dans cette exécution.

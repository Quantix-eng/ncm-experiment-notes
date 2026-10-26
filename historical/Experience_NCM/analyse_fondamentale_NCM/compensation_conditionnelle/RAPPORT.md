# Codec, adaptation et test de compensation conditionnelle

Campagne expérimentale du 24 octobre 2026, CPU FP32, checkpoints gelés. [Article anglais](PAPER.md), [PDF](PAPER.pdf), [protocole gelé](OBJECTIFS_ET_PROTOCOLE.md). Les six campagnes précédentes restent intactes ; le patching et les puzzles à indice disambiguïsant sont hors périmètre.

## Résultats principaux

**Plan croisé sur 256 nouveaux puzzles : R=-0.001641 nat, IC 97,5 % [-0.02636, 0.02538] ; I=-0.003803 nat, IC 97,5 % [-0.00725, -0.00095].** Le critère co-primaire annoncé n'est pas satisfait : une compensation utile et spécifique à l'omission n'est pas établie. Le récit « le générateur tente de compenser » ne peut pas être déduit du seul changement de ses codes.

**Le générateur LM1B small est exécuté, pas seulement son tokenizer.** Sur 128 trajectoires natives, 416/2340 positions adoptant à 9 leur token raw final l'abandonnent à 10 ou 11 : **17.78%**, IC 95 % [0.16111, 0.19603] en proportions. Sur le masque secondaire des tokens finaux non spéciaux : 410/2274 = 18.03%. L'ancre finale est rétrospective, pas une réponse correcte.

**Codec LM1B oracle :** 97/1675 corrections à 9 sont ensuite dégradées, soit 5.79%, sur 128 phrases validation réutilisées, padding exclu. Ce contrôle possède de vraies cibles de reconstruction et doit rester distinct des textes générés.

## 1. Ce que le récit peut affirmer

Le contrôle oracle Sudoku antérieur démontre que le codec suffit à produire une reconstruction non monotone, même à partir de la bonne solution. Il ne démontre pas que tous les revirements du générateur proviennent exclusivement du codec. L'attention du décodeur et celle du générateur ne sont pas la même composante. La grille intermédiaire n'est pas réinjectée dans le générateur pour lui montrer ses erreurs.

Le titre retenu est **Apparent Answer Revision in NCM: Codec Dynamics and a Test of Conditional Residual Compensation**. La mention « a Test of » est conservée si la compensation n'est pas établie, afin de distinguer la question expérimentale d'une conclusion positive. L'observation sur Sudoku et LM1B étend la portée du phénomène au-delà du Sudoku ; elle ne prouve ni universalité, ni défaut général de NCM, ni correction sémantique en texte.

## 2. Protocole, données et incertitude

Omission de la contribution 9, conditions, métriques et critère gelés avant calcul. Les 600 hard test et les 256 externes antérieurs sont des réplications exploratoires déjà examinées. Le test principal porte sur 256 nouvelles solutions générées après le gel par backtracking aléatoire, puis retrait glouton jusqu'à 30 indices. Unicité vérifiée indépendamment par MRV et Algorithm X. Les clés D4/normalisation des chiffres sont exclues contre les solutions officielles, externes antérieures et nouvelles déjà retenues ; toutes les isomorphies et une contamination d'entraînement ne sont pas exclues.

L'effectif est un budget de précision fixé, sans garantie de puissance ; le générateur de Sudoku n'est pas uniforme. Aucun seuil, niveau, dose ou sous-groupe n'est sélectionné sur les nouvelles réponses. Les jeux restent séparés, sans regroupement selon le signe des résultats.

La perte L est la NLL des chiffres de référence sur les cases initialement vides, moyennée par puzzle. R=L10-L11 mesure le bénéfice de régénérer la suite sous perturbation. I=L10+L01-L11-L00 retranche le bénéfice que la même suite adaptée apporte à l'état natif. Un R positif seul peut refléter une meilleure suite pour les deux états. Ces contrastes ne sont pas une décomposition causale unique ou une mesure d'intention de réparer.

Bootstrap percentile apparié par puzzle, 4 000 tirages ; IC descriptifs à 95 % et IC à 97,5 % pour les deux contrastes co-primaires avec ajustement Bonferroni. Le critère annoncé exige les deux bornes basses à 97,5 % positives sur les nouveaux puzzles. La couverture bootstrap est approximative et conditionnelle à ces poids.

## 3. Les quatre conditions et les contrôles

Après niveau 9, z=z9 et δ=-r9. T0 est la somme native des résidus 10..12 ; Tδ celle de la suite régénérée depuis z8, à puzzle constant.

- Y00 : état natif et suite native.
- Y10 : état sans r9 et contributions natives suivantes figées.
- Y11 : état sans r9 et suite suivante régénérée.
- Y01 : état natif et suite adaptée à l'omission.

Les ajouts suivent l'ordre natif. La branche perturbée part directement de z8 pour éviter un aller-retour de soustraction en FP32 ; les écarts de réassociation des sommes sont sauvegardés. Les croisements sont des interventions artificielles, possiblement hors distribution. Un contrôle gaussien ajoute à la fin de Y10 une direction indépendante de même norme que ΔT=Tδ-T0 ; cela apparie l'amplitude, pas toute la géométrie ou la distribution.

Parité du runner avec le chemin natif sur 32 puzzles : codes, logits et accumulations 8/9 identiques. Les 600 baselines sont comparées aux traces antérieures ; les 128 premières omissions recoupent l'étude temporelle. Les caches et états de branches sont contrôlés. Les logits et les deux suites de codes sont conservés par puzzle.

## 4. Résultats complets du plan croisé

### hard_test, n=600
- Y00 : 516 grilles exactes ; NLL cases vides 0.454972 ; 0 grilles modifiées vs natif.
- Y10 : 514 grilles exactes ; NLL cases vides 0.449336 ; 17 grilles modifiées vs natif.
- Y11 : 515 grilles exactes ; NLL cases vides 0.453295 ; 63 grilles modifiées vs natif.
- Y01 : 517 grilles exactes ; NLL cases vides 0.454412 ; 59 grilles modifiées vs natif.
- random_tail_delta : 401 grilles exactes ; NLL cases vides 0.464175 ; 168 grilles modifiées vs natif.
- R : -0.003959 nat ; IC 95 % [-0.02655, 0.01786] ; IC 97,5 % [-0.02967, 0.02037].
- I : -0.004520 nat ; IC 95 % [-0.00739, -0.00151] ; IC 97,5 % [-0.00787, -0.00108].
- Codes modifiés entre suites : 8975 ; MSE latente moyenne Y00/Y10/Y11/Y01 : 0.125744 / 0.161364 / 0.160526 / 0.130659.
- Y10 vers Y11 : 6 grilles sauvées, 5 échecs introduits ; gain de réussite 0.17 points, IC 95 % en proportions [-0.00833, 0.01333].
### external_previous, n=256
- Y00 : 219 grilles exactes ; NLL cases vides 0.527539 ; 0 grilles modifiées vs natif.
- Y10 : 216 grilles exactes ; NLL cases vides 0.517220 ; 8 grilles modifiées vs natif.
- Y11 : 216 grilles exactes ; NLL cases vides 0.475835 ; 28 grilles modifiées vs natif.
- Y01 : 220 grilles exactes ; NLL cases vides 0.483869 ; 26 grilles modifiées vs natif.
- random_tail_delta : 164 grilles exactes ; NLL cases vides 0.521873 ; 75 grilles modifiées vs natif.
- R : 0.041386 nat ; IC 95 % [0.00381, 0.09190] ; IC 97,5 % [0.00026, 0.09858].
- I : -0.002284 nat ; IC 95 % [-0.00671, 0.00194] ; IC 97,5 % [-0.00741, 0.00253].
- Codes modifiés entre suites : 3705 ; MSE latente moyenne Y00/Y10/Y11/Y01 : 0.131101 / 0.166073 / 0.162340 / 0.132430.
- Y10 vers Y11 : 1 grilles sauvées, 1 échecs introduits ; gain de réussite 0.00 points, IC 95 % en proportions [-0.01172, 0.01172].
### fresh, n=256
- Y00 : 222 grilles exactes ; NLL cases vides 0.356166 ; 0 grilles modifiées vs natif.
- Y10 : 221 grilles exactes ; NLL cases vides 0.349218 ; 3 grilles modifiées vs natif.
- Y11 : 224 grilles exactes ; NLL cases vides 0.350859 ; 23 grilles modifiées vs natif.
- Y01 : 224 grilles exactes ; NLL cases vides 0.354004 ; 23 grilles modifiées vs natif.
- random_tail_delta : 175 grilles exactes ; NLL cases vides 0.365382 ; 68 grilles modifiées vs natif.
- R : -0.001641 nat ; IC 95 % [-0.02418, 0.02160] ; IC 97,5 % [-0.02636, 0.02538].
- I : -0.003803 nat ; IC 95 % [-0.00676, -0.00124] ; IC 97,5 % [-0.00725, -0.00095].
- Codes modifiés entre suites : 3690 ; MSE latente moyenne Y00/Y10/Y11/Y01 : 0.117448 / 0.153873 / 0.153021 / 0.122108.
- Y10 vers Y11 : 4 grilles sauvées, 1 échecs introduits ; gain de réussite 1.17 points, IC 95 % en proportions [-0.00391, 0.03125].

![NLL croisée](figures/01_crossed_nll.png)

![Contrastes annoncés](figures/02_contrasts.png)

Un R favorable sur l'ancien externe ne suffit pas si la suite adaptée bénéficie aussi à l'état natif : le quatrième terme est nécessaire pour la question de spécificité. La réussite exacte est secondaire et ne remplace pas la NLL annoncée. Aucun IC recouvrant zéro ne prouve une équivalence de performance.

Sur les nouveaux puzzles, cosinus moyen de ΔT avec -δ : 0.0838. Ratio moyen ||δ+ΔT||/||δ|| : 1.1450. Une valeur supérieure à 1 ne correspond pas à une annulation euclidienne. La géométrie latente et le bénéfice de reconstruction peuvent diverger ; les pertes restent le test principal. Les MSE vers l'encodage de la solution sont des diagnostics utilisant la référence, pas des signaux de contrôle disponibles au modèle.

![Géométrie latente](figures/03_geometry.png)

La non-confirmation porte sur une omission complète, un checkpoint et ces données. Elle n'exclut pas d'autres formes de compensation à d'autres niveaux ou doses. La tolérance à l'omission et l'adaptation des codes sont compatibles avec les résultats, sans qu'une intention de réparer soit démontrée.

## 5. Générateur LM1B : exécution et observable

Checkpoint LM1B small publié, 97 065 600 paramètres, environ 390 Mo ; empreinte SHA-256 vérifiée contre l'entrée LFS officielle et révision épinglée identique à celle du tokenizer antérieur. Génération native eager, 128 trajectoires B=1 avec graines distinctes ; préfixe CLS=101, premier code échantillonné dans l'unigramme sauvegardé, température 1, top-k=0, top-p=1, CFG=0. Les douze longueurs sont 1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108 et 128 ; le niveau 9 n'est pas une résolution identique au Sudoku.

Chaque accumulation est reconstruite explicitement avant modification, pour éviter l'aliasing des historiques natifs. Le dernier état est égal au natif. Les deux premières graines et leurs douze lectures sont rejouées exactement. Le chargeur compare tous les tenseurs enregistrés au checkpoint et reconstruit les petites têtes partagées par copie ordinaire, afin que leurs vues utilisent les vrais biais chargés.

Incident documenté : le premier chargeur assignait les paramètres sans réactualiser les références des vues de biais partagés. Ses 46 trajectoires sont écartées et conservées sous data/invalid_loader_trajectories ; **aucune n'entre dans les résultats**. Le calcul corrigé reprend les mêmes graines et tous les paramètres de sampling. Un incident disque a motivé le chargement meta/mmap, la reconstruction exacte des buffers RoPE non persistants et le nettoyage de caches régénérables non manifestés. Les poids sources ne sont pas modifiés.

L'argmax brut du décodeur définit les lectures principales. Le post-traitement officiel fréquence/adjacence est reproduit uniquement pour la sortie finale d'inspection ; il diffère du raw sur 14.37% des positions. Ces heuristiques ne sont pas mélangées aux courbes de révision. Aucune perplexité GPT2 ni qualité sémantique n'est évaluée.

En génération inconditionnelle, le token final brut sert seulement d'ancre rétrospective. Une adoption à k signifie différent de l'ancre à k-1, égal à k ; un abandon signifie différent à un niveau ultérieur. Le dernier accord vaut 100 % par construction et les abandons après adoption à 11/12 sont structurellement nuls. Toutes les transitions sont rapportées, sans sélection d'un pic favorable.

![Révisions LM1B](figures/04_text_generation.png)

L'analyse principale utilise les 128 positions fixes. Le masque secondaire retient les identifiants finaux 104..30521 : il exclut tokens spéciaux, padding et lignes inutilisées, mais conditionne sur le futur. Les IC rééchantillonnent les trajectoires, pas les tokens supposés indépendants. Exemple WordPiece aux niveaux 8..12 : **was → unemployment → unemployment → rate → unemployment**. Identifiants, graines et positions des exemples sont dans results/text_counterexamples.json.

## 6. Codec texte oracle et portée de la généralisation

128 phrases validation déjà sélectionnées sont encodées puis décodées progressivement ; les références non-PAD permettent ici de parler de corrections et dégradations de reconstruction. L'exactitude finale vaut 99.84%. Les lots de quatre sont vérifiés contre des lots unitaires sur les quatre premières phrases. Ces références sont distinctes des générations inconditionnelles et restent exploratoires, car déjà examinées.

![Codec oracle LM1B](figures/05_text_oracle.png)

Une révision du token généré n'est pas une amélioration de sens, et une ancre finale de faible qualité reste une ancre. La présence de non-monotonie sur un générateur texte et sur Sudoku établit des exemples hors Sudoku, pas une loi de toute la famille NCM. Une attribution à l'architecture exigerait des entraînements et architectures contrôlés.

## 7. Conclusion et reproduction

Le critère co-primaire annoncé n'est pas satisfait : une compensation utile et spécifique à l'omission n'est pas établie. Le résultat solide distingue suffisance du codec pour produire des revirements, adaptation effective des codes et compensation utile. Il ne démontre ni origine exclusivement codec de toutes les révisions, ni transition de phase, ni contrôle métacognitif natif. Patching d'attention et puzzles disambiguïsants restent réservés à une autre étude.

Reproduction : `./run.sh`. Les poids antérieurs sont lus dans le parent ; seul le nouveau générateur est téléchargé dans ce dossier. Les NPZ contiennent logits et codes croisés, graines et lectures texte ; results/verification.json annonce le rejeu effectivement passé, results/final_manifest.json fige les artefacts. sources_compensation.zip archive les scripts, modèles, configurations et protocoles, sans gros poids. Les six manifestes antérieurs sont contrôlés. Les incidents et leurs corrections sont conservés dans JOURNAL.md et logs/.

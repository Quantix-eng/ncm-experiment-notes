# Des revirements observés aux mécanismes qui les produisent

Note de recherche — 24 octobre 2026. Analyse du code, réanalyse descriptive des traces antérieures et propositions d'expériences. **Les expériences de compensation, de patching et de transfert proposées ci-dessous ne sont pas exécutées dans cette note.** Les rapports et manifestes des campagnes précédentes restent inchangés.

## Conclusion de travail

La question la plus féconde devient : **quelle part des révisions appartient à la reconstruction progressive du codec, et quelle part dépend d'un calcul adaptatif du générateur ?** Une réfutation solide n'a pas besoin d'être remplacée par une découverte positive artificielle. En revanche, cette distinction fournit un programme mécaniste testable et des explications concurrentes précises.

Un contrôle supplémentaire, réalisé ici sur les traces existantes, fait avancer cette distinction : **le codec produit des corrections puis des dégradations même lorsqu'il encode directement la solution correcte et que le générateur n'intervient pas.** Il faut donc expliquer ce phénomène de reconstruction avant d'attribuer les revirements à un raisonnement sur les indices.

La priorité est double : expliquer les changements de marge dans le décodeur, puis mesurer une éventuelle compensation conditionnelle par les niveaux suivants. Une recherche de têtes d'attention spécialisées vient après ces contrôles, car « attention », « VQ-VAE » et « structure du Sudoku » ne sont pas trois causes mutuellement exclusives.

## 1. Ce qui est établi, ce qui vient d'être contrôlé

### 1.1 Le générateur ne voit pas notre grille intermédiaire

Dans l'implémentation exécutée, `generate_cached` accumule les contributions latentes, prédit les codes du niveau suivant et ne retourne qu'après les douze niveaux. `SudokuModel.predict` applique ensuite le décodeur. Les décodages intermédiaires ajoutés pour l'étude sont des instruments d'observation : **aucun retour de ces argmax de grille vers les têtes de génération n'existe dans ce chemin**.

Cela exclut un mécanisme littéral « le modèle lit notre grille intermédiaire, constate une erreur, puis la corrige » dans cette implémentation. Cela n'exclut ni calcul de contraintes dans les états ni sensibilité des têtes à l'état latent courant. Les indices du puzzle restent accessibles par le préfixe à chaque niveau.

La recherche du code le plus proche d'un vecteur intervient dans l'encodage VQ. La génération native prédit, elle, des logits sur les identifiants et en prend l'argmax. Une résistance au petit bruit pendant la génération ne doit donc pas être attribuée automatiquement à une cellule de Voronoï du quantificateur de l'encodeur : les frontières pertinentes ne sont pas les mêmes.

Preuves : [boucle de génération](../vendor/ncm_mac/cached.py), [appel du décodeur après génération](../vendor/ncm_mac/sudoku.py), [quantification et résidus](../vendor/ncm_mac/_vendor/models/quant.py).

### 1.2 Nouveau contrôle : une solution connue produit aussi des revirements

Nous comparons, pour les mêmes solutions de référence :

- **Génération libre** : puzzle → générateur → codes → reconstructions cumulatives.
- **Codec oracle** : solution correcte → encodeur/quantificateur → codes → reconstructions cumulatives. « Oracle » indique l'accès à la réponse ; ce chemin n'est pas un système de résolution déployable.

Sur les **600 puzzles hard test**, parmi les cases incorrectes à 8 et correctes à 9 :

- Génération libre : **3 088/12 333** redeviennent incorrectes à 10, 11 ou 12, soit **25,04 %**, IC 95 % **[24,28 ; 25,81]**.
- Codec oracle : **3 176/12 717**, soit **24,97 %**, IC 95 % **[24,26 ; 25,73]**.

En ne conservant que les cases initialement vides, les comptes sont respectivement **1 947/7 685** et **2 030/8 012**. Le résultat ne dépend donc pas de l'inclusion des indices dans l'observable.

Le codec oracle résout 600/600 grilles au dernier niveau, mais zéro grille aux niveaux 1..11. Les exactitudes de cases oracle aux niveaux 8, 9, 11 et 12 valent respectivement **46,25 %, 63,28 %, 83,78 % et 100 %**. La reconstruction partielle reste donc loin d'une solution déjà achevée, même lorsque l'encodage utilise la bonne réponse.

Le phénomène se retrouve avec les deux autres tokenizers, sur leurs 2 000 solutions officielles : **7 306/39 926 = 18,30 %** de revirements oracle sur easy ; **11 159/40 565 = 27,51 %** sur medium. Ces difficultés partagent les familles de solutions et ne constituent pas trois réplications indépendantes des données.

**Interprétation autorisée :** le sous-système codec suffit à produire ce type de trajectoire non monotone, sans calcul de résolution par le générateur. L'attention du générateur n'est pas une condition nécessaire à l'existence du phénomène. **Interprétation non autorisée :** « 99 % des revirements générés sont causés par le codec ». Les taux voisins concernent des ensembles de cases corrigées différents ; leur ressemblance ne mesure ni une fraction causale ni une équivalence statistique. Le décodeur du codec contient lui-même de l'attention.

Méthode : réanalyse des tableaux `decoded` et `reconstructed` déjà sauvegardés dans les NPZ initiaux ; bootstrap de 2 000 tirages par puzzle, graine 202610241. La graine diffère de celle de l'étude temporelle : ses comptes sont reproduits, mais les dernières décimales des IC peuvent varier. Rejeu indépendant sur les 16 premières solutions du test : codes d'encodage et douze reconstructions exactement identiques aux traces ; zéro appel aux blocs ou têtes du générateur. **35 contrôles passent.** Ce contrôle réutilise des données examinées, sans statut confirmatoire indépendant.

### 1.3 Une précaution sur l'entraînement des lectures partielles

Il serait trop simple d'expliquer tous les préfixes médiocres par « le décodeur n'a été entraîné que sur les douze niveaux ». La configuration hard comporte `fine_drop.prob=0.5` ; le code prévoit le retrait des niveaux fins pendant l'entraînement. L'article décrit explicitement le nested dropout et une perte de reconstruction à un cutoff échantillonné (§3.3).

Cela ne prouve pas l'historique exact d'exposition des poids publiés. Mais nous devons tenir compte de ce dispositif : un éventuel défaut de supervision intermédiaire ne peut pas être affirmé sans audit supplémentaire. La supervision de reconstructions partielles ne garantit d'ailleurs pas une amélioration monotone de chaque case sur chaque exemple.

## 2. Première piste : des frontières de lecture traversées par les résidus

### Hypothèse

Les niveaux ajoutent des vecteurs à une représentation cumulative. Le décodeur transforme cette représentation en logits contextuels. Un ajout peut rapprocher l'état d'une bonne reconstruction globale tout en faisant passer certaines cases de l'autre côté d'une frontière entre chiffres. Les premiers résidus ne sont pas nécessairement des hypothèses Sudoku complètes et les ajouts ne sont pas des éditions explicites de chiffres.

Une baisse de MSE latente ne garantit même pas mathématiquement une meilleure classification. Exemple construit, distinct des résultats NCM : cible latente `(1,0)`, chiffre correct lu si la première coordonnée est positive. Passer de `(0.1,2)` à `(-0.1,0)` réduit la distance à la cible d'environ 2,19 à 1,10, tout en rendant le chiffre incorrect. Le lien MSE → exactitude de case n'est donc pas monotone par nécessité.

### Expérience discriminante

Pour un résidu natif `r_k` et une case i, fixer deux chiffres a et b et suivre la marge `m_i(z) = logit_b(D(z)) - logit_a(D(z))` le long de `z(t)=z_(k-1)+t r_k`.

1. Comparer les trajectoires codec oracle et génération libre, et les niveaux 8..12, sans privilégier 9 après examen des résultats.
2. Mesurer les traversées de zéro, leur éventuelle multiplicité, les marges initiales et finales, la NLL et la MSE vers l'encodage de la solution. Cette dernière utilise la référence comme diagnostic, pas comme entrée d'une méthode déployable.
3. Comparer le changement exact de marge à l'approximation locale `gradient(m_i) · r_k`. Une intégration des gradients le long du segment doit reconstruire le changement de marge, avec contrôle de convergence numérique.
4. Tester ensuite les composantes identifiées par suppression et restauration de leur contribution, avec contrôles de norme et composantes témoins. Les gradients seuls ne constituent pas une localisation causale.

Choisir a et b après avoir observé un changement est permis pour expliquer rétrospectivement cette transition, mais pas pour prétendre la prédire. Pour une évaluation prospective, les candidats et la sélection des cases doivent utiliser seulement les informations du niveau précédent, par exemple ses deux premiers chiffres.

**Ce qui ferait un résultat positif :** une décomposition parcimonieuse et reproductible des changements de marge, qui prédit les traversées sur de nouveaux puzzles et dont l'intervention bloque puis restaure les revirements. Dire seulement « l'argmax change quand la marge change de signe » serait une tautologie, pas une découverte mécaniste.

**Limite :** le codec oracle localise déjà le phénomène à un sous-système ; il ne départage pas encore approximation résiduelle, routage d'attention du décodeur, MLP et normalisations.

## 3. Deuxième piste, la plus importante : compensation conditionnelle ou simple tolérance ?

### Pourquoi l'expérience actuelle ne suffit pas

Le maintien de 103/128 réussites après suppression du résidu 9, contre 105/128, admet plusieurs explications : le résidu était peu utile ; le décodeur tolère son absence ; les niveaux suivants compensent activement ; ou les niveaux tardifs résolvent largement le puzzle à partir des indices, sans dépendre fortement de ce résidu.

La régénération de codes différents ne démontre pas à elle seule une compensation utile. Il faut comparer une **suite figée** à une **suite recalculée**, puis croiser les deux états et les deux suites.

### Le plan croisé à quatre conditions

Après un niveau k, appeler z l'accumulation native et δ une perturbation. La somme des contributions suivantes natives est `T0` ; celle obtenue en régénérant la suite depuis `z+δ` est `Tδ`. Les deux suites sont enregistrées explicitement, contributions résiduelles comprises.

- **Natif** : `Y00 = D(z + T0)`.
- **État perturbé, suite native figée** : `Y10 = D(z + δ + T0)`.
- **État perturbé, suite adaptée** : `Y11 = D(z + δ + Tδ)`.
- **État natif, suite adaptée à la perturbation** : `Y01 = D(z + Tδ)`.

Les états croisés sont des interventions artificielles ; leur définition explicite évite de les confondre avec des trajectoires natives. L'expérience teste la nécessité de l'adaptation au changement d'état, pas une intention de réparer une grille.

Utiliser d'abord la NLL moyenne sur les cases initialement vides, puis l'exactitude par case et par grille. Si `Lab` désigne la perte de chaque condition, le gain de régénération sous perturbation est `R = L10 - L11`. Un contraste de spécificité est `I = (L10-L11) - (L00-L01)`. Un I positif signifie que la suite adaptée bénéficie davantage à l'état perturbé qu'à l'état natif, pour cette intervention et cette perte. Ce contraste n'est ni une décomposition causale unique ni une preuve de mécanisme métacognitif ; ne pas annoncer un « pourcentage d'effet médié » sans hypothèses supplémentaires.

Compléter avec `ΔT = Tδ-T0` : orientation par rapport à `-δ`, norme de `δ+ΔT`, évolution de la distance à l'encodage de la solution. Une annulation latente et une récupération de NLL soutiendraient une compensation résiduelle ; l'une peut exister sans l'autre à cause de la géométrie du décodeur.

### Interventions et contrôles indispensables

Commencer par l'omission déjà étudiée (`δ=-r9`), puis substituer un seul code par une alternative plausible de la tête, et tester plusieurs niveaux. Ajouter des perturbations aléatoires de norme identique et des résidus témoins. Une alternative à forte probabilité n'est pas garantie sur distribution ; le signaler et contrôler les statistiques d'activations.

Ajouter une suite adaptée à une autre perturbation ou un autre puzzle, appariée autant que possible, pour vérifier la spécificité de la récupération. Conserver longueur, couches, décodeur et coût de continuation identiques. Un sham doit reproduire exactement les résultats natifs.

Prédéfinir la dose, les règles d'éligibilité et les exclusions sur développement. Ne pas retenir uniquement les puzzles où la corruption a fait chuter la réponse sur le jeu final : rapporter aussi l'effet sur tous les puzzles et toute stratification annoncée. Si les perturbations ne créent presque aucune perte, le test de récupération est peu informatif ; augmenter la dose seulement dans le pilote, avant gel.

### Lecture des résultats possibles

- Y10 se dégrade, Y11 récupère, et la récupération est spécifique à la perturbation : soutien à une compensation conditionnelle.
- Y10 et Y11 restent proches du natif : tolérance à la perturbation ou contribution dispensable ; pas de preuve d'adaptation nécessaire.
- Y10 et Y11 se dégradent pareil : pas de compensation efficace pour ce type de perturbation.
- Y11 change fortement les codes sans améliorer la perte : adaptation présente, utilité non établie.
- La récupération disparaît sans accès aux bons indices du préfixe : piste d'un recalcul ancré dans le puzzle, à distinguer d'une simple correction locale du latent.

**Contribution publiable si confirmée :** une capacité mesurée de compensation conditionnelle, avec signature latente et intervention spécifique. La quantification résiduelle existe déjà dans la littérature ; la nouveauté ne serait pas son existence, mais cette caractérisation causale sur ces modèles et tâches.

## 4. Troisième piste : où passe l'information utile ?

### Deux attentions différentes

Le décodeur Sudoku comporte un bloc Transformer avec attention et MLP ; le générateur comporte ses propres blocs. Une carte d'attention ne permet pas, à elle seule, de dire lequel produit le comportement. Les travaux sur l'attention comme explication motivent cette prudence, sans établir que toute attention serait dépourvue de rôle causal.

**Dans le décodeur**, conserver les codes et les états d'entrée fixés, puis comparer les activations de deux reconstructions voisines. Patcher séparément sorties d'attention, sorties MLP et éventuellement matrices d'attention. Si une matrice est figée, les valeurs qu'elle agrège doivent être contrôlées séparément : routage et contenu ne sont pas la même intervention. La suppression d'un module entier est un stress-test grossier ; préférer restauration, interventions réciproques et contrôles appariés.

**Dans le générateur**, une fois la compensation démontrée, patcher des sorties de blocs ou de têtes entre les calculs depuis z et z+δ. Mesurer d'abord les logits des codes suivants, puis l'effet sur la réponse finale après génération normale. Sans ce chemin via les codes, une amélioration d'un lecteur externe ne localise pas un mécanisme de génération.

Ne pas assimiler les 37 positions du niveau 9 aux 81 cases : le rééchantillonnage et l'attention rendent cette correspondance indirecte. Le cache de préfixe est lui-même contextualisé ; patcher son vecteur à une position ne revient pas à modifier uniquement le chiffre brut de cette position.

**Critère fort :** un petit ensemble de composantes localisé sur développement, dont le patch restaure spécifiquement la réponse à la perturbation sur des puzzles réservés, avec effet réciproque, contrôles de normes et mesure des dommages collatéraux. Des têtes nombreuses ou redondantes peuvent interagir : l'absence d'effet d'une ablation isolée ne prouve pas l'absence de circuit.

## 5. Quatrième piste : les contraintes Sudoku orientent-elles causalement les corrections ?

L'analyse des voisins de ligne, colonne et bloc constitue un diagnostic, pas une preuve d'algorithme. Une attention vers les pairs peut refléter position, proximité dans la sérialisation ou distribution des chiffres. Les positions témoins doivent être appariées sur ces facteurs ; lignes, colonnes et blocs se recouvrent.

Une expérience plus propre utilise **deux puzzles valides qui ne diffèrent que par un indice informatif** :

1. Construire une grille partielle B ayant exactement deux solutions S_A et S_B. Vérifier ce nombre en comptant jusqu'à trois : un résultat « deux » doit être obtenu après exploration complète, pas avec un compteur plafonné à deux.
2. Choisir une case vide j où les solutions diffèrent. Ajouter S_A[j] ou S_B[j] donne deux puzzles P_A et P_B à solution unique, de même nombre d'indices, ne différant qu'en j.
3. Choisir des cases cibles encore vides où S_A et S_B diffèrent. L'indice ajouté doit donc réellement changer la réponse correcte, sans produire un puzzle contradictoire.
4. Comparer la propagation de cet indice dans les codes et la grille finale ; patcher ensuite les composantes candidates entre les deux chemins. Rapporter la réussite native de ces nouveaux puzzles et le décalage de distribution.

Ce plan fournit un changement sémantique contrôlé, plus interprétable qu'un indice faux rendant le Sudoku insoluble. Un effet à une case distante n'établit toujours pas une règle humaine précise. Il faut montrer la spécificité du transport de l'information et reproduire après permutation des chiffres et symétries valides, avec réalignement.

Cette expérience répond aussi à l'alternative « les dernières têtes recalculent surtout depuis les indices ». Un plan croisant état latent A/B et préfixe A/B peut tester leur interaction ; ses états mixtes restent des interventions potentiellement hors distribution.

## 6. Généralisation et utilité, sans déplacer la cible

Une fois un mécanisme opérationnel défini sur Sudoku, appliquer le même protocole codec oracle puis suite figée/régénérée à un générateur texte. Les audits LM1B et TinyStories antérieurs portaient sur leurs tokenizers, **pas leurs générateurs**. Ils ne suffisent donc pas à établir une compensation temporelle en texte.

Sur le texte, remplacer les métriques de validité Sudoku par reconstruction, NLL et stabilité du token ; des changements de token ne sont pas nécessairement des corrections sémantiques. Une généralité architecturale demanderait plusieurs familles de données et, pour une attribution causale à l'architecture plutôt qu'aux poids, des comparaisons d'architectures/entraînements contrôlés. Deux checkpoints texte ne prouveraient pas l'universalité.

Une solution pratique peut découler du mécanisme : supprimer certaines prédictions puis laisser la suite compenser, ou réserver le calcul aux exemples où une règle d'arrêt observable est satisfaite. Elle doit être sélectionnée sur développement, évaluée pour la fidélité et la qualité sur de nouvelles données et chronométrée avec un protocole apparié. **103/128 contre 105/128 ne prouve pas une équivalence**, et l'omission actuelle exécute encore la tête ciblée. Une éventuelle économie de calcul exige une implémentation qui évite réellement cette prédiction. Ni l'entropie ni un score utilisant la référence ne constituent une règle d'arrêt déployable.

## 7. Ordre d'exécution recommandé et validité

1. **Acquis dans cette note :** audit du chemin sans retour de la grille, réanalyse codec oracle/génération et rejeu oracle sur 16 solutions. Ne pas présenter cette exploration comme une confirmation indépendante.
2. **Priorité suivante :** plan croisé suite figée/régénérée et contrôles de perturbation. C'est le contraste qui manque pour nommer une compensation adaptative.
3. **En parallèle conceptuel :** géométrie des marges dans le décodeur. Elle explique le phénomène de lecture ; ne pas la substituer au test causal sur le générateur.
4. **Seulement si un effet utile existe :** patching des modules puis têtes, et paires de puzzles à indice disambiguïsant.
5. **Après validation :** réplication texte et éventuel raccourci de calcul mesuré.

Les données officielles et l'ancien corpus externe sont déjà largement explorés. Utiliser train pour la recherche, validation pour les choix, puis générer et figer de nouvelles familles de solutions avant les nouveaux tests. Vérifier format, respect des indices, unicité ou nombre de solutions requis et recouvrements détectables. L'absence de recouvrement avec les corpus publiés n'exclut pas une présence dans l'entraînement inconnu.

Un pilote de quelques dizaines de puzzles peut fixer doses et variances ; la taille de confirmation doit ensuite être justifiée par l'effet minimal intéressant et une analyse de puissance ou de précision. Aucun effectif magique ne garantit une preuve. Annoncer un contraste principal, les marges de non-infériorité éventuelles, les contrôles négatifs et les exclusions avant le jeu final. Bootstrap par famille source ; toutes les versions d'un puzzle et les graines restent ensemble. Pour comparer les ajustements d'une sonde, inclure aussi la variation des graines/découpages si la conclusion dépend de l'apprentissage de cette sonde.

Séparer recherche exploratoire de têtes et test réservé ; corriger la multiplicité des affirmations confirmatoires. Contrôler la parité native, les caches, le chemin via les codes, la norme des perturbations, les états hors distribution et les effets collatéraux. Un contrôle aléatoire faible ne suffit pas à valider une intervention ciblée de forte amplitude.

## 8. Formulation scientifique et apport possible au papier

**Déjà défendable :** « Les trajectoires de révision non monotones persistent lors de l'encodage de solutions correctes, en l'absence de génération. Elles ne suffisent donc pas à identifier une décision de résolution ; leur origine doit être recherchée au moins dans le codec progressif. » C'est une localisation expérimentale positive, encore à l'échelle d'un sous-système.

**À tester :** « Les niveaux fins produisent des contributions dépendantes de l'erreur qui compensent spécifiquement une perturbation du latent, plutôt que de seulement la tolérer. » Cette formulation peut être vraie ou fausse ; le plan croisé permet de le savoir.

**Seulement après localisation causale :** « Telles composantes transportent telle information de contrainte et commandent telle modification des codes. » Une belle carte d'attention, une sonde prédictive ou une annulation de norme ne suffisent pas à cette affirmation.

Un titre provisoire honnête serait *Apparent Answer Revision in NCM: Codec Dynamics and Conditional Residual Compensation?* Le point d'interrogation doit disparaître seulement si l'expérience correspondante confirme la compensation ; sinon le papier doit rester centré sur les dynamiques du codec. La validité ne peut pas être garantie absolument : on peut garantir la traçabilité des mesures, l'explicitation des hypothèses et la capacité du protocole à contredire notre explication favorite.

## 9. Sources et reproduction

Le code de cette réanalyse est [scripts/audit.py](scripts/audit.py). Les mesures, entrées épinglées et contrôles sont dans `results/audit.json`, `results/input_manifest.json` et `results/verification.json`. Les tableaux par niveau et les contre-exemples sont conservés pour les deux chemins. Le calcul oracle original se trouve dans [extract.py](../scripts/extract.py). Les poids restent en lecture dans le parent.

Pour rejouer, depuis ce dossier :

```sh
PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ../.venv/bin/python scripts/audit.py
```

Références primaires consultées :

- [Article NCM, §3.3–3.4](https://arxiv.org/html/2610.14162v1) ; copie locale [paper.txt](../sources/paper.txt), notamment la description du nested dropout. Le code et les poids publiés priment pour décrire le chemin réellement exécuté ici.
- [Lee et al., CVPR 2022, Autoregressive Image Generation using Residual Quantization](https://arxiv.org/abs/2203.01941) : antécédent d'une génération de représentations par quantification résiduelle ; cette idée générale ne serait pas une nouveauté de notre étude.
- [Zhang et Nanda, ICLR 2024, Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042) : le choix de corruption, de métrique et de patch peut modifier les conclusions de localisation ; d'où les contrôles croisés et les mesures de logit/perte proposés ici.
- [Jain et Wallace, NAACL 2019, Attention is not Explanation](https://arxiv.org/abs/1902.10186) : avertissement empirique contre l'identification directe des poids d'attention à une explication. Cela ne remplace pas les interventions sur le présent NCM.

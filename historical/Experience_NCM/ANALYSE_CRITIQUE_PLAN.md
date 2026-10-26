# Réponse à la critique du plan — 23 octobre 2026

**Verdict : les réserves principales sont justifiées pour passer du cadrage à un protocole exécutable. Le plan était déjà explicitement exploratoire ; la critique ne démontre pas, à elle seule, une puissance insuffisante. Une de ses conclusions sur le vérificateur est trop restrictive.** Les corrections sont intégrées au [plan v0.2](PLAN_RECHERCHE_METACOGNITION.md) et à sa [spécification](experiments/metacognition/protocol-v0.2.json).

## 1. Puissance : réserve valide, diagnostic à chiffrer

68–69 échecs sur 500 puzzles ne déterminent pas seuls la puissance d'une différence appariée d'AUROC. Il faut connaître la performance de référence et la dépendance des scores. Cette dépendance est explicitement nécessaire au dimensionnement apparié. [Hwang et al., 2022](https://pubmed.ncbi.nlm.nih.gov/35429054/).

Une simulation synthétique a été exécutée : **72 scénarios, 1 000 répétitions chacun**, sans donnée ni exécution NCM. Modèle binormal à variance égale, taux d'échec fixé à 13,7 %, intervalles normaux par variance de placement de DeLong. Pour n=500 et gain réel de 0,03, la probabilité de détecter un gain positif varie de **13,2 % à 93,9 %**. À AUROC de référence 0,85 : 19,6 %, 30,4 % et 93,9 % pour des corrélations des scores au sein des classes de 0,30, 0,70 et 0,95. L'erreur Monte-Carlo de ces proportions est au plus d'environ 1,6 point de pourcentage (un écart-type). Les rejets bilatéraux sous effet nul vont de 3,8 % à 6,7 % selon les scénarios ; ce contrôle ne valide pas toutes les distributions possibles.

**Limites :** ni ajustement des sondes ni calibration simulés ; les intervalles du protocole réel sont bootstrap, pas ceux de cette simulation ; aucune puissance réelle du NCM n'est estimée. L'analyse complète du réentraînement reste à faire sur le développement, sans test. [Script](experiments/metacognition/simulate_auc_design.py), [résultats](audit/metacognition/design-power-sensitivity.json).

Un défaut supplémentaire du plan initial est corrigé : la règle « estimation≥0,03 ET intervalle>0 » n'a pas une puissance de 80 % au gain vrai de 0,03, même si le test de gain positif devient puissant. Le filtre sur l'estimation passe environ une fois sur deux à cette frontière. Nous séparons donc discrimination positive, importance pratique et preuve éventuelle d'un gain minimal.

## 2. Niveau 9 : convention initiale, pas optimum démontré

La critique est correcte : le niveau avait été choisi sans justification détaillée. Le remplacer par « là où la performance est variable » nécessiterait d'observer des résultats et un algorithme de sélection défini sur le développement. Il n'est pas nécessaire de choisir un niveau optimal pour un pilote.

**Décision : conserver 9 avant toute trajectoire**, avec trois niveaux disponibles pour corriger et 37 codes au niveau courant ; justification de design ajoutée explicitement après le choix initial. Les longueurs de niveaux sont variables : 116 codes cumulés sur 322, et non 75 % du coût connu. Les niveaux 3/6/11 restent secondaires ; aucun remplacement du critère principal après observation.

## 3. Sonde : sous-spécification reconnue et corrigée

Le code local permet de fixer le prélèvement : sortie du huitième bloc Transformer (`blocks[7]`), avant normalisation finale, uniquement sur les positions cibles de l'appel du niveau courant. Moyenne et écart-type population par canal donnent 1 024 valeurs, réduites à 16 composantes PCA ajustées seulement sur apprentissage. Le préremplissage du préfixe est exclu.

Famille, objectif L2, grille de six régularisations, cinq plis groupés, critère de sélection, standardisation, tolérances, graines et calibration sont maintenant consignés. Il s'agit d'une spécification à implémenter, pas d'une sonde déjà entraînée ou d'un pré-enregistrement public. Les contrôles de capacité sont importants en probing. [Hewitt et Liang, 2019](https://aclanthology.org/D19-1275/).

## 4. Références : renforcer sans promettre une complétude absolue

Réserve fondée. La référence principale est enrichie des probabilités complètes du décodeur courant, des probabilités de codes disponibles et de l'historique décodé, en plus des résumés et du puzzle. La même famille et le même budget d'hyperparamètres sont employés avec et sans les 16 variables internes ; un contrôle ajoute 16 variables aléatoires.

Cette référence est riche, mais pas universelle ; sa forte dimension peut aussi nuire à l'apprentissage et doit être analysée sur développement. Plus fondamentalement, les états déterministes sont une fonction du puzzle : notre test porte sur l'**accessibilité prédictive avec une sonde limitée**, pas sur une information théoriquement absente de l'entrée.

## 5. Perturbations : oui aux comparateurs, prudence sur l'appariement

La norme et l'entropie ne suffisent pas à rendre deux perturbations fonctionnellement équivalentes. Prévoir des codes plausibles, des remplacements de même codebook, un passage fictif sans modification et des directions aléatoires appariées en norme ; séparer sensibilité à une anomalie et prédiction de fiabilité.

L'entropie pertinente pour l'appariement est celle **avant intervention** ; apparier systématiquement l'entropie après intervention pourrait conditionner sur une conséquence du mécanisme étudié. La perturbation aléatoire n'est pas un placebo supposé sans effet. Toutes les paires sont choisies sans réussite future ; le transfert de la sonde gelée reste séparé du réentraînement.

## 6. Politique : utilité mieux définie, conclusion du critique à nuancer

Le vérificateur détectait tous les échecs finaux archivés ; cela limite fortement l'intérêt d'une simple abstention finale apprise. Mais il **ne produit pas de nouvelle solution**. Une exploration ciblée peut augmenter la résolution si elle trouve des candidats corrects : ce n'est pas impossible face au vérificateur, mais à comparer à une recherche simple utilisant le même vérificateur au même budget.

L'utilité principale devient le **coût à taux de résolution vérifiée et couverture comparables**, avec coût de toutes les vérifications et recherches inclus. À candidat inchangé, aucune sélection ne crée un nouveau succès. Les preuves sur corpus unique ne valent pas certification universelle du vérificateur.

## 7. Métacognition native et confirmation : réserves déjà prévues, engagements renforcés

Le plan initial ne promettait ni résultat positif, ni métacognition native, ni test définitif avec 500 puzzles. La conclusion « information décodable, contrôle natif non démontré » reste explicitement recevable. Une correction ou un contrôle ajouté ne prouve pas un mécanisme natif.

Le corpus de confirmation est maintenant **conçu dès la phase 1** : générateur indépendant, unicité vérifiée, provenance/graines et sélection gelées avant mesure NCM. Sa taille reste conditionnée au dimensionnement et au budget, et le corpus reste à construire. Son changement de distribution est une limite à rapporter. Aucune affirmation prématurée de protocole confirmatoire suffisamment puissant.

**Travail réalisé :** analyse de la critique, inspection du code, simulation statistique synthétique et révision du plan/spécification. **Travail restant :** implémentation et vérification de la spécification, partitions, pilote technique, étude de dimensionnement avec réentraînement et préparation du corpus indépendant. Aucun poids modifié, aucune trajectoire NCM extraite et aucune sonde entraînée.

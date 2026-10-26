# Corriger le pooling : preuve de concept sans réentraînement

## But

Tester une modification explicite qui réintroduit une sensibilité à l'ordre au niveau 5, puis mesurer son passage à travers le codebook gelé et son coût en reconstruction. Distinguer ces résultats d'une amélioration du générateur ou de la robustesse.

Une variance plus élevée ne suffit pas. Un biais déterministe indépendant de l'entrée ne peut pas augmenter sa variance inter-puzzles en arithmétique exacte ; il constitue un contrôle négatif. Un max-pooling de seuls embeddings de tokens reste invariant à la permutation ; les positions et résidus peuvent modifier ce constat. Un séparateur ajouté identiquement dans chaque fenêtre ne suffit pas non plus, sauf s'il change les frontières ou la répartition des tokens.

## Modification principale

Pour les mêmes fenêtres adaptatives B_j, remplacer la moyenne u_j par u_j + alpha × moyenne(t_ji × r_i), où t varie linéairement de -1 à +1 dans chaque fenêtre et r est le résidu avant quantification. Les poids deviennent (1+alpha t)/|B_j| ; leur somme vaut 1. Ils sont positifs ou nuls pour alpha≤1 et peuvent être négatifs pour alpha=2. Un échange de deux tokens change le terme pondéré lorsque leurs poids et embeddings diffèrent. Aucune nouvelle dimension ni codebook n'est ajouté.

Comparer aussi un décalage d'un token des fenêtres (bord droit répliqué), un mélange moyenne/max adaptatif et un biais fixe pseudo-aléatoire de RMS 0,25. La variante alpha=0 est un sham exact. Les paramètres et critères sont dans protocol.json, gelés avant les calculs.

## Évaluation

1. Vérifier l'accumulation explicite contre encode/decode natifs ; poids et EMA restent gelés. Mesurer variance sur embeddings bruts et résidus, sensibilité à un échange intra-ligne, entropie conditionnelle à la position, codes modifiés et reconstruction stricte.
2. Sur les 128 premières solutions hard validation, tester la grille finie. Sélectionner selon le critère du JSON, sans ajustement sur test. Si aucune variante ne passe, conserver le meilleur diagnostic et ne pas l'appeler correction gratuite.
3. Sur 600 solutions test hard et 256 externes : native, sélection, weighted_1 et max_1. Sur 600 solutions easy et medium : native et sélection, avec des poids distincts mais des familles de solutions partagées. La référence est la solution connue : il s'agit de reconstruction, jamais d'une résolution à partir des indices.
4. Sur 128 solutions test hard, permutation valide des chiffres et transposition ; évaluer séparément les entrées transformées. Sur les mêmes 128 sources, bruit gaussien dans l'état de l'encodeur avant quantification, cinq graines et trois amplitudes relatives au RMS train. Les perturbations sont identiques entre variantes. Bootstrap par puzzle, en agrégeant les graines ; pas par token ni par bruit supposé indépendant. Cela mesure une robustesse du codec au bruit latent, pas la difficulté Sudoku.
5. Après patch de la méthode native de pooling, vérifier encode et la génération normale sur huit puzzles. Comparer les codes générés aux nouvelles cibles d'encodage. Si le générateur n'appelle pas le pooling modifié, une génération inchangée n'est pas une amélioration de la résolution et n'atteste pas l'apprentissage des nouveaux codes.

## Livraison

Rapport complet, extension anglaise et article intégré copiés dans ce nouveau dossier, sources, prédictions et codes par puzzle, figures, empreintes et vérifications. Les rapports et manifestes précédents restent inchangés. Pas de réentraînement, de recherche supplémentaire d'hyperparamètres après test ni de conclusion de robustesse sur la seule base de la reconstruction propre.

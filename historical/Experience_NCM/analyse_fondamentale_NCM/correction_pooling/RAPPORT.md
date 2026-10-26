# Correction du pooling sans réentraînement

L'[article intégré](PAPER.md), son [PDF](PAPER.pdf) et le [supplément expérimental anglais](SUPPLEMENT.md) ajoutent une proposition explicite et ses tests. Les versions antérieures restent intactes.

## Ce qui est corrigé

La pondération dépendant de la position casse l'invariance de la moyenne aux permutations dans une fenêtre, à taille de sortie inchangée. Aucun poids, codebook ni décodeur n'est réentraîné. Le cas alpha=1 utilise des poids non négatifs de somme 1 ; alpha=2 est seulement un diagnostic avec poids éventuellement négatifs. Un biais fixe ne peut pas ajouter de variance entre entrées ; il est testé comme contrôle négatif.

La sélection sur 128 solutions validation retient **weighted_1** : 128/128 reconstructions exactes et 16/16 codes du niveau 5. Le sham reproduit les codes natifs.

## Résultats réservés

- **hard_test** : natif 600/600, modification 600/600 ; 0 échecs introduits, 0 corrigés ; 15/16 codes, entropie conditionnelle 2.5887 nat.
- **external** : natif 256/256, modification 256/256 ; 0 échecs introduits, 0 corrigés ; 15/16 codes, entropie conditionnelle 2.5773 nat.
- **easy_test** : natif 600/600, modification 600/600 ; 0 échecs introduits, 0 corrigés ; 15/16 codes, entropie conditionnelle 2.5555 nat.
- **medium_test** : natif 599/600, modification 599/600 ; 0 échecs introduits, 0 corrigés ; 15/16 codes, entropie conditionnelle 2.5369 nat.

Les reconstructions utilisent des solutions connues, pas seulement les indices. Un résultat propre identique ne prouve pas une équivalence populationnelle ni une amélioration de résolution. Les réplications easy/medium partagent les familles de solutions.

## Robustesse mesurée

- **digits**, 128 solutions transformées : natif 128/128, modification 128/128.
- **transpose**, 128 solutions transformées : natif 128/128, modification 128/128.
- **Bruit latent sigma=0.05 × RMS train**, cinq graines sur 128 solutions : natif 100.00%, modification 100.00% ; variation 0.00 points, IC 95 % [0.00, 0.00].
- **Bruit latent sigma=0.2 × RMS train**, cinq graines sur 128 solutions : natif 100.00%, modification 100.00% ; variation 0.00 points, IC 95 % [0.00, 0.00].
- **Bruit latent sigma=0.5 × RMS train**, cinq graines sur 128 solutions : natif 100.00%, modification 100.00% ; variation 0.00 points, IC 95 % [0.00, 0.00].

Ce stress-test porte sur le codec sous bruit latent. Il ne mesure ni résistance à des indices faux ni robustesse du générateur, et ses trois contrastes sont exploratoires.

Les deux codecs restent au plafond sur les trois amplitudes : aucun avantage de robustesse n'est démontré. Le bruit agit pourtant sur les codes : à sigma=0,5, 35431 codes natifs et 38521 codes modifiés diffèrent de leurs versions propres, sur l'ensemble des cinq graines.

Un contrôle post-hoc remplace seulement les neuf codes du niveau 5 en gardant les autres codes natifs : 128/128 reconstructions exactes, contre 128/128 lorsque toute la solution est réencodée. Les niveaux fins changent alors de 6039 codes. La lecture partielle après niveau 5 atteint 18.31% de cases exactes contre 21.19% nativement ; ce n'est pas une résolution à partir des indices. Le contrôle et sa motivation sont dans [AMENDEMENT_COMPENSATION.md](AMENDEMENT_COMPENSATION.md).

## La limite de la « solution gratuite »

Le pooling corrigé est appelé pendant encode, mais **jamais pendant les huit générations natives contrôlées**. Les codes, logits et réponses générés restent identiques. Le patch répare donc un opérateur d'encodage, sans modifier ce chemin de résolution. Le taux d'accord des codes générés du niveau 5 avec les cibles d'encodage passe de 100.00% à 1.65%. Le générateur gelé n'a pas appris ces nouvelles cibles.

Il faut donc distinguer correction mathématique, reconstruction propre, robustesse du codec et amélioration du NCM complet. Nous fournissons les quatre contrôles sans déduire les derniers des premiers. Un futur apprentissage conjoint de la pondération et des cibles est une piste motivée, pas un résultat accompli ici.

## Sources et vérification

`sources_correction.zip` contient les scripts utilisés et les sources des modèles. Les tableaux détaillés et les prédictions par solution sont dans `results/`. Les figures sont en PNG/SVG/PDF. Le [protocole](OBJECTIFS_ET_PROTOCOLE.md), le [journal](JOURNAL.md), `results/verification.json` et `results/final_manifest.json` décrivent la méthode et les contrôles passés. Depuis ce dossier : `./run.sh` pour reproduire la correction ; le runner de la campagne texte reste dans le dossier précédent.

# Codec, compensation conditionnelle et générateur texte

Protocole rédigé avant les calculs de cette campagne. Les données anciennes ont déjà été examinées ; les nouvelles expériences sur ces corpus restent exploratoires. Les nouveaux puzzles seront générés après ce gel, sans choix de dose ou de niveau selon leurs résultats.

## Plan croisé

Checkpoint Sudoku hard gelé. Après niveau 9 natif, z=z9 et δ=-r9. T0 est la somme native des résidus 10..12, Tδ la somme obtenue après régénération de 10..12 depuis z+δ. Définir Y00=D(z+T0), Y10=D(z+δ+T0), Y11=D(z+δ+Tδ), Y01=D(z+Tδ). Les croisements sont des interventions artificielles, potentiellement hors distribution. Les poids et le décodeur sont identiques.

Accumuler les résidus dans l'ordre natif et vérifier le natif bit à bit ; les expressions réassociées peuvent différer par arrondis, à quantifier. Sauvegarder logits finaux, codes des deux suites, pertes et géométrie par puzzle. Sham, nettoyage du cache et parité sur les données antérieures sont requis.

Deux contrastes co-primaires sur la NLL moyenne des cases initialement vides : récupération R=L10-L11 ; interaction I=L10+L01-L11-L00. Bootstrap apparié par puzzle, 4 000 tirages. IC descriptifs à 95 % et IC à 97,5 % pour contrôle Bonferroni des deux contrastes annoncés. Une conclusion de compensation conditionnelle exige les deux bornes basses à 97,5 % positives sur les 256 nouveaux puzzles. Ce critère teste cette intervention, sans prouver une correction consciente ni tout mécanisme possible de révision.

Les 600 hard test et 256 externes anciens servent de réplications exploratoires. Générer 256 nouvelles solutions par backtracking aléatoire, puis retirer des indices en préservant une solution unique jusqu'à 30 indices. Vérifier par MRV et Algorithm X. Exclure les clés D4/chiffres des solutions officielles et externes antérieures ; ne pas prétendre exclure toutes les isomorphies ou une contamination d'entraînement. L'effectif est un budget de précision fixé, sans prétention de puissance garantie.

Mesurer aussi réussite exacte, codes modifiés, alignement de ΔT=Tδ-T0 avec -δ, ratio ||δ+ΔT||/||δ||, distance à l'encodage de la solution. Contrôle de lecture par direction gaussienne indépendante de même norme que ΔT, pour distinguer une adaptation orientée d'un déplacement quelconque. Ce contrôle reste descriptif ; les normes identiques n'apparient pas toute la géométrie.

## LM1B small

Télécharger le générateur publié épinglé, réutiliser le tokenizer et les sources officielles déjà copiés. Exécuter le générateur natif eager sans optimisation personnalisée, sur CPU FP32, 128 trajectoires indépendamment ensemencées. Même température/top-k/top-p que la configuration publiée et premier code tiré de son unigramme sauvegardé ; préfixe CLS=101. Reconstituer explicitement chaque accumulation pour éviter l'aliasing connu des historiques de idx_to_zhats ; vérifier le dernier état contre l'implémentation native.

Sans réponse vraie en génération inconditionnelle, un token changé n'est pas une erreur corrigée. Observable rétrospectif : adoption du token raw final puis abandon à un niveau intermédiaire ultérieur. Rapporter chaque transition et les changements sans ancre. Le niveau 9 LM1B ne représente pas la même résolution que le niveau 9 Sudoku. Les 128 positions fixes forment l'analyse principale ; un masque final non spécial est secondaire et conditionne sur le futur. Les heuristiques finales fréquence/adjacence sont séparées de la lecture argmax brute.

Contrôle codec oracle sur 128 phrases validation déjà sélectionnées : encodeur seul, reconstructions progressives, références non-PAD ; c'est un diagnostic de reconstruction, distinct des textes générés. Pas de nouvelle mesure GPT2 de perplexité ni de revendication de qualité sémantique.

## Rapport et portée

Titre visé : Apparent Answer Revision in NCM: Codec Dynamics and Conditional Residual Compensation. Le titre et la conclusion seront ajustés si la compensation n'est pas établie. La présence du phénomène sur Sudoku et LM1B étend sa portée au-delà de Sudoku, sans prouver une propriété universelle du NCM ni une attribution exclusive au codec.

Le patching d'attention et les puzzles à indice disambiguïsant sont explicitement hors périmètre. Fournir rapport français, article anglais/PDF, codes, traces, archive de sources et vérification d'intégrité. Toutes les écritures restent dans ce nouveau dossier.

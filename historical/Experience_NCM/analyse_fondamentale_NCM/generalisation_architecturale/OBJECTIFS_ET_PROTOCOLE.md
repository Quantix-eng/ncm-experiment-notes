# Généralisation de l'invariance : protocole exploratoire v1

## Question et portée

Distinguer un effet des lignes Sudoku, une vulnérabilité mathématique du pooling et un défaut empirique partagé par les tokenizers NCM texte. Aucun résultat ne rendra un article « incontestable » ; les conclusions seront limitées aux checkpoints, entrées et interventions effectivement examinés.

LM1B small/medium/large partagent un seul tokenizer publié : ce sont un seul cas pour cet audit. Ajouter TinyStories donne un second tokenizer texte, avec architecture et données différentes. Les générateurs texte ne sont pas nécessaires à l'encodage et ne seront pas évalués. Leurs performances de génération ne pourront pas être déduites des présentes mesures.

## Texte

- Télécharger le tokenizer LM1B et sa configuration officielle épinglés à un commit Hugging Face. Copier le tokenizer TinyStories local dans ce dossier et vérifier son empreinte contre les métadonnées publiées si disponibles. Sources du modèle copiées sans modification.
- Corpus : validation officielle tokenisée de chaque tâche. Tirer 256 exemples représentatifs et 128 exemples d'au moins 96 tokens hors padding, si disponibles. Une sélection longue peut recouvrir l'échantillon représentatif ; ce sont des strates descriptives, jamais deux réplications indépendantes.
- Vérifier format, vocabulaire, longueurs, padding et répétitions. Conserver indices et tokens exacts. Interpréter séparément les positions dominées par le padding. Les tokens spéciaux autres que PAD restent comptés dans la reconstruction.
- Auditer les 12 niveaux : histogrammes, entropie marginale et conditionnelle à la position, part du code modal, variété de séquences de codes, variance du résidu et du signal après sous-échantillonnage, erreurs de quantification, diminution de MSE résiduelle. Un code absent d'un petit corpus n'est pas un code définitivement mort.
- Comparer l'accumulation explicite au chemin encode officiel sur 8 exemples, et les distances rapides à des distances directes FP64 sur un petit échantillon. Vérifier poids/EMA inchangés.
- Sur 32 exemples : permutation dans chaque bloc de quatre tokens, puis permutation globale. Mesurer invariance des moyennes d'embeddings à 32 positions et modification des codes natifs. Les entrées permutées sont diagnostiques, pas un corpus naturel de qualité linguistique.
- Distinguer encodeur additif, positions, GroupNorm, premier sous-échantillonneur appris et contributions résiduelles précédentes. Une invariance du pooling pris isolément ne prouve pas celle du tokenizer complet.

## Tous les niveaux Sudoku

- Réutiliser en lecture les codes encodés/générés et diagnostics des campagnes précédentes ; aucune réinterprétation de leurs tests comme données nouvelles.
- Déterminer un template modal par position à chaque niveau sur les seuls 1 000 puzzles train hard. Sur les 96 premiers puzzles validation, tester séparément chaque niveau remplacé par ce template, contribution conservée et suite régénérée ; tester aussi chaque contribution supprimée avec prédiction native.
- Classer les remplacements par nombre de statuts succès/échec modifiés, puis de grilles modifiées, puis numéro du niveau. Geler les trois premiers avant test.
- Sur les 600 puzzles test et 256 externes : tester séparément les trois niveaux retenus, les deux premiers ensemble (10 prédictions), et les trois ensemble (9 prédictions). Rapporter réussite, discordances, codes, erreurs corrigées/introduites, IC bootstrap appariés et bornes Wilson. Aucun critère d'équivalence de performance n'est préspécifié : un IC recouvrant zéro ne démontrera pas l'équivalence.
- Les calculs de suppression et de remplacement répondent à des questions différentes. Une réduction du nombre de prédictions n'est pas une mesure de vitesse.

## Livraison

Article en anglais, rapport de lecture en français, preuves et limites, figures PNG/SVG, résultats par exemple/puzzle, tous les scripts et sources de modèle, empreintes, commande de reproduction et contrôles. Écritures uniquement dans ce nouveau sous-dossier de `analyse_fondamentale_NCM`. Les campagnes précédentes restent intactes.

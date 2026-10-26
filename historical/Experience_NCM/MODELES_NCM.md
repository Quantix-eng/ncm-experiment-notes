# Modèles NCM publics — inventaire du 22 octobre 2026

Périmètre : **Next-Concept Modeling de HazyResearch**, génération multiscalaire par codes VQ. Les Large Concept Models à représentations de phrases et les autres architectures de next-concept prediction constituent des familles distinctes ; ils ne sont pas assimilés à ce NCM.

## Catalogue officiel

**Neuf checkpoints complets disponibles, plus OpenWebText annoncé mais non livré.** Tous les noms et statuts ci-dessous proviennent du [dépôt officiel des poids](https://huggingface.co/hazyresearch/ncm-checkpoints/tree/d342ce8aaea58830eef3b8ad473664673e41f692), révision `d342ce8aaea58830eef3b8ad473664673e41f692`, encore courante lors de la vérification. Le code et la fiche du dépôt de poids indiquent MIT.

M = million de paramètres, arrondi à deux décimales. **Total = générateur + VQ-VAE complet** (encodeur, quantificateur et décodeur), sans compter les buffers. Ce n'est pas une mesure de RAM ni le nombre de paramètres activés dans chaque passe.

| Modèle | Tâche | Générateur (M) | VQ-VAE (M) | Total (M) | Niveaux | Poids disponibles |
|---|---|---:|---:|---:|---:|---|
| Sudoku easy | Résolution, 40 indices | 25,46 | 14,90 | 40,35 | 12 | Oui |
| Sudoku medium | Résolution, 35 indices | 25,46 | 14,90 | 40,35 | 12 | Oui |
| Sudoku hard | Résolution, 30 indices | 25,46 | 14,90 | 40,35 | 12 | Oui |
| TinyStories | Génération d'histoires courtes | 88,61 | 36,43 | 125,04 | 12 | Oui |
| XSum | Résumé d'articles | 88,02 | 60,63 | 148,65 | 9 | Oui |
| WMT14 De→En | Traduction allemand→anglais | 119,96 | 144,00 | 263,97 | 11 | Oui |
| LM1B small | Génération de texte non conditionnée | 97,07 | 111,40 | 208,47 | 12 | Oui |
| LM1B medium | Même tâche | 316,52 | 111,40 | 427,93 | 12 | Oui |
| LM1B large | Même tâche | 724,80 | 111,40 | 836,21 | 12 | Oui |
| OpenWebText / OWT | Génération de texte non conditionnée | Non confirmé ici | Non confirmé ici | Non confirmé ici | — | Non : `owt/TODO.md` seulement |

Les trois Sudoku sont des checkpoints distincts de même architecture. Les trois LM1B partagent le même fichier `lm1b/tokenizer.pt` ; la colonne VQ-VAE ne représente donc pas trois tokenizers différents. Les configurations ne décrivent pas des assistants conversationnels instruction-tuned.

**Méthode de comptage :** configurations officielles courantes téléchargées à révision fixe ; instanciation des architectures sur le dispositif PyTorch `meta`, sans allocation des poids ; somme de `numel()` des paramètres uniques par composant. Utilisation du code local dérivé du commit amont `c70274273f489abc73832b2b98584a8483b6b203`. Les nombres décrivent les architectures configurées ; aucune nouvelle validation du chargement des neuf checkpoints n'est revendiquée. Les comptages Sudoku hard/TinyStories concordent avec les chargements locaux antérieurs. Aucun poids supplémentaire téléchargé et aucune nouvelle inférence.

Reproduction : `.venv/bin/python scripts/inventory_ncm_models.py` (accès réseau requis). Résultats exacts, métadonnées, liste des fichiers et empreintes des configurations : [audit/ncm-model-inventory.json](audit/ncm-model-inventory.json). La commande consulte la révision courante : utiliser celle consignée ci-dessus pour retrouver exactement cet état.

## Autres publications repérées et limites d'exhaustivité

Un dépôt indépendant [binxx71/aift-ncm-deepseek-architecture](https://huggingface.co/binxx71/aift-ncm-deepseek-architecture) revendique une combinaison AIFT/NCM/MLA/GRPO. La liste consultée ne contient qu'un README, `.gitattributes` et un fichier `.pt` de 1,55 GB. **Nombre de paramètres non documenté et non vérifié**, pas de configuration ni de script d'inférence séparés dans ce dépôt. Aucun fichier de poids téléchargé ou exécuté. La taille du fichier ne suffit pas à déduire le nombre de paramètres. Cette publication n'est pas comptée parmi les neuf modèles officiels ni considérée ici comme une base reproductible validée.

La recherche ciblée GitHub/Hugging Face a identifié cette publication supplémentaire, sans établir un inventaire exhaustif de tous les forks ou dépôts publics. « Tous les modèles » désigne ici le catalogue officiel complet vérifié, complété par ce dépôt tiers signalé. Les ablations de l'article ne sont pas automatiquement des checkpoints publiés. [Code et catalogue des auteurs](https://github.com/HazyResearch/ncm), [article](https://arxiv.org/html/2610.14162v1).

## Intérêt pour notre étude

- **Sudoku hard :** conserver le pilote actuel, tâche vérifiable et inférence CPU déjà validée.
- **LM1B small/medium/large :** meilleur candidat parmi ce catalogue pour examiner une variation de taille à tokenizer commun. La capacité n'est pas le seul facteur possible ; contrôler aussi données, budgets et procédures d'entraînement avant toute conclusion causale sur la taille. Il faudra définir une cible d'évaluation adaptée au texte.
- **XSum :** candidat pour étudier fidélité au document et incertitude conditionnelle ; une évaluation de factualité est nécessaire, ROUGE seul ne suffit pas pour cette cible.

Le modèle officiel le plus grand disponible ici atteint **836,21 M de paramètres au total**, dont **724,80 M pour le générateur**. Aucun checkpoint officiel de plusieurs milliards n'apparaît dans le catalogue vérifié. Sa faisabilité d'inférence et surtout d'entraînement sur notre Mac 8 Go reste à mesurer ; taille des paramètres et mémoire totale d'exécution ne se confondent pas.

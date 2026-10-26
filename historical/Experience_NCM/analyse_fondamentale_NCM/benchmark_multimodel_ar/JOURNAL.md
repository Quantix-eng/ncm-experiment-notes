# Journal de l'extension autorégressive

25 octobre 2026. Objectif : étendre le protocole à un autre système appris, en traitant les signatures proposées par architecture comme des hypothèses et non comme des étiquettes connues.

## Préparation avant sorties

1. Inventaire local : trois checkpoints Sudoku de même architecture NCM, aucun modèle AR prêt à l'inférence dans le projet. Choix de GPT-2 small pour ajouter une architecture différente ; la tâche textuelle sera donc un facteur de variation supplémentaire.
2. Consultation ciblée de la fiche et de la documentation officielles GPT-2, puis lecture des blocs exécutés dans Transformers. Téléchargement de fichiers à la révision HF `607a30d783dfa663caf39e06633721c8d4cfcd7e`, sans exécuter le modèle. Les métadonnées et poids sont conservés dans `artifacts/gpt2`.
3. Installation de Transformers 4.57.1 et safetensors 0.6.2 et de leurs dépendances dans le sous-dossier local `deps`. Aucun paquet des environnements précédents n'est remplacé. Le bootstrap utilise NumPy/PyTorch existants avant de charger les outils HF locaux ; les versions réellement importées sont enregistrées dans le gel.
4. Choix des 16 paires de mots et des trois modèles de prompts. `prepare.py` vérifie seulement le tokenizer : 8 positions par prompt, cibles d'un token, une seule différence A/B, token final commun. Aucun filtrage sur les performances de GPT-2.
5. Rédaction du protocole, du runner, de l'adaptateur, du diagnostic et de l'audit ; contrôle syntaxique. Le modèle est chargé pour épingler les sources importées, sans passe forward. Gel local de 1 291 fichiers dans `results/protocol_frozen.json`, avant toute sortie nouvelle.

## Campagne prévue et conservation

Le runner collecte les 48 lectures natives et shams, les 128 essais C et les 384 conditions E. Les mêmes seuils descriptifs que pour NCM sont appliqués sans traduction en vérité mécaniste. Seules les représentations du dernier token sont échangées ; le contexte des positions antérieures appartient au receveur.

L'analyse et l'audit ont été inclus dans le gel. Les rapports et le complément de papier sont rédigés après résultats. Les comptes effectifs et l'état de l'exécution figurent dans `results/execution.json`, `results/summary.json` et `results/verification.json`. Les fichiers de comparaison historiques sont vérifiés en lecture seule. Aucun nouvel entraînement, réglage de dose après résultats ou essai de diffusion ne fait partie de cette campagne.

L'audit est réalisé par le même agent avec une implémentation explicite des sous-couches pour la reprise et un calcul scalaire des pertes. Il ne s'agit pas d'une réplication externe. Les couches du calcul d'un prochain token ne sont pas les étapes temporelles d'une sortie autorégressive déjà émise ; aucune équivalence entre ces deux axes n'est revendiquée.

## Résultats et export complémentaire après analyse

Le runner s'est terminé sans correctif ni amendement : 48 parités/shams, 128 essais C, 384 conditions E. Le diagnostic fixé donne huit familles à préférence dominée par l'entrée et huit abstentions pour échec d'au moins une ancre native. Aucun seuil, prompt, niveau ou dosage n'est modifié après ces résultats.

L'audit gelé est passé à 2026-10-25T15:11:39Z : 1 024 recompositions, 16 reprises C et 16 E par sous-couches explicites, six passes natives et 640 pertes scalaires. Les manifestes antérieurs et les fichiers gelés sont inchangés.

Après lecture des scores principaux, ajout de `postprocess.py`, **absent du gel initial**, pour exporter les changements appariés d'argmax et leurs gains/pertes de justesse. Ce calcul descriptif utilise exclusivement les sorties existantes, ne sélectionne aucune nouvelle intervention et n'affecte pas les critères. Il confirme zéro changement d'argmax de Y10, deux pertes sans gain dans chacune des conditions adaptées, et un gain sans perte de la régénération contre la suite figée du receveur dans les échanges A/B. Les versions des distributions installées sont aussi exportées à titre de reproduction ; seule la version NumPy réellement importée sert au calcul.

Rédaction de `RAPPORT.md` et `PAPER_ADDENDUM.md` après résultats. La signature proposée par architecture n'est pas forcée. Les comparaisons AR/NCM restent conditionnelles aux tâches et canaux différents ; la diffusion n'a pas été exécutée. Aucun nouveau travail théorique de coordination n'est inféré de ces observations.

Production d'une version 2 autonome, `PAPER_MULTIMODEL.md`, par assemblage éditorial avec `build_paper.py`, ajouté après le gel expérimental. Le script reprend les parties antérieures, ajoute les méthodes et résultats AR, et actualise résumé, portée et limites. Il n'effectue aucun calcul de réseau ou de statistique et conserve le manuscrit source inchangé. Les parties théoriques reprises ont été relues dans la campagne antérieure ; aucune nouvelle relecture externe du manuscrit consolidé n'est revendiquée.

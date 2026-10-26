# CPU Apple Silicon — amélioration complémentaire

**Pas d’optimum démontré. Une option de débit supplémentaire est validée pour les corpus longs.**

Quatre processus CPU persistants, un thread chacun, traitent les lots de 8 inchangés. Ordre initial conservé, y compris le dernier lot incomplet. Poids, FP32, argmax et noyau numérique du commit `254bb49` inchangés. Le défaut reste un processus à quatre threads.

## Résultats

- **384 grilles, trois difficultés :** codes, reconstruction finale et logits identiques octet par octet (SHA-256 avec type et forme). Réponses identiques aux empreintes du rapport immuable `254bb49`. Lots incomplets également vérifiés.
- **Contrôle CLI séparé, 513 grilles easy : 38.34 → 33.11 s**, import, vérification des poids, chargement, chauffe, calcul et arrêt inclus. Débit **+15.8 %**, temps réduit de **13.6 %**. Réponses strictement identiques. Une invocation par configuration : résultat local, sans intervalle de confiance.
- Dans la validation, RSS maximal d’un enfant : **554 Mo**. Plusieurs modèles résident en mémoire ; les RSS partagent des pages et ne constituent pas un total RAM physique. Le démarrage du pool coûte environ 5,3–6,4 s dans ce protocole.
- Les ratios appariés avec tous les modèles résidents sont plus élevés et variables ; ils ne sont pas utilisés comme gain annoncé.

## Pistes écartées

- Le profilage attribue 52–68 % du temps aux projections matricielles. Cela indique un coût dominant, pas une preuve de performance maximale.
- Disposition des poids en colonnes : ~10 % en sonde, mais logits différents (max 1,43e-6). Écartée de la voie stricte.
- TorchScript figé : ~6 % au batch 1, ~2 % au batch 8 après chauffe ; compilation ~7 s. Non intégré, bénéfice limité et validation globale non poursuivie.

## Utiliser la version candidate 0.2.0

```sh
.venv/bin/ncm-sudoku --difficulty hard --n 0 --workers 4 --output audit/hard-parallel.json
```

Cette option améliore le débit d’un corpus, pas la latence d’une grille isolée. Elle ajoute mémoire et démarrage ; le mode séquentiel reste adapté aux petits essais. Aucun gain global contre l’article n’est calculé en multipliant des expériences différentes. Validation exhaustive des 6 000 grilles toujours requise avant publication.

Preuves : [comparaison exacte](audit/cpu-parallel-validation.json), [contrôle CLI séparé](audit/cpu-cli-control.json), [profils](audit/cpu-profile-b8.json), [graphe apparié](audit/cpu-graph-paired.json), [journal](FEUILLE_DE_ROUTE.md).

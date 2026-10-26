# Amendement A1 — audit à précision égale et backend officiel

25 octobre 2026. Amendement après examen des résultats de développement ; aucune sortie des cas réservés n'a été calculée.

## Motif documenté

La comparaison gelée entre Torch FP32 et NumPy FP64 échoue dans 15/256 sources, mais conserve tous les argmax. Une comparaison des mêmes équations en FP64 concorde à 4,2633e-12. La première tolérance n'était pas justifiée par une borne d'erreur de propagation de ce réseau. Le premier audit et son gel restent inchangés, avec leur statut d'échec.

L'objectif de cet amendement est de comparer les équations à précision égale avec le code officiel et d'employer le backend officiel pour les interventions. Il ne relève pas la tolérance de l'ancien test. Les résultats A1 restent sur développement et ne deviennent pas confirmatoires par ce nouveau gel.

## Règles A1 fixées avant son exécution

Charger les checkpoints par `HookedTransformer` 1.19.0 avec les configurations publiées, sans normalisation, sans réinitialisation des poids, sur CPU. Comparer ses logits FP64 aux logits NumPy FP64 sur les 256 sources, avec `atol=1e-8`, `rtol=1e-10`. Un échec arrête A1 sans essai réservé.

Exécuter les cinq croisements avec les hooks officiels, aux deux mêmes sites et avec les mêmes donneurs que le pilote initial. Les unités sont `attn.hook_result` pour la tête entière ; les suites figées utilisent `hook_mlp_out` et `hook_attn_out`, dans leur ordre natif. Vérifier l'égalité bit à bit du replay natif et du sham dans ce backend. Comparer tous les argmax à l'archive du premier pilote, et rapporter l'écart continu entre les deux implémentations ; aucune tolérance continue opportuniste n'est interprétée comme une équivalence d'effet.

Ce contrôle teste TransformerLens officiel, pas encore le programme compilé Tracr ni sa correspondance sur les tâches réservées. Toute sortie réservée reste interdite tant que l'oracle complet, ses positions définies et l'extension prédictive ne sont pas prêts.

## Dépendances isolées

Les tentatives d'import ont révélé des incompatibilités sans relation avec les effets scientifiques : datasets5.1/HuggingFace Hub0.36, pyarrow26/NumPy1.26, Accelerate récent/namespace NumPy interne, puis dépendances de télémétrie récentes et protobuf. Les versions conservées pour A1 sont datasets2.21.0, pyarrow19.0.1, Accelerate0.33.0, wandb0.16.6, protobuf4.25.8 et TransformerLens1.19.0. Aucune session wandb n'est démarrée. NumPy et Torch sont importés avant les dépendances optionnelles. Les fichiers de la bibliothèque officielle utilisés par A1 et les scripts sont épinglés dans un nouveau manifeste avant exécution.

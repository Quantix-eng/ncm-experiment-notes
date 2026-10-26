# Provenance

Code sous `src/ncm_mac/_vendor/` : [HazyResearch/ncm](https://github.com/HazyResearch/ncm), commit `c70274273f489abc73832b2b98584a8483b6b203`, licence MIT conservée dans `LICENSE`. Modèles, tokenizers et utilitaires sont inclus ; les poids ne sont pas redistribués ici.

Modifications locales : imports qualifiés dans le paquet `ncm_mac._vendor`, copie des instantanés dans `MultiscaleResidualVectorQuantizer.idx_to_zhats`, et option `return_reconstruction` de `generate` pour réutiliser l'accumulateur déjà calculé. L'appel historique à `generate` garde son type de retour. Les poids, couches, paramètres, masques, température et argmax Sudoku restent ceux des auteurs.

Les ajouts dans `src/ncm_mac/` hors `_vendor`, `scripts/` et `tests/` implémentent l'exécution, l'évaluation et la validation locales. `upstream-ncm/` reste une référence non modifiée, exclue du futur dépôt ; il est nécessaire uniquement aux comparaisons de parité.

## Pourquoi le calcul final est conservé

En CPU FP32 sans autocast, `generate` et `idx_to_z_q` démarrent par le même accumulateur nul. Pour les niveaux 0 à K−2, ils appliquent aux mêmes codes la même montée en résolution, la même convolution résiduelle et la même addition en place, dans le même ordre. Les pools utilisés pour prédire le niveau suivant ne modifient pas cet accumulateur. L'option ajoutée applique seulement la dernière convolution et addition à partir du dernier code prédit. Elle évite donc le second passage sur les niveaux précédents, sans changer la récurrence numérique. La validation compare également les logits, pas seulement l'argmax.

Le correctif `clone()` ne concerne que l'historique retourné : les objets deviennent indépendants, tandis que l'accumulateur de calcul continue inchangé. L'ancien comportement d'aliasing ne doit pas être reproduit pour les études intermédiaires.

## Cache du contexte Sudoku

`ncm_mac/cached.py` exploite le masque d'attention original : les positions du préfixe ne peuvent jamais lire les codes cibles. Le préfixe est donc indépendant du niveau généré. Ses clés/valeurs par couche sont calculées une fois, avec les mêmes positions RoPE ; chaque niveau ne calcule ensuite que ses nouvelles requêtes/clés/valeurs, en conservant l'attention bidirectionnelle à l'intérieur du niveau. Ce cache concerne uniquement le préfixe, pas les anciens codes cibles. Les caches sont réinitialisés à chaque niveau et supprimés dans un `finally`.

L'équivalence mathématique ne suffit pas à promettre une identité numérique sur tous les backends : les comparaisons bit à bit restent obligatoires. Le chemin est limité à l'architecture Sudoku publiée et rejette les autres configurations.

Article : Hermann Kumbong et al., *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*, [arXiv:2610.14162v1](https://arxiv.org/abs/2610.14162v1).

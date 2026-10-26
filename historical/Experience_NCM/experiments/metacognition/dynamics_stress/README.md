# Calibration de stress — arrêt sur apprentissage

Ce suivi a été spécifié après le constat que le remplacement d'un code au niveau8 changeait rarement la résolution finale. Il ne remplace pas les deux critères de l'étude native.

**Seul `pilot` a été exécuté.** Sur128 anciens puzzles d'apprentissage, remplacer1/4/8/13/26 codes par des alternatives uniformes n'a pas atteint la baisse de résolution à suite fixe d'au moins10points requise par le [design](design-v1.json). `selected_count=null`, arrêt prévu respecté : aucun corpus externe de stress généré, aucun résultat causal adaptatif de ce suivi. `generate` et `run` restent des prototypes conditionnels non exécutés et non validés sur un corpus externe.

```sh
.venv/bin/python experiments/metacognition/dynamics_stress/stress.py pilot
.venv/bin/python scripts/diagnose_native_codec.py
```

Les deux commandes refusent d'écraser leurs sorties. Reproduction dans une autre copie de travail sans leurs dossiers de résultats. Source et tableaux du pilote conservés dans `audit/metacognition/native-stress-v1/` ; les empreintes permettent une relecture des comptes. Le diagnostic du décodeur est post-hoc, sur ces128 mêmes puzzles ; ce n'est pas une confirmation externe. Il ne teste pas si le générateur a besoin des niveaux précédents pour prédire les codes finaux.

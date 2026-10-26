# D11 — Rapport de réutilisation et de compatibilité

## Décision appliquée

Cette livraison utilise les partitions et les contrefactuels publics de [MIB–IOI](https://huggingface.co/datasets/mib-bench/ioi). Elle se présente comme une extension d'évaluation de compensation, sans remplacer les pistes de localisation de circuits et de variables de [MIB](https://proceedings.mlr.press/v267/mueller25a.html). Aucun dépôt ni soumission au leaderboard n'a été réalisé.

## Format effectivement fourni

`01_ioi/results/*_observations.json` associe à chaque famille un template, les donneurs effectif et mal apparié, les sept conditions (cinq croisements et deux blocages), NLL, marge IO−S, exactitude à deux candidats, exactitude vocabulaire et logit centré. Le score self-repair et la variation de DE à facteur LN original sont conservés séparément. `*_states.npz` conserve les résidus finaux et les suites pour permettre le redécodage ; ce choix évite de prendre un arrondi de rapport pour donnée primaire.

Le script `analyse.py` produit D,R,I,S,B, les résultats par template, les seuils opérationnels et la couverture des affirmations avec/sans spécificité. La version sans contrôle figé s'abstient explicitement sur la nécessité d'adaptation. `verify.py` vérifie des interventions contre une passe complète officielle munie d'un hook et refait les pertes par une sommation scalaire distincte. Ces vérifications ne sont pas une évaluation extérieure de méthodes.

Les résultats et temps réellement obtenus figurent dans les rapports des trois branches. Un temps écoulé de session n'est pas un coût marginal universel. Aucun classement commun DLA/self-repair/EAP-IG n'est fabriqué : ces méthodes n'ont pas toutes le même objet. Les mesures publiées sont des comparateurs calculés par notre implémentation, pas des soumissions indépendantes par leurs auteurs.

## Ce qui reste à établir

L'adaptateur n'est pas un plugin officiel MIB validé par ses mainteneurs. Le choix explicite de BOS et le pipeline Hugging Face sont documentés, sans annoncer une équivalence bit à bit à TransformerLens/MIB. Une réutilisation extérieure, plusieurs méthodes développées indépendamment et une validation humaine sont encore nécessaires avant de revendiquer une référence communautaire. Le paquet actuel rend ces tests concrets et auditables ; il ne les remplace pas.

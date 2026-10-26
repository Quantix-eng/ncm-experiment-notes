# Dynamique native et compensation — étude locale v1

Le code et les données des études gelées, ainsi que les poids, sont inchangés ; les documents de synthèse sont actualisés. Référence : [protocole](protocol-v1.json). Deux critères principaux sont fixés avant l'inférence externe : gain de classement des erreurs persistantes par une lecture dynamique, puis gain de résolution vérifiée par l'adaptation des niveaux futurs après remplacement d'un code. Intervalles à 97,5 % pour ces deux critères ; les autres comparaisons sont secondaires.

Depuis la racine :

```sh
.venv/bin/python experiments/metacognition/dynamics/study.py technical
.venv/bin/python experiments/metacognition/dynamics/study.py training
.venv/bin/python experiments/metacognition/dynamics/study.py fit
.venv/bin/python experiments/metacognition/dynamics/study.py generate
.venv/bin/python -m unittest discover -s tests -p test_native_dynamics.py
.venv/bin/python experiments/metacognition/dynamics/study.py lock
.venv/bin/python experiments/metacognition/dynamics/study.py extract
.venv/bin/python experiments/metacognition/dynamics/study.py evaluate
.venv/bin/python experiments/metacognition/dynamics/study.py causal
VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/verify_native_dynamics_bounded.py
.venv/bin/python scripts/summarize_native_dynamics_diagnostics.py
```

Les sorties numériques sont dans `audit/metacognition/native-dynamics-v1/`. Chaque stage refuse d'écraser ses sorties, hormis la relecture arithmétique qui peut réécrire son rapport. Une réplication complète nécessite une autre copie de travail sans ce dossier de résultats. Les traces d'apprentissage antérieures et leurs manifestes sont nécessaires : le replay des codes ne relance pas le générateur. Les cellules de chaque solution restent dans le même groupe. Les décompositions PCA utilisent seulement les groupes d'ajustement, et les hyperparamètres sont choisis par les cinq plis d'apprentissage.

`generate` peut s'exécuter pendant le replay/apprentissage : il ne consulte aucun résultat NCM et écrit un autre dossier. Le corpus comprend 2 000 puzzles nouveaux à solution unique, dont les 1 500 premiers sont fixés pour les interventions. Le gel lie sources numériques, coefficients et corpus avant la première inférence externe. Aucun échantillon n'est ajouté selon l'effet mesuré.

Les coordonnées latentes sont celles de l'entrée du décodeur ; elles ne sont pas présumées localisées sur une cellule après mélange des positions par ses blocs. Le delta est une différence entre deux états, et une amélioration d'un observateur limité n'est pas une information théorique supplémentaire. La cohorte d'erreurs courantes utilise la solution uniquement pour son masque et sa cible ; elle n'est pas déployable telle quelle.

L'intervention remplace un seul code au niveau 8. Un passage fictif doit reproduire exactement la trajectoire de référence. Le témoin « suite fixe » garde les codes propres des niveaux 9–12, alors que le traitement adaptatif les régénère. Le second remplacement choisit l'alternative distincte la plus proche en norme de perturbation latente ; l'appariement discret est imparfait et son écart est rapporté. Les probabilités rapportées sont celles de la distribution propre avant le forçage, pas celles des logits artificiellement forcés.

La compensation latente mesure la projection de la réponse des futurs niveaux dans la direction opposée à la perturbation initiale. Une projection positive ne garantit ni amélioration sémantique ni surveillance de fiabilité. Le critère causal principal porte donc sur la résolution stricte, format/contraintes/indices inclus. Le témoin à suite fixe est une intervention mécaniste, pas une politique concurrente de résolution.

Les simulations de sensibilité de puissance sont des approximations normales conditionnelles (puzzles mixtes et discordances supposés), sans garantie de puissance ni intégration de la variance de réentraînement. Le corpus généré n'est pas une réplication IID garantie de l'officiel ; l'exclusion des transformations ne couvre pas tous les isomorphismes ou une contamination de préentraînement. Une preuve de métacognition native demanderait en plus une dissociation du traitement du contenu et de l'évaluation de sa fiabilité.

La relecture exécutée utilise des blocs de1024cellules et une implémentation arithmétique indépendante, pour limiter la pression mémoire. `verify.py` conserve la variante complète initiale figée avec le protocole ; sa comparaison stricte des intervalles partage la limite d’arrondi documentée. Utiliser l’auditeur borné corrigé ci-dessus pour la relecture courante. Les descriptifs et intervalles exacts supplémentaires sont post-hoc et ne remplacent pas le bootstrap principal. Le pilote initial est conservé ; `technical-replication-final-source/technical.json` valide les sources finales après ajout de l’archivage causal.

# Contrôle précoce — faisabilité locale

Étude exploratoire du 24 octobre 2026 : abandonner certaines trajectoires au niveau 9 pour économiser du calcul, en comptant les résolutions perdues par puzzle soumis. Ce n'est ni une nouvelle confirmation de H1 ni un indicateur de confiance calibré validé indépendamment.

Le [design](design-v1.json) définit les sondes, les comparateurs, les marges et la porte de faisabilité. Les anciens scores OOF avaient été inspectés avant ce design ; les nouveaux seuils sont choisis sur le développement. Aucune revendication de pré-enregistrement public. Les protocoles et scripts antérieurs restent inchangés.

Le moteur `engine.py` conserve l'accumulateur latent et les clés/valeurs du préfixe au niveau 9. Les trajectoires conservées sont regroupées en lots de huit ; seul le dernier lot est complété. Cette contrainte conserve la parité numérique avec la référence. Le moteur ne consulte ni les solutions ni les labels pour décider de continuer.

Les modèles `codes`, `hidden`, `codes_hidden` sont ajustés avec pénalité fixe 1 et cinq plis groupés, prétraitements propres au pli. `summary_archived` utilise les modèles externes archivés et calcule en ligne les neuf grilles intermédiaires. Les modèles utilisés sur chaque pli n'ont pas appris ses labels ; le choix du seuil consulte toutefois les résultats OOF de tout le développement. Les scores restent des logits de risque non calibrés pour cette étude.

Depuis la racine, dans l'environnement existant :

```sh
.venv/bin/python experiments/metacognition/early_control/study.py technical
.venv/bin/python experiments/metacognition/early_control/study.py fit
.venv/bin/python experiments/metacognition/early_control/study.py profile
.venv/bin/python scripts/verify_early_control.py
.venv/bin/python -m unittest discover -s tests -p 'test_early_control.py'
```

Les trois commandes expérimentales refusent d'écraser leurs sorties existantes sous `audit/metacognition/early-control-v1/`. Pour une reproduction complète, utiliser une copie de travail séparée avec les actifs vérifiés et les archives d'apprentissage. Ne pas supprimer les preuves de la première exécution. L'auditeur relit les résultats existants sans inférence ni ajustement.

Le pilote technique teste 32 puzzles, les caractéristiques, la reprise, les regroupements et le remplissage. Le profilage teste aussi réellement le cas où aucune trajectoire n'est conservée. La version du pilote technique précède l'ajout des comparateurs aléatoires et de la règle conditionnelle de répétition dans le conducteur ; le moteur numérique est identique. Les empreintes de chaque phase permettent de distinguer ces révisions, toutes antérieures au profilage final.

Les coûts incluent extraction des variables, score, regroupement, reprise, décodage final et vérification. Initialisation du modèle, chargement des coefficients et des données, ajustement et audit ultérieur sont exclus : coût d'inférence avec modèle résident. La file contient au plus sept trajectoires après vidage des lots, mais peut augmenter la latence individuelle ; l'étude mesure le traitement d'un corpus, pas une garantie de latence interactive.

Une confirmation indépendante ne serait justifiée qu'après passage de la porte optimiste, puis dimensionnement, séparation des données et gel. Le rapport racine précise la décision effectivement obtenue : [RAPPORT_CONTROLE_PRECOCE.md](../../../RAPPORT_CONTROLE_PRECOCE.md).

Clôture documentaire sans inférence : `.venv/bin/python scripts/close_early_control.py`, après l’audit arithmétique et la mise à jour des rapports. Le manifeste de clôture archive les empreintes des documents et la décision.

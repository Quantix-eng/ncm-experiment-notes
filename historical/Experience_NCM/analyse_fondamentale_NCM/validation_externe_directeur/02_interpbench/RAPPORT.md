# InterpBench — audit des ressources et pilote de développement

**Dernier état :** l'[évaluation réservée C2 et le contrôle C3](RAPPORT_C2.md) sont maintenant exécutés. Le présent rapport conserve le pilote initial et ses arrêts ; il ne décrit plus l'état final de la branche.

**Mise à jour de clôture :** le [complément](COMPLEMENT_CLOTURE.md) décrit les audits officiels et l'oracle de développement ajoutés après ce premier rapport, ainsi que les correspondances encore non résolues avant réserve. Les résultats ci-dessous conservent la chronologie initiale.

25 octobre 2026. Mise en œuvre de la proposition 2 du directeur. [Feuille de route](FEUILLE_DE_ROUTE.md), [extension prédictive préparatoire](EXTENSION_PREDICTIVE.md), [protocole du pilote](PROTOCOLE_DEVELOPPEMENT.md).

## Résultat livré et statut scientifique

Les six cas recommandés existent et leurs artefacts sont conservés. Un pilote a exécuté les deux tâches de développement avec les poids publiés, sans apprentissage : 256/256 séquences natives sont correctes et 512/512 échanges régénérés sur des têtes hors circuit respectent l'invariance prédite par les programmes extérieurs. Toutes les conditions croisées restent exactes ; ce pilote ne met donc en évidence aucun dommage d'exactitude à réparer.

Le premier audit numérique échoue sur un critère fixé avant inférence : 15/256 comparaisons de logits Torch FP32–NumPy FP64 dépassent sa tolérance. L'échec, les fichiers et le gel sont conservés. Il ne sera pas transformé en réussite par un changement silencieux de tolérance. Une vérification complémentaire des mêmes équations en FP64 concorde à 4,27e-12 près et tous les argmax FP32/FP64 concordent. Cela indique un effet de précision dans cette comparaison plutôt qu'un désaccord des équations testées, mais ne remplace pas une vérification du chargeur officiel. Aucune des quatre tâches réservées n'a été exécutée.

Les comptes sont ceux d'un pilote interne sur ressources externes. Ils ne valident pas encore l'extension prédictive du benchmark et ne prouvent aucun avantage face à l'évaluation IIA standard.

## Ressources extérieures vérifiées

L'article v3 présente 86 modèles semi-synthétiques ; le manifeste public contient effectivement 86 identifiants, dont IOI. La version sémantique du manifeste est `1.0.0`. Les deux nombres désignent des choses différentes : version d'artefact et effectif de la collection. [Article original](https://arxiv.org/html/2407.14494v3), [manifeste épinglé](https://huggingface.co/cybershiptrooper/InterpBench/blob/a1242a84f8a4d07d1be9a8f5ec710415198016b2/benchmark_metadata.json).

Les révisions conservées sont GitHub InterpBench `220791e4a9cd29957ed5edfa445e7c09152d7298`, Hugging Face `a1242a84f8a4d07d1be9a8f5ec710415198016b2` et fork Tracr `98f4910e3966805c83d80eb087c7a4e3328171c6`. [Inventaire local, configurations et empreintes](results/artifact_inventory.json). Les poids vont de 14 058 à 507 554 octets par cas ; leur taille rend le calcul local possible. Cela ne réduit pas le travail requis pour vérifier les correspondances causales.

Chaque cas conserve `ll_model.pth`, `ll_model_cfg.pkl`, `meta.json` et `edges.pkl`. Les poids de développement sont chargés comme tenseurs avec `torch.load(..., weights_only=True)`. Les configurations exécutées viennent des métadonnées JSON. Les graphes sont inspectés avec un désérialiseur interdisant les classes globales ; les pickles de configuration ne sont pas exécutés. Le code distant a été lu comme source avant son utilisation ; aucun entraînement n'a été relancé.

## Vérification de la sélection proposée

- **8 — identité**, développement : chaque entier est restitué à sa position. Deux blocs, quatre têtes par bloc, dimension résiduelle 4 ; le code marque ce cas comme trivial. [Source officielle](https://github.com/FlyingPumba/InterpBench/blob/220791e4a9cd29957ed5edfa445e7c09152d7298/circuits_benchmark/benchmark/cases/case_8.py).
- **33 — parité de longueur de chaque token**, développement : `len(token) % 2 == 0`. Le mot « longueur » ne désigne pas la longueur de la séquence. Le cas est aussi marqué trivial. [Source officielle](https://github.com/FlyingPumba/InterpBench/blob/220791e4a9cd29957ed5edfa445e7c09152d7298/circuits_benchmark/benchmark/cases/case_33.py).
- **13 — tendance numérique**, réservé : comparaison à la valeur suivante, avec fin de séquence indéfinie dans le programme. Attention bidirectionnelle. [Source officielle](https://github.com/FlyingPumba/InterpBench/blob/220791e4a9cd29957ed5edfa445e7c09152d7298/circuits_benchmark/benchmark/cases/case_13.py).
- **18 — fréquence des tokens**, réservé : fréquent si compte > longueur/2, commun si > longueur/4, rare sinon. Attention bidirectionnelle. [Source officielle](https://github.com/FlyingPumba/InterpBench/blob/220791e4a9cd29957ed5edfa445e7c09152d7298/circuits_benchmark/benchmark/cases/case_18.py).
- **19 — doublons consécutifs**, réservé : conserver le token quand il diffère du précédent, sinon produire `None`. Longueur maximale 15, BOS compris. [Source officielle](https://github.com/FlyingPumba/InterpBench/blob/220791e4a9cd29957ed5edfa445e7c09152d7298/circuits_benchmark/benchmark/cases/case_19.py).
- **21 — premières occurrences uniques**, réservé : programme fondé sur un comptage des occurrences antérieures et une agrégation. Son exemple de documentation ne suffit pas à spécifier toutes les positions ; utiliser les équations et le compilateur. [Cas officiel](https://github.com/FlyingPumba/InterpBench/blob/220791e4a9cd29957ed5edfa445e7c09152d7298/circuits_benchmark/benchmark/cases/case_21.py), [programme partagé](https://github.com/FlyingPumba/InterpBench/blob/220791e4a9cd29957ed5edfa445e7c09152d7298/circuits_benchmark/benchmark/common_programs.py).

La famille de sorties est catégorielle token par token. Certaines positions RASP `None` n'ont pas une classe de sortie dédiée : les trois têtes de sortie des cas 13, 19 et 21 ne donnent pas automatiquement une référence catégorielle à ces positions. La gestion officielle des sorties indéfinies devra être vérifiée avant toute mesure réservée ; le masque de l'oracle dépendra de l'intervention. Cette précaution évite de transformer un argmax arbitraire en vérité mécaniste.

## Pilote exécuté

Le gel du pilote est daté de **2026-10-25T17:40:53.230407+00:00** ; 55 fichiers, dont scripts, données, sources et six checkpoints, sont épinglés. Ce gel local n'est ni une prérégistration indépendante ni une preuve d'exhaustivité de toutes les explorations. Les sources sont tirées sans sélection sur dommage : 128 séquences uniques par tâche, 3 à 9 tokens utiles, puis BOS/PAD jusqu'à dix positions. Les donneurs sont i+1 et i+2 modulo 128 ; ils ont la même forme tensorielle mais peuvent différer en longueur utile.

Les sites L0H0 et L1H0 sont fixés avant calcul. Les deux graphes externes ne contiennent aucune tête d'attention ; leur prédiction fonctionnelle est donc l'invariance sous échange de tête entière native. La frontière conserve le résidu juste après l'attention du bloc, avant son MLP ; le suffixe ajoute dans l'ordre les contributions MLP/attention restantes. Les cinq conditions Y00, Y10, Y11, Y01 et Y1wrong sont sauvegardées. Le mauvais suffixe provient du deuxième donneur au même site.

L'adaptateur autonome et la relecture NumPy utilisent les équations publiées de TransformerLens 1.19.0, sans normalisation et avec GELU pour ces checkpoints. La première implémentation tourne en CPU FP32, un thread. Le temps du runner est **0,354 s**, chargement de ces petits poids compris ; l'installation des dépendances et les audits sont exclus. Ce temps n'est pas un coût comparatif de méthodes ou une estimation de campagne complète. [Runner](pilot.py), [sorties détaillées](results/development.json), [logits](results/development_logits.npz).

Chaque cellule tâche/site a 128/128 séquences entièrement exactes dans chacune des cinq conditions : 512/512 échanges Y11 conformes à l'invariance, et aucune erreur d'exactitude induite par Y10. Les shams et les suffixes natifs reproduisent exactement les sorties du runner. Les sources d'un même cas ne constituent pas des tâches indépendantes ; les deux sites et les donneurs sont corrélés.

Les moyennes descriptives D, R, I, S en nats/token sont :

- Cas 8, L0H0 : 2,4691e-5 ; 2,1865e-5 ; 5,0811e-5 ; 4,4894e-5.
- Cas 8, L1H0 : 5,0521e-4 ; 3,0981e-4 ; 4,1432e-4 ; 1,9902e-4.
- Cas 33, L0H0 : quatre zéros FP32 observés.
- Cas 33, L1H0 : 0 ; -3,1733e-7 ; -3,1733e-7 ; -1,8318e-7.

Ces petits effets continus n'annulent pas l'invariance catégorielle observée et ne suffisent pas à établir une invariance exacte des logits. Aucun seuil de réparation ou de spécificité n'est déduit après coup de ces nombres.

## Échec numérique conservé et diagnostic

Le vérificateur distinct, gelé avant calcul, impose `atol=2e-4`, `rtol=2e-5` pour comparer NumPy FP64 aux logits FP32 du runner. Il s'arrête sur `dev-8-017`, avec écart maximal 5,5386e-4. [Échec enregistré](results/verification_failed.json). L'audit post-hoc examine ensuite toutes les sources sans modifier ce critère : 241/256 passent, 256/256 conservent tous leurs argmax, l'écart maximal est 1,4335e-3. Les 55 empreintes initiales restent identiques. [Diagnostic complet](results/numerical_failure_diagnostic.json).

Une seconde analyse post-hoc exécute les mêmes équations Torch en FP64 puis les compare à NumPy FP64. L'écart maximal descend à 4,2633e-12 sur 256 sources. Elle est compatible avec une amplification d'arrondis FP32 dans ce réseau ; elle ne certifie pas que l'implémentation autonome respecte toutes les conventions du chargeur officiel. Le premier audit reste un échec selon son critère annoncé. [Résultats FP64](results/fp64_equation_diagnostic.json).

## Ce qui manque à la validation externe

Le chargeur officiel combine TransformerLens, Tracr et IIT pour construire les modèles de référence et leurs correspondances. L'environnement initial ne contient aucun de ces trois paquets. Le `pyproject.toml` officiel contient notamment TransformerLens 1.19.0 et plusieurs dépendances Git ; une installation globale aurait risqué de modifier les anciennes campagnes. Les dépendances préparatoires sont donc isolées dans ce dossier. Leurs incompatibilités doivent être résolues et journalisées avant une parité officielle.

Le graphe d'arêtes publié suffit à annoncer les contrôles négatifs du pilote, mais ne fournit pas à lui seul toutes les valeurs contrefactuelles d'une variable alignée. Il faut le programme compilé et sa correspondance. La vérité publiée porte sur les têtes entières et les MLP ; aucune annotation Q/K/V n'en est déduite. [Consigne officielle de granularité](https://huggingface.co/cybershiptrooper/InterpBench), [chargeur officiel inspecté](https://github.com/FlyingPumba/InterpBench/blob/220791e4a9cd29957ed5edfa445e7c09152d7298/circuits_benchmark/utils/ll_model_loader/interp_bench_model_loader.py).

La spécification de prédictions réservées a été écrite séparément. Elle exige des hypothèses scellées avant leurs falsifications, un oracle inaccessible au diagnostic, une mesure conjointe couverture/contradictions et une comparaison honnête à IIA/SIIA. Aucune précision élevée obtenue par abstention générale ne sera présentée comme utile. Aucun succès des deux programmes triviaux ne sera transféré aux quatre tâches réservées. Cette branche est une préparation et un pilote de contrôle ; la validation scientifique externe reste à effectuer.

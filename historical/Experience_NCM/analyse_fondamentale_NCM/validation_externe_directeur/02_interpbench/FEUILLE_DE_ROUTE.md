# Feuille de route — validation InterpBench

25 octobre 2026. Proposition 2 du directeur. Cette feuille distingue l'accès aux artefacts, la conformité des interventions et la validation d'un diagnostic. [Rapport de mise en œuvre](RAPPORT.md).

## Objectif et portée

Tester des affirmations causales bornées contre des programmes et alignements produits par une équipe extérieure. La question principale sera l'accord des prédictions sur des interventions réservées ; aucune tâche connue ne reçoit automatiquement une étiquette parmi les six témoins historiques. Les 86 modèles annoncés par l'article sont corroborés par les 86 entrées des métadonnées publiques épinglées. [Article, version 3](https://arxiv.org/html/2407.14494v3), [métadonnées officielles](https://huggingface.co/cybershiptrooper/InterpBench/blob/a1242a84f8a4d07d1be9a8f5ec710415198016b2/benchmark_metadata.json).

## P1 — Fixer les tâches, versions et unités

**Fait.** Épingler GitHub, Hugging Face et le compilateur Tracr ; conserver les sources lues, les configurations, les six checkpoints et les graphes. Les seules tâches de développement sont 8 et 33. Les réservées sont 13, 18, 19 et 21, sans remplacement après examen des effets. Tous les apparentements d'une séquence restent dans sa partition. Le pilote prend 128 sources par tâche de développement, soit 256 ; cette allocation est écrite avant calcul. L'évaluation ultérieure gardera 256 familles par tâche réservée.

**Correction appliquée.** Le cas 33 est la parité de longueur de chaque mot. Les cas 19 et 21 rendent certaines positions indéfinies ; ce sont des transductions catégorielles token par token, et non des classifications binaires de séquences. Le vocabulaire de sortie, le masque des positions définies et les conventions BOS/PAD devront être vérifiés pour chaque tâche.

## P2 — Contrôler le domaine et le chargeur

**Fait, avec échec conservé.** Exécuter un pilote CPU FP32 sans apprentissage sur développement, après [gel local](results/protocol_frozen.json). Vérifier les réponses contre deux programmes élémentaires indépendants du diagnostic, les shams et la recomposition des suffixes. La première comparaison NumPy FP64 échoue sur la tolérance de logits préfixée ; l'arrêt et les observations sont conservés. Le diagnostic suivant compare les équations en FP64 pour séparer un désaccord d'implémentation d'un effet de précision. [Diagnostic](results/numerical_failure_diagnostic.json), [comparaison des équations FP64](results/fp64_equation_diagnostic.json).

**À terminer avant réserve.** Faire la parité du modèle autonome et des interventions contre TransformerLens officiel à précision égale. Si une correction est nécessaire, garder le pilote et le gel initiaux, expliquer le changement, horodater un amendement et refaire un gel. Modifier une tolérance après résultats ne fait pas disparaître son premier échec. Le coût de dépendances et les incompatibilités doivent figurer au rapport.

## P3 — Restaurer l'oracle interventionnel complet

**Préparé, non exécuté sur réserve.** Compiler les programmes RASP officiels avec la version de Tracr épinglée, puis exporter les encodeurs, sorties définies et correspondances entre variables du programme et têtes/MLP entraînés. Le graphe `edges.pkl` ne suffit pas, seul, à donner une valeur contrefactuelle à toutes les variables internes. Ne jamais reconstituer cet oracle à partir des diagnostics que l'on cherche à évaluer.

Les unités sont les têtes projetées entières et les MLP tels qu'alignés par l'outil officiel. Aucune vérité Q/K/V n'est déduite d'un label de tête. Les échanges proviennent de valeurs natives ; zéro et bruit gaussien restent hors du premier périmètre. Comparer d'abord programme et réseau sous les mêmes échanges, puis mesurer l'échec du diagnostic seulement dans le domaine de conformité documenté. [Avertissement officiel de granularité](https://huggingface.co/cybershiptrooper/InterpBench).

## P4 — Spécifier l'extension prédictive

**Spécification écrite, capacité non validée.** [EXTENSION_PREDICTIVE.md](EXTENSION_PREDICTIVE.md) définit une sortie commune : prédictions de changement/invariance et, lorsqu'elles sont émises, de réponse catégorielle sur une requête interventionnelle précise. Une abstention explicite est autorisée. Les régimes O, C et E fixent l'information offerte à la méthode ; ils ne sont pas des classes de mécanismes.

Séparer une banque de requêtes servant à émettre les hypothèses et une banque distincte servant à leur falsification. Les donneurs de falsification ne doivent jamais avoir servi à choisir la variable, la dose ou la règle d'émission. L'oracle reste caché à la méthode sur ces requêtes. Le protocole historique ne possédait pas cette interface ; son développement est une nouvelle étape, avec gel avant réserve.

## P5 — Évaluer et comparer sans fabriquer un concurrent

**À exécuter après P2–P4.** Sur 256 familles de chacune des quatre tâches, conserver les cas favorables, les contradictions et les abstentions. Rapporter l'accord programme/réseau, IIA et SIIA aux unités appropriées, la couverture de chaque type d'affirmation, le risque parmi les affirmations émises et le couple risque–couverture sous forme de données et de texte. Rapporter chaque tâche avant une moyenne équipondérée des quatre tâches. Aucun échange corrélé ne devient une réplication indépendante.

L'IIA standard est une mesure de conformité entre réseau et programme. Ce n'est pas, seule, une méthode qui prédit une nouvelle intervention. Une comparaison de méthodes exigera des prédictions communes et des informations d'accès déclarées. Si les contrôles supplémentaires n'améliorent aucune conclusion corroborée, la valeur ajoutée face à l'évaluation interventionnelle standard restera non démontrée.

## P6 — Conditions de passage et d'arrêt

La réserve reste fermée si la parité du chargeur n'est pas justifiée, si l'oracle contrefactuel nécessite nos propres diagnostics, si la granularité ou le traitement des positions indéfinies reste ambigu, ou si le réseau diverge du programme sans domaine de validité annoncé. Une réussite limitée aux deux tâches triviales de développement ne déclenche aucune revendication externe. La validation finale sera limitée aux affirmations, tâches et interventions testées ; elle n'établira pas une taxonomie universelle.

Les estimations du directeur de 4–7 jours d'intégration et 1–6 heures CPU demeurent des estimations. Le pilote local est mesuré séparément ; son temps ne prédit pas celui de l'intégration complète.

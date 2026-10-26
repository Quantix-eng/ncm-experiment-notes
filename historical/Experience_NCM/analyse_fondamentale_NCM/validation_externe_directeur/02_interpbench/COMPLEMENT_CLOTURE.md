# InterpBench — complément de clôture

**Reprise ultérieure achevée :** voir le [rapport C2/C3](RAPPORT_C2.md). Le blocage décrit ci-dessous est l'état conservé lors de l'arrêt précédent, avant la nouvelle autorisation de poursuivre.

25 octobre 2026. État postérieur au [rapport initial](RAPPORT.md), qui conserve le déroulement du premier pilote. Aucune inférence des quatre tâches réservées n'a été exécutée. Travail arrêté à la demande de l'utilisateur.

## Audits et progrès réellement acquis

L'amendement A1 emploie le backend officiel TransformerLens1.19.0 : sa version FP64 concorde avec NumPy FP64 à **4,2633e-12** près sur 256 sources. Les 2 816 comparaisons d'argmax avec les archives FP32 concordent ; 512 replays natifs et 512 shams sont bit à bit exacts dans ce backend. Les écarts de logits FP32 entre implémentations atteignent 0,00673 : ils ne sont pas effacés ni transformés en invariance numérique. [Audit officiel](results/official_audit_a1.json), [versions réellement importées](results/environment_actual_imports.json). Le premier échec de tolérance reste conservé.

Un oracle complémentaire utilise les fonctions RASP officielles et les étapes du compilateur Tracr jusqu'au modèle Craft. Les fonctions du compilateur ne sont pas réécrites ; un chargement de namespace évite l'import automatique des dépendances JAX inutilisées. La première tentative B0 et son échec d'import sont conservés, puis B1 est gelé avant son pilote. Aucun de ces résultats n'est une réplication scientifique extérieure.

Sur les deux tâches de développement, **256 sorties natives et 1 536 échanges** ont été contrôlés contre le programme/Craft : **1 536/1 536 accords**, dont 499 changements de réponse oracle ; deux contrôles de propagation des valeurs indéfinies passent. Ces échanges sont corrélés et portent sur les deux programmes de développement simples, pas sur quatre nouvelles familles réservées. [Résultats B1](results/craft_development_b1.json).

## Précondition non satisfaite avant réserve

La compilation statique, sans appliquer les entrées réservées aux réseaux, révèle des correspondances non résolues :

- Cas13 : les arêtes publiées contiennent L0H0 et L0H1, tandis que le programme compilé courant ne donne que L0H0.
- Cas18 : les arêtes publiées ne contiennent pas de tête, tandis que la compilation en produit deux.
- Cas21 : une tête compilée supplémentaire et des étiquettes auxiliaires ne sont pas directement associées à des nœuds RASP.
- Cas19 : les listes statiques de têtes concordent ; cela ne suffit pas à établir toutes les préconditions d'une campagne réservée.

[Schémas statiques conservés](results/static_oracle_schemas_b1.json). Ces différences ne prouvent pas une erreur du benchmark publié : version des fonctions partagées, conventions de compilation et alignement doivent encore être départagés. Il serait incorrect de les résoudre en rebaptisant nos diagnostics « oracle » ou en ouvrant seulement une tâche favorable.

## Suite nécessaire, non exécutée

Identifier la provenance exacte de compilation des checkpoints, établir les correspondances et les masques des sorties `None`, puis sceller les règles d'émission et d'abstention avant les réponses de falsification. Les 256 entrées de chacune des quatre tâches sont préparées, sans sortie réseau. L'extension IIA/couverture/contradictions demeure à évaluer : **validation réservée et avantage scientifique non établis**.

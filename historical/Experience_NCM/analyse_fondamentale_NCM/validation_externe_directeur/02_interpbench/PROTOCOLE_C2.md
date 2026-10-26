# InterpBench C2 — évaluation réservée de prédictions d'invariance

25 octobre 2026. Amendement avant toute application des quatre réseaux réservés. B0/B1, leurs échecs et résultats de développement sont conservés. Aucun réentraînement ni changement de poids. Les sorties privées de falsification sont calculées seulement après gel des prédictions.

## Résolution des préconditions et domaine

Le cas13 a été corrigé dans le dépôt le9septembre2024, commit `bceb41a5c272fb1b1a91611396c89bbf81c87ee0`. Le checkpoint et les arêtes sont identiques à ceux publiés avant cette correction, preuve par identifiants de fichiers des deux snapshots HF dans `sources/history_audit/history_comparison.json`. Utiliser le programme pré-correction, extrait sans modification du parent `48520bd45b1d96d7f61dbd6600f564f090a92fb3`. C'est un alignement de provenance, pas une sélection du programme selon les sorties neuronales. Les trois autres programmes restent ceux du gel B1. Le cas13 est donc explicitement une tâche historique, pas le programme actuel corrigé.

La compilation officielle Craft fournit les composants et leurs bases de sortie. Les arêtes exportées ne sont pas utilisées seules pour déduire l'absence d'un composant : les étapes auxiliaires SelectorWidth des cas18/21 ne sont pas toutes représentées par des nœuds RASP directement échangeables. C2 ne les déclare pas inertes. Les interventions positives portent uniquement sur les composants dont **toutes** les bases de sortie ont un nœud RASP, permettant un masque sémantique sous échange. Les composants auxiliaires non couverts sont exclus du domaine, avant sorties, et dénombrés. Un contrôle négatif est la première tête par couche absente du modèle Craft complet, ou un MLP absent ; l'absence dans `edges.pkl` ne suffit jamais.

L'oracle est le programme RASP externe sous échange des variables de sortie du composant, vérifié contre le modèle Craft officiel exécutant le même échange. On conserve séparément conformité réseau/oracle et qualité des prédictions. Une divergence du réseau avec le programme n'est pas automatiquement une erreur de notre diagnostic. Si Craft et RASP divergent sur une requête, celle-ci est déclarée hors domaine oracle validé et conservée comme échec ; aucun label certain ne lui est inventé.

## Données et contrats d'accès

Les256sources par tâche13/18/19/21 de `data/reserved_inputs.json` sont inchangées. BOS puis tokens utiles, sans PAD ; source et donneurs ont la même longueur. Ce choix évite d'attribuer à des PAD une sémantique de token utile dans les agrégations. Trois donneurs distincts sont fixés par les trois suivants dans le groupe ordonné de même longueur, sans sorties : deux de proposition et un de falsification. Ce nouvel appariement remplace les donneurs de longueurs différentes du fichier préparatoire, qui n'était pas un gel d'expérience.

L'unité est la famille source ; sites et donneurs ne sont pas des réplications indépendantes. Les deux sondes sur une source réservée sont une information disponible **au moment du test**, pas un entraînement sur ses réponses futures. Le domaine testé est la prédiction d'une intervention à donneur nouveau dans une famille déjà sondée, pas la généralisation sans sondes à une nouvelle famille. Les fonctions de prédiction n'accèdent qu'aux sorties autorisées de cette famille ; les réponses futures des autres requêtes ne sont jamais utilisées.

O fournit les sorties natives. C fournit les cinq croisements sur les deux donneurs de proposition au même composant, aux unités tête entière/MLP. E vérifie séparément le plan2entrées×2états complets×3suites au point de coupure. L'échange de mémoire entière bloque l'accès au receveur en régénération ; ce résultat structurel ne sert pas de vérité de classe interne ni de prédiction de l'effet d'une tête partielle.

## Prédictions fixées et comparaisons

La proposition évaluée est étroite : **les sorties catégorielles seront-elles inchangées sous l'échange nommé avec le troisième donneur ?** Toutes les règles émettent soit la sortie native complète comme prédiction d'invariance, soit une abstention. Elles ne prédisent ni une des six classes analytiques ni un mécanisme de réparation.

- O prudent : abstention systématique, couverture nulle explicitement visible.
- O naïf : invariance toujours annoncée, contrôle interne forcé.
- C sans contrôle figé : confiance égale à la fraction des deux Y11 de proposition dont tous les argmax utiles sont natifs.
- C sans contrôle mal apparié : fraction d'accord natif sur Y00,Y10,Y11,Y01 des deux propositions.
- C complet : fraction d'accord natif sur les cinq conditions des deux propositions.

Grille de confiance fixée à0,50/0,75/1,00 ; point principal strict1,00. Le prédicteur ne reçoit aucun label oracle contrefactuel. Les contrastes NLL D/R/I/S sont conservés sur les positions natives définies mais ne sont pas confondus avec une récupération souhaitable d'une variable dont l'oracle change légitimement la réponse. Ce sont des règles internes de prédiction d'invariance, pas des méthodes indépendantes prétendument proposées par la littérature.

Geler et enregistrer **toutes** les prédictions avant de calculer les sorties programme/Craft/réseau avec le troisième donneur. Rapporter par tâche : couverture, contradictions parmi les affirmations et nombre absolu d'affirmations ; risque contre réseau et contre programme séparément. L'IIA/SIIA indique la conformité réseau/programme, pas une méthode de prédiction concurrente. Moyenne équipondérée des quatre tâches, sansp-value ni faux IC de population sur les milliers d'échanges. Une abstention parfaite n'établit pas l'utilité. Aucun changement de seuil après la réserve.

## Contrôles et arrêt

Vérifier gel des sources, finitude, native/replay/sham exacts dans le backend officiel. Vérifier l'oracle RASP/Craft pour chaque requête sur son masque propre ; BOS/PAD/None n'ont pas de classe fictive. Les échecs de conformité du réseau restent dans les résultats, avant toute interprétation des diagnostics. Contrôler E contre des forwards officiels et la composition des suites dans leur ordre. Toute correction logicielle conserve la version précédente et demande un nouvel amendement ; aucune correction des critères sur sorties favorables.

Ce protocole clôt une extension délimitée aux variables sémantiquement alignées. Il ne prétend ni couvrir les auxiliaires du compilateur, ni reproduire toute l'évaluation publiée d'InterpBench, ni disposer d'une réplication humaine extérieure.

# InterpBench — évaluation réservée C2 et contrôle de prétraitement

25 octobre 2026. **Campagne achevée sur un domaine sémantique restreint** : quatre tâches,256 sources chacune,5632 requêtes de falsification. Les résultats ne valident ni les six classes analytiques ni tous les composants d'InterpBench. [Protocole C2](PROTOCOLE_C2.md), [gel](results/c2_frozen.json), [résultats](results/c2_summary.json), [audit](results/c2_verification.json).

## Correspondances : ce qui a été résolu

Le [commit du9septembre2024](https://github.com/FlyingPumba/InterpBench/commit/bceb41a5c272fb1b1a91611396c89bbf81c87ee0) corrige le programme du cas 13, changeant sa fonction et son nombre de têtes. Les poids et arêtes actuels sont identiques à ceux du snapshot HF antérieur à cette correction. C2 utilise donc le programme historique, conservé depuis le parent du commit, plutôt que le programme corrigé incompatible. Cette décision de provenance précède les sorties réservées. Elle ne reconstitue pas tout l'environnement d'entraînement inconnu. Le sens opérationnel des labels est celui du programme historique, avec ses défauts, pas une définition intuitive corrigée de la tendance. [Preuves de fichiers et patch](sources/history_audit/history_comparison.json).

Pour les cas 18/21, l'absence d'une tête dans les arêtes exportées ne suffit pas à la déclarer inerte. Le modèle Craft complet contient des auxiliaires de compilation sans nœud RASP directement échangeable. C2 les conserve dans le modèle, mais les exclut des interventions positives dont le masque sémantique ne serait pas vérifiable. Les contrôles négatifs sont absents du modèle Craft complet, pas seulement des arêtes exportées. Les variables admissibles et exclusions sont [fixées avant calcul](data/c2_plan.json).

L'oracle exécute les fonctions RASP publiées et le modèle Craft compilé, sans dépendre des diagnostics. Une tête entière ou un MLP est échangé avec une activation native de même longueur ; les positions `None` n'ont pas de classe fictive. Sur les **5618 requêtes à masque non vide**, Craft et RASP concordent ;14 requêtes sans cible définie restent explicitement non évaluables par l'oracle.

## Ce qui est prédit et ce qui reste privé

Chaque source reçoit deux sondes au même composant. Les règles voient seulement les sorties autorisées et prédisent une sortie native inchangée pour un **troisième donneur**, ou s'abstiennent. Toutes les prédictions sont [scellées](results/c2_predictions_frozen.json) avant calcul des sorties de ce troisième donneur. Ce test porte sur l'extrapolation à un donneur nouveau dans une famille sondée ; ce n'est pas une généralisation sans sondes à une famille nouvelle.

Les règles O prudente, O naïve et trois règles C avec/sans contrôles sont des **baselines internes**, pas des méthodes indépendantes de la littérature. Leurs scores sont des fractions d'accord sur les sondes, pas des probabilités calibrées. Seuil principal 1,00 ; grille 0,50/0,75/1,00 fixée avant sorties. Les contrastes NLL sont conservés comme descriptions, sans assimiler un échange qui change légitimement la réponse oracle à une corruption qu'il faudrait réparer.

## Conformité des checkpoints au programme

- **Cas 13 historique** :256/256 sources natives exactes ;1024/1024 échanges alignés et 512/512 non alignés conformes au programme.
- **Cas 18** :119/256 sources exactes ;116/256 échanges alignés et 232/512 non alignés conformes. **Ce checkpoint ne satisfait pas la fidélité requise dans ce régime.** Ses erreurs contre l'oracle ne sont pas automatiquement des erreurs du diagnostic.
- **Cas 19** :253/253 sources à cible définie exactes, trois sources natives sans cible définie ;507/507 échanges alignés et 759/759 non alignés évaluables conformes.
- **Cas 21** :256/256 sources exactes ;768/768 échanges alignés et 1277/1280 non alignés conformes. La conformité est élevée mais non parfaite.

Ces comptes IIA/SIIA décrivent la conformité réseau/programme sur les unités retenues ; ils ne sont pas une précision de classification mécaniste. Aucun cas défavorable ni le cas 18 ne sont retirés des agrégats.

## Prédictions réservées : couverture et contradictions

Moyennes équipondérées entre les quatre tâches au seuil strict :

- O prudente : **0 % de couverture**, risque indéfini ; ne pas annoncer 100 % de précision par abstention.
- O naïve : couverture 100 %, contradictions neuronales **41,04 %**.
- C sans contrôle figé : couverture **55,74 %**, contradictions neuronales **2,39 %**.
- C sans contrôle mal apparié : couverture **50,90 %**, contradictions neuronales **2,34 %**.
- C cinq conditions : couverture **50,52 %**, contradictions neuronales **2,35 %**.

Le risque neuronal ci-dessus concerne tous les tokens utiles, y compris ceux sans référence RASP. Le risque contre programme ne porte que sur les positions définies et combine conformité native et prédiction future : il vaut **13,07 %** pour la règle C cinq conditions, dominé par le cas 18. Un [audit post-hoc des masques](results/c2_mask_and_c3_audit.json) fournit aussi les comparaisons réseau/programme sur le même masque, sans changer prédictions ni scores primaires.

La règle C cinq conditions émet respectivement 578/1536,439/768,551/1280 et 1316/2048 affirmations sur les quatre tâches. Parmi celles évaluables par le programme, les contradictions sont 14/578,217/439,1/545 et 3/1316. Les comptes par tâche, les masques vides et les trois seuils sont conservés dans le résumé ; les échanges ne sont pas des unités indépendantes et aucunep-value populationnelle n'est produite.

**Valeur ajoutée non démontrée.** Au point strict, ajouter le contrôle mal apparié retire 23 affirmations, sans retirer aucune contradiction neuronale ; deux contradictions contre programme sont retirées dans le cas 18 déjà non conforme. Les points de couverture différents ne constituent pas une comparaison à couverture égale. Le protocole complet ne montre donc pas ici un avantage prédictif général sur sa version amputée. L'IIA n'est pas artificiellement transformée en méthode concurrente qui émettrait ces prédictions.

## Contrôle du padding : C3, explicitement post-hoc

C2 utilise des séquences non paddées. Après observation du défaut natif du cas 18, le [contrôle C3](PROTOCOLE_C3_PADDING.md) confronte les mêmes sources à l'entrée BOS/PAD du générateur officiel. Les résultats C2 ne sont pas remplacés.

Le cas 18 passe de **119/256 sans PAD à 122/256 avec PAD** contre le programme. Le modèle Craft avec PAD reste conforme au programme pour 256/256 sources. Le padding seul ne résout donc pas la divergence du checkpoint. Les cas 13/19/21 restent entièrement exacts sur leurs références natives définies sous les deux prétraitements. [Résultats C3](results/c3_padding.json). Aucun programme alternatif ni permutation de labels n'a été choisi pour améliorer le résultat.

## Régime E, audits et coût

Le plan 2 entrées×2 états complets×3 suites couvre 55296 conditions. Ses 18432 régénérations reproduisent exactement les logits du donneur, avec une erreur maximale nulle, conformément au chemin receveur bloqué. Ce contrôle structurel reste distinct de l'échange partiel d'une tête et n'entre pas clandestinement dans les prédictions C.

L'audit sans nouveau forward vérifie 45056 propriétés scalaires, les logits sauvegardés, les masques, les donneurs, le gel des prédictions et les sources historiques. Les contrôles native/replay/sham passent dans le backend officiel. Les prédictions et sondes ont pris 83,83 s, la falsification et E103,83 s, hors installation et audits ; ces temps écoulés ne sont pas un benchmark matériel isolé. Le contrôle C3 et son audit sont enregistrés séparément.

## Portée et clôture

L'évaluation réservée est exécutée, avec oracles externes et résultats négatifs conservés. Elle établit une conformité élevée sur trois programmes et un échec de fidélité pour le quatrième, dans les domaines explicités. Elle ne valide pas les auxiliaires exclus, une taxonomie universelle, une supériorité sur plusieurs méthodes indépendantes ou une réplication humaine. Une cause précise de la divergence du cas 18 et l'utilité de nouveaux prédicteurs restent à étudier dans une nouvelle campagne ; ils ne justifient pas de réajuster la présente réserve.

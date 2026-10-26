# Consolidation : tâche commune, six entraînements et contrôle des accès

25 octobre 2026. **La campagne est terminée et auditée.** Deux petits Transformers, AR et parallèle, ont été entraînés sur la même tâche, avec trois graines appariées, puis évalués sur 40 familles réservées. Les six réussissent toutes les variantes natives, y compris les séquences générées librement. Les échanges montrent pourtant des dépendances différentes selon le point de coupure et l'entraînement. La calibration prospective des corruptions a échoué selon son propre critère ; cet échec est conservé.

Le [protocole avant entraînement](PROTOCOLE.md), les [résultats complets](results/summary.json), l'[audit](results/verification.json), le [complément de papier](PAPER_ADDENDUM.md) et la [version consolidée du manuscrit](PAPER_MATCHED_TASK.md) constituent les livrables. Rapports exclusivement Markdown ; les campagnes et articles antérieurs sont inchangés.

## 1. Apport et dispositif

La comparaison précédente changeait simultanément architecture et tâche, NCM/Sudoku contre GPT-2/texte. Cette campagne ajoute un contraste plus contrôlé : même tâche, corpus, partitions, nombre de paramètres, initialisation appariée, batches et budget. Elle ajoute aussi trois entraînements par système, une sélection de dose sur développement séparé et une seconde coupure annoncée à l'avance.

Chaque réseau a **77 576 paramètres**, quatre blocs, largeur 48, quatre têtes et MLP 96. L'AR utilise des requêtes causales et le teacher forcing en apprentissage ; le parallèle utilise des requêtes fixes qui peuvent s'attendre mutuellement. Le parallèle n'est **pas un NCM** : il n'a ni tokenizer appris, ni VQ-VAE, ni hiérarchie de concepts. Le contraste concerne deux systèmes d'entraînement définis, pas un effet pur de l'architecture ou une ablation du NCM.

La tâche donne quatre symboles distincts parmi huit, un bit de rotation 0/1 et un bit de nuisance. La référence est une rotation de la séquence ; la nuisance ne la change pas. A/B diffèrent du bit de rotation ; Aplus change seulement la nuisance. Les 210 orbites de rotations/renversements sont réparties avant apprentissage : 140 entraînement, 30 développement, 40 évaluation. Toutes les orientations d'une orbite restent dans sa partition. Les 4 480 exemples d'entraînement utilisent ses orientations et variantes ; aucun équivalent par rotation ou renversement n'est présent dans l'évaluation. L'alphabet et la règle de tâche sont partagés.

Les graines **17, 29, 43** ont chacune reçu 1 200 mises à jour AdamW, batch 64, sans sélection du meilleur checkpoint. Les empreintes des paramètres initiaux et tirages de batches sont identiques entre les deux systèmes de chaque paire. Aucun entraînement défavorable n'a été retiré.

Le score primaire porte sur le **premier symbole**, sans passé correct fourni à l'inférence. L'AR génère séparément la séquence complète en quatre appels libres ; le parallèle en un appel. Les logits ultérieurs du premier appel AR ne sont pas traités comme une séquence libre. Le vocabulaire de sortie comprend les huit symboles de tâche.

## 2. Gel, séparation développement/évaluation et validité logicielle

Le gel local avant apprentissage date du **2026-10-25T15:38:19Z** et épingle 785 fichiers. La dose et les six checkpoints finaux sont scellés séparément le **2026-10-25T15:40:53Z**, avant l'évaluation. Ce sont des gels locaux, sans registre externe. Aucun amendement logiciel ni changement de critère n'a été nécessaire.

CPU FP32, un thread ; PyTorch 2.8.0 et NumPy 1.26.4. Les pertes et préférences sont calculées en FP64 à partir des logits. La collecte des évaluations dure 10,23 s ; l'entraînement total 136,38 s. Ces temps décrivent l'exécution locale et ne constituent pas un benchmark de vitesse entre architectures.

La coupure **après deux blocs est primaire** ; celle après un bloc est secondaire descriptive. L'état comprend sept positions de contexte et quatre positions de requêtes. Deux contrats s'appliquent aux mêmes poids :

- **R, contexte retenu** : positions de contexte du receveur et requêtes du donneur au point de coupure ; la continuation s'exécute normalement.
- **M, mémoire entière échangée** : tout l'état appartient au donneur ; sous **régénération**, le receveur ne fournit aucun tenseur au modèle. Sa référence ne sert qu'au score.

Dans M, une suite native figée du receveur reste un contrôle distinct : elle apporte précisément une information issue d'une passe antérieure du receveur. Il n'y a donc aucune contradiction à ce qu'elle produise sa réponse alors que la suite régénérée suit le donneur. « Chemin receveur bloqué » concerne la branche régénérée ; il ne signifie pas que toute politique de suite serait privée d'information du receveur. Voir la [note sur les contrats](CONTRATS_ET_ESTIMANDS.md).

## 3. Performances natives réservées

Pour **chacun des six modèles** : 40/40 premiers symboles exacts sur A, 40/40 sur B, 40/40 sur Aplus, et les mêmes comptes pour les **séquences complètes générées librement**. Chaque entraînement réussit donc 120/120 variantes. Ce sont les mêmes 40 familles pour les six modèles, pas 720 unités indépendantes.

Les échecs d'ancrage natif ne limitent donc plus cette étude, contrairement à la moitié des familles GPT-2 du pilote précédent. La tâche reste simple, synthétique et de même famille que l'entraînement : ce plafond ne démontre pas une généralisation à de nouvelles règles ou une capacité de raisonnement difficile.

## 4. Calibration des corruptions : résultat négatif conservé

Le développement teste les doses relatives 0,1 ; 0,3 ; 1 ; 3, sans recherche de directions favorables. Une dose commune devait provoquer, pour chaque système moyenné sur ses trois graines, une baisse d'exactitude figée de 10–40 points et un dommage moyen d'au moins 0,05 nat. **Aucune dose ne remplit ces conditions.**

À rho=3, sur développement :

- AR : dommage moyen 0,00164032 nat ; baisse de justesse nulle.
- Parallèle : dommage moyen 0,05683645 nat ; baisse de justesse de **0,278 point**, loin des 10 points exigés.

Le repli préfixé conserve rho=3 pour l'évaluation, avec statut `no_common_informative_dose_fallback`. La grille n'a pas été élargie après connaissance de ces résultats. Cette étude **n'établit donc pas une corruption informative commune calibrée**, même si la séparation des partitions et la procédure de sélection ont bien été appliquées.

## 5. C sur évaluation : régénérer peut aggraver la sortie

Les nombres suivants concernent la coupure primaire, dans l'ordre des graines 17,29,43. Chaque compte porte sur 320 essais d'un modèle : 40 familles × deux ancrages × quatre perturbations, regroupés par famille pour l'incertitude.

- AR : suites figées Y10 **320,320,320 /320** ; suites adaptées Y11 **237,267,210 /320**. Dommages moyens D : **0,001721 ; 0,000737 ; 0,005007 nat**. Récupérations relatives R : **−0,801597 ; −0,556550 ; −1,035251 nat**.
- Parallèle : Y10 **316,316,317 /320** ; Y11 **237,244,255 /320**. D : **0,058768 ; 0,072958 ; 0,069832 nat**. R : **−0,670127 ; −0,596030 ; −0,532856 nat**.

Les trois dommages moyens AR sont équivalents dans la marge fixée ±0,01 nat ; les trois dommages parallèles dépassent cette marge, sans fournir les pertes de justesse exigées sur développement. Les IC ajustés de R sont entièrement négatifs pour chacun des six modèles : ils décrivent ici une perte de NLL sous adaptation, pas une réparation. I et S sont aussi négatifs en moyenne aux six points primaires. Les valeurs complètes et leurs IC sont publiées par modèle dans le résumé, sans IC artificiel regroupant les trois graines comme des sources supplémentaires.

Aucun entraînement ne satisfait le critère commun de récupération utile et spécifique. La régénération peut amplifier une perturbation que le décodage avec une suite native antérieure supportait mieux. Cela ne prouve ni absence générale de compensation dans ces réseaux, ni mécanisme destructeur particulier. Rho=3 et les états croisés peuvent être hors distribution ; le résultat reste conditionnel à cette intervention et au repli de calibration.

Le bloc 1, secondaire, ne rétablit pas une récupération utile : les R moyens restent négatifs. Il n'a pas été choisi comme remplacement favorable du bloc primaire.

## 6. E primaire : suivre le donneur ne dépend pas seulement du type de réseau

Avec contexte retenu R, après deux blocs, les suites régénérées donnent **80/80 premiers symboles du donneur et 0/80 du receveur**, pour chacun des six modèles. Les deux références sont distinctes. Les suites natives figées du donneur donnent également 80/80 réponses du donneur.

En revanche, les suites natives figées du receveur donnent :

- AR : **80,80,80 /80** réponses du receveur pour les graines 17,29,43.
- Parallèle : **56,44,36 /80**, respectivement.

Les états initiaux sont les mêmes pour comparer les politiques de suite d'un échange. L'adaptation ne fournit ici aucun gain d'exactitude du receveur contre sa suite native figée ; elle en perd 80 pour chacun des AR, et 56,44,36 pour les parallèles. Une suite native a toutefois coûté une génération préalable ; aucune réponse gratuite ni accélération n'est démontrée.

Les diagnostics stricts conservent toutes les abstentions :

- AR, graines 17/29/43 : **20/25/27 familles dominées par l'état** et **20/15/13 non monotones**, sur 40 par entraînement.
- Parallèle : **18/17/28 dominées par l'état** et **22/23/12 non monotones**, sur 40.

Toutes les ancres natives sont correctes. Les abstentions proviennent ici de rapports u/g négatifs, parfois proches de zéro, pas d'un échec de tâche. Les effets continus et sorties du donneur ne doivent pas être effacés par ces étiquettes ; inversement, les rapports ne sont ni arrondis ni tronqués pour fabriquer une classe. Ces observations exposent une limite pratique d'un domaine de classification strict, sans le modifier après résultat. Une abstention ne prouve pas une complexité interne plus grande.

Dans les échanges A/Aplus, la référence est inchangée et les suites régénérées restent exactes en 80/80 cas pour les six modèles à cette coupure. Les variations de NLL et de probabilités sont conservées séparément, sans ratio entre références identiques.

## 7. E secondaire : profondeur et initialisation changent la dépendance

Après **un bloc**, avec contexte retenu, les réponses du receveur obtenues par régénération sont :

- AR, graines 17/29/43 : **0,80,0 /80**.
- Parallèle, mêmes graines : **0,40,80 /80**.

Tous les résultats des graines sont rapportés. L'AR 29 suit le receveur dans tous ces échanges précoces ; les deux autres AR suivent le donneur. Le parallèle 43 suit toujours le receveur ; le parallèle 29 a 40 réponses du receveur et 40 du donneur, et le parallèle 17 suit le donneur dans 78 cas avec deux autres sorties.

Les diagnostics secondaires correspondants sont :

- AR 17 : 35 états dominants et cinq non monotones ; AR 29 : 20 entrées dominantes et 20 non monotones ; AR 43 : 13 états dominants et 27 non monotones.
- Parallèle 17 : 20 états dominants, neuf interactions non additives et 11 non monotones ; parallèle 29 : 40 interactions non additives ; parallèle 43 : 37 entrées dominantes et trois non monotones.

Il ne faut pas appeler les 40 interactions du parallèle 29 un mélange causal identifié : le critère d'additivité échoue précisément. Chez AR 29 et parallèle 43, les mêmes poids passent d'une réponse régénérée suivant le receveur à la coupure précoce à une réponse suivant le donneur à la coupure primaire. Cela établit une dépendance au lieu d'intervention, sans identifier un algorithme interne ni une transition cognitive.

La suite native figée du receveur atteint déjà 80/80 bonnes réponses pour les six modèles à la coupure précoce. Les succès de régénération du receveur ne démontrent donc pas sa nécessité pour l'exactitude dans ces cas.

## 8. M : un contrôle de chemin absent, pas une découverte mécaniste

Sous mémoire entière transférée, les sorties régénérées sont **exactement les logits natifs du donneur**, pour chaque architecture, graine, coupure et combinaison receveur/donneur. L'effet du receveur u et l'interaction w sont exactement nuls. Le diagnostic retourne 40/40 états dominants dans chaque cellule, attendu sous les ancres correctes.

Cela vérifie un résultat structurel connu sur des **réseaux réellement entraînés**, sans exiger un oracle parfait dans leur continuation. Ce n'est pas une précision de 100 % d'identification d'un programme interne. Le canal a été supprimé par l'expérimentateur ; la conclusion porte sur ce contrat. Les politiques de suite figée restent des interventions distinctes qui peuvent réinjecter de l'information du receveur.

La comparaison R/M est particulièrement visible pour AR 29 et parallèle 43 à la coupure précoce : les suites régénérées suivent le receveur dans R et le donneur dans M. Ces exemples sont des parties de l'ensemble annoncé, pas une sélection de modèles remplaçant les autres. L'ensemble des graines montre que cette différence n'est pas universelle dans R.

## 9. Audit, portée et suite

L'audit gelé passe : **42 240 recompositions exactes**, **576 régénérations par équations explicites**, **19 200 pertes scalaires**, **72 reprises natives** et **3 840 conditions de chemin bloqué**. Ces comptes incluent variantes, graines, coupures et conditions répétées ; ils ne sont pas des effectifs statistiques indépendants. Les partitions, initialisations et batches appariés, checkpoints, sélection de dose, contrastes, intervalles C, diagnostics, exactitudes et gains/pertes sont vérifiés. Les fichiers gelés et manifestes NCM, GPT-2 et article précédent sont inchangés. Audit par le même agent, pas réplication extérieure.

L'étude améliore le benchmark sur quatre points : tâche commune, entraînements répétés, frontière développement/évaluation et contrats d'accès explicites. Elle conserve trois limites substantielles : **échec du critère de calibration**, **sensibilité au point de coupure et à la graine**, **nombreuses abstentions de la règle stricte**. Ces résultats limitent les promesses du dispositif et rendent ses usages mieux définis.

La validité externe reste ouverte : tâche synthétique facile, petit alphabet et une seule règle, trois graines, architectures miniatures, pas de NCM fidèle entraîné à tâche commune, ni méthode d'attribution proposée par une équipe indépendante. L'identité des symboles et l'algorithme de tâche sont partagés entre partitions ; pas de généralisation à des familles de raisonnement entièrement nouvelles. Les modèles ne sont pas optimisés pour rendre les diagnostics corrects.

Une campagne suivante devrait traiter la calibration comme un objet scientifique à part entière : comparer sur développement distinct une corruption naturelle d'état et une corruption destinée à modifier la lecture figée, puis réserver une autre évaluation avant toute inférence. Le coût et les informations utilisés pour construire la perturbation devront être déclarés. Il faudra aussi tester des règles d'abstention calibrées contre les effets continus, et des méthodes d'attribution développées indépendamment. Aucun de ces ajustements n'est appliqué aux résultats présents.

Le lien avec Wyner reste théorique et conditionnel : ces réseaux déterministes et ces interventions ne mesurent ni information commune minimale ni compensation du pooling. L'apport de la campagne concerne l'attribution fonctionnelle et son protocole, pas une nouvelle preuve sur NCM.

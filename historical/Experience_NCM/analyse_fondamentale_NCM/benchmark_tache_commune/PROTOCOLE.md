# Tâche commune, entraînements répétés et contrats d'accès

25 octobre 2026. Nouvelle campagne avant entraînement et sorties. Rapports Markdown uniquement. Aucune campagne antérieure n'est modifiée. Le but est de contrôler davantage les facteurs qui empêchaient d'interpréter la comparaison NCM/Sudoku–GPT-2/texte.

## 1. Questions et limites annoncées

Comparer sur une même tâche un petit Transformer autorégressif et un petit Transformer parallèle, avec trois graines, puis modifier le contrat d'accès **sur les mêmes poids**. Mesurer séparément le fonctionnement des contrôles, les effets fonctionnels et les abstentions. Aucun modèle ne reçoit une vérité de mécanisme cognitif. Le parallèle construit ici n'est pas un NCM : pas de VQ-VAE, tokenizer appris ou hiérarchie de concepts.

Le contraste d'accès fermé a une vérité **structurelle limitée** : lorsque toute la mémoire du donneur est transférée et qu'aucun autre canal d'entrée n'est accessible, changer l'identité du receveur ne peut pas changer la sortie. C'est un contrôle de chemin absent, attendu par construction, pas la découverte empirique que le modèle « ignore » une entrée qu'on lui a retirée.

L'étude compare deux systèmes d'entraînement spécifiés, pas l'architecture seule : l'AR apprend sous teacher forcing et le parallèle sous requêtes fixes. Ils partagent tâche, partitions, nombre de paramètres, initialisations appariées, exemples et budget d'optimisation. L'efficacité d'entraînement et les informations disponibles aux positions ultérieures diffèrent ; l'horizon primaire est le **premier symbole** pour éviter de donner son passé correct à un modèle mais pas à l'autre lors des interventions.

## 2. Tâche et partitions

Une séquence de quatre symboles distincts parmi huit est donnée, avec un bit b et un bit de nuisance n. Réponse de longueur quatre : rotation à gauche de b positions, b dans {0,1}. n ne change pas la référence. L'entrée contient `[BOS, b, n, x0,x1,x2,x3]` ; quatre requêtes suivent. A=(b=0,n=0), B=(b=1,n=0), Aplus=(b=0,n=1). A/B diffèrent d'un token ; leur premier symbole cible diffère toujours. Aplus conserve longueur et réponse. Le score primaire est la NLL et l'argmax sur les huit symboles, sans choix forcé A/B. L'exactitude de la séquence complète est secondaire.

Les 1 680 permutations de quatre symboles parmi huit sont regroupées par rotations et renversements : 210 orbites de huit séquences. Graine de partition fixe : 2026102601. Répartir **140 orbites entraînement, 30 développement, 40 évaluation**, sans intersection. Toutes les orientations et variantes d'une orbite restent ensemble. Entraînement : huit orientations × deux b × deux n, soit 4 480 exemples. Développement/évaluation : une orientation tirée avant apprentissage et trois variantes par orbite. Les données sont générées avant gel ; aucune sélection sur les performances. L'identité des symboles est partagée entre partitions : ce n'est pas une généralisation à un nouvel alphabet ou à une nouvelle tâche.

L'évaluation n'est pas consultée avant gel de la dose de corruption choisie sur développement. Les 40 orbites, pas les variantes, signes ou tokens, sont les unités de regroupement. Les trois graines sont trois entraînements, pas 120 familles indépendantes.

## 3. Modèles et apprentissage fixé

Deux Transformers pré-LayerNorm, quatre blocs, largeur 48, quatre têtes, MLP 96, GELU, positions apprises, vocabulaire d'entrée 15, sortie 8 ; dropout nul. Préfixe causal dans les deux modèles et sans accès aux requêtes. L'AR rend les requêtes causales et utilise `[START,y0,y1,y2]` en entraînement. Le parallèle utilise `[START,MASK,MASK,MASK]` et permet l'attention entre toutes les requêtes. Paramètres identiques en nombre et initialisation pour chaque paire de graines **17,29,43**.

AdamW, lr maximale 0,002, beta=(0,9;0,99), eps=1e-8, weight decay 0,01, clip gradient 1 ; 1 200 mises à jour, batch 64, 50 pas de montée linéaire puis cosinus jusqu'à 10 % de la lr maximale. Mêmes tirages de batches pour une paire AR/parallèle. CPU FP32, un thread, graine fixe, aucune recherche d'hyperparamètres. Garder le checkpoint **final**, sans choix sur validation. Aucun arrêt anticipé sur score, aucun remplacement de graine défavorable. Consigner loss et empreintes initiales/finales ; les références n'alimentent jamais l'inférence d'évaluation.

L'AR génère la réponse entière librement en quatre appels ; le parallèle en un appel. Le premier appel AR reçoit les mêmes tokens de requêtes fixes que le parallèle : aucune réponse vraie n'y est fournie. Les autres logits de cet appel AR ne sont pas des prédictions de séquence libre ; seule la première position est évaluée dans O/C/E. Rapporter séparément les exactitudes des séquences générées librement.

## 4. États et accès comparés

Deux coupures préfixées : après le **bloc 2, primaire**, et le **bloc 1, secondaire descriptif**. Le lecteur final est LayerNorm + projection sur huit symboles. O conserve les lectures du premier symbole après embeddings et quatre blocs ; elles ne sont pas des tokens émis.

Au point de coupure, H comprend sept positions de contexte et quatre de requêtes. La continuation conserve séparément les sorties d'attention et MLP de tous les blocs restants, sans différence de tenseurs pour fabriquer les résidus. Les ajouts sont séquentiels FP32.

- **Accès R, contexte retenu** : H initial combine les sept positions du receveur et les quatre requêtes du donneur. Les blocs restants s'exécutent normalement ; le préfixe causal ne dépend pas des requêtes. Le canal d'entrée est donc la représentation du préfixe au point de coupure, pas une lecture oracle de la réponse.
- **Accès M, mémoire entière échangée** : H entier est celui du donneur. Le nom du receveur ne fournit aucun tenseur au réseau. Sa référence sert seulement au score. Sous régénération, la sortie doit être exactement native du donneur. Les suites natives receveur/donneur restent toutes deux testées.

Ces contrats ont les mêmes poids et sources. Le contraste M sert à falsifier une attribution au receveur quand son chemin a été bloqué. Il ne permet pas d'inférer que le modèle appris serait naturellement dépourvu de dépendance à l'entrée. Avec état et suite figés, le lecteur n'a aucun accès au receveur dans les deux contrats.

## 5. Calibration séparée des corruptions

C perturbe seulement les quatre requêtes, sur deux supports disjoints (moitiés des coordonnées de caractéristiques), deux signes. Directions gaussiennes fixées avant sorties, même norme par ancrage : rho × ||H_requêtes||. Graines distinctes par partition, identiques entre modèles pour les directions avant renormalisation. Doses candidates **0,1 ; 0,3 ; 1 ; 3**, toutes conservées sur développement. Aucun choix de direction sur les dégâts observés.

Sélectionner une **dose commune aux deux modèles**, au bloc 2 : la plus faible dont, pour chaque architecture sur ses trois graines et toutes les sources/variantes A/B/signes de développement, la perte moyenne d'exactitude sous suite figée est entre 10 et 40 points et le dommage NLL moyen au moins 0,05 nat. Si aucune dose ne convient, conserver rho=3, annoncer l'échec de calibration et ne pas présenter le test comme calibré. Les performances natives de développement et les résultats de toutes les graines restent publiés ; aucune graine n'est retirée. Rapporter si chaque modèle atteint 85 % de justesse native au premier symbole ; ce seuil décrit la faisabilité, sans arrêter ou régler l'apprentissage.

Sceller `calibration.json` et les checkpoints **avant** d'ouvrir les sorties d'évaluation. Appliquer la dose retenue aux deux profondeurs et aux 40 familles d'évaluation sans sélection sur dommage individuel. Aucun réajustement de dose, coupe ou budget ensuite. Les directions sont continues et peuvent être hors distribution. C est identique entre R et M à ancrage natif fixé ; ne pas compter deux fois ce test.

## 6. Évaluation et statistiques

C conserve Y00,Y10,Y11,Y01,Y1wrong, les deltas, états finaux, suites et logits. Tother utilise l'autre support avec même signe/norme. D=L10−L00, R=L10−L11, I=L10+L01−L11−L00, S=L1wrong−L11. Moyenne sur deux ancrages et quatre perturbations par famille. Pour **chaque architecture, graine et coupure**, bootstrap de ses 40 familles, 20 000 tirages : IC descriptifs 95 % et 98,75 % pour les quatre critères primaires au bloc 2, marge 0,01 nat. Utilité et spécificité seulement si les quatre bornes inférieures ajustées dépassent la marge. Le bloc 1 est secondaire ; aucune sélection entre les coupes ni revendication confirmatoire de ce seul choix. Pas de p-value ni de pooling des graines comme observations indépendantes.

Conserver justesse, dommages et gains appariés, même si NLL et argmax divergent. Les moyennes globales sur les graines sont descriptives ; rapporter séparément leurs valeurs et la réplication ou non du signe. Ne pas généraliser à une population d'entraînements sur trois graines.

E : pour chaque coupure, contrat R/M et comparaison A/B ou A/Aplus, exécuter 2 receveurs × 2 donneurs × trois suites (native gauche, native droite, régénérée). Rapporter exactitude receveur/donneur/autres et comparaison appariée régénérée versus suite native du receveur. Pour A/B, calculer u,v,w,g sur log p(cibleB)−log p(cibleA), mêmes formules et seuils que les études précédentes : deux ancres exactes, g>0,1, |w/g|≤0,2, ratio dans [0,1], entrée si u/g≥0,8, état si≤0,2, sinon deux sensibilités/mécanisme non modélisé. Garder les motifs d'abstention et toutes les sources dans les effets moyens. Ces seuils ne fournissent pas une vérité de mécanisme interne.

Le contrat M doit donner u=0, w=0 et sortie régénérée native du donneur, à égalité numérique exacte pour les logits croisés. Une violation est une défaillance de l'adaptateur et entraîne l'arrêt, pas une découverte de sensibilité cachée. Sur R, aucun diagnostic n'est annoncé d'avance. Les effets sur Aplus sont rapportés sans diviser par une différence de références nulle.

## 7. Gel, audit et portée

Avant apprentissage, épingler ce protocole, paramètres, corpus, scripts de modèle/entraînement/calibration/évaluation/analyse/audit et sources importées. Gel local sans registre externe. Une erreur logicielle après gel conserve code et observations fautifs ; toute correction exige un amendement horodaté, sans modification silencieuse des critères. Le gel de calibration constitue une seconde frontière avant évaluation.

Audit : partitions et orbites, initialisations appariées, checkpoints finaux, parité forward/reprise/sham, toutes les recompositions résiduelles, normes, pertes et diagnostics, invariances M et sorties natives donneur. Rejouer par appels explicites attention/MLP toutes les branches C et E des deux premières sources pour chaque graine, architecture et coupe. Le même agent réalise l'audit ; pas de réplication externe. Les anciens manifestes restent vérifiés en lecture seule.

L'apport attendu est une comparaison de systèmes appris à tâche commune, une calibration indépendante des corruptions et un contrôle de contrat d'accès. Il ne remplace pas des méthodes d'attribution développées indépendamment, une validation extérieure, un modèle NCM miniature fidèle ou des données naturelles. Pas d'optimisation par les diagnostics, pas de preuve de métacognition, pas de nouvelle mesure de Wyner. Arrêt au rapport et au complément du papier après la campagne, y compris en cas d'échec scientifique.

Positionnement : [Geiger et al., NeurIPS 2021](https://papers.nips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html) utilisent des interventions d'échange pour tester des abstractions causales. [Zhang et Nanda, ICLR 2024](https://openreview.net/pdf?id=Hf17y6u9BC) étudient la sensibilité aux métriques et corruptions. Ces antécédents motivent l'explicitation des accès et de la calibration, sans revendication de priorité. Sources primaires consultées le 25 octobre 2026.

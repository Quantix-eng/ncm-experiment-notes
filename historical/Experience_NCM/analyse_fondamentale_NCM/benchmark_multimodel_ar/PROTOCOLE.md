# Extension autorégressive : attribution d'une continuation interne dans GPT-2

25 octobre 2026. Nouvelle campagne, avant sorties du modèle. Rapports Markdown uniquement. Les études NCM et le manuscrit antérieur restent inchangés. L'objectif est d'élargir les architectures et les contrats d'accès du protocole, sans imposer une signature à une famille de modèles.

## 1. Question et hypothèses

Le protocole O/C/E peut-il être exécuté sur un Transformer autorégressif appris, et ses contrôles empêchent-ils les mêmes erreurs d'attribution ? L'hypothèse « AR = suit la suite figée, NCM = entrée, diffusion = mélange » n'est pas utilisée pour étiqueter les observations. Aucune diffusion n'est exécutée ici. `frozen_recipient_native` est une **politique expérimentale** disponible pour tout adaptateur, pas une classe de mécanisme. L'impossibilité, dans un décodage append-only ordinaire, de réécrire les tokens déjà émis est une propriété de l'interface, pas une découverte d'attribution.

Cette campagne intervient **à l'intérieur du calcul du prochain token**, entre couches. Elle ne mesure pas la révision de tokens déjà émis et ne transpose pas des couches GPT-2 aux échelles NCM comme si elles représentaient le même temps. Une sensibilité interne au contexte est compatible avec une génération AR. L'étiquette descriptive et le résultat négatif/non identifiable sont admissibles sans ajustement.

## 2. Modèle et sources

GPT-2 small, `openai-community/gpt2`, révision indiquée dans `artifacts/gpt2/download_metadata.json`, poids safetensors téléchargés avant gel. Modèle préentraîné sans apprentissage supplémentaire ; CPU FP32, un thread, lot un, mode évaluation, attention eager, sans cache KV mutable. Transformers 4.57.1 et safetensors 0.6.2 sont installés dans `deps`, sans modifier l'environnement historique. NumPy/PyTorch sont importés depuis l'environnement existant avant ces dépendances. Épingler versions effectives, poids et sources importées.

Sources primaires consultées le 25 octobre 2026 : [fiche GPT-2](https://huggingface.co/openai-community/gpt2), [documentation GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2), et le fichier local `deps/transformers/models/gpt2/modeling_gpt2.py`. Le modèle est un Transformer causal ; les dépendances et la réalisation numérique exécutées sont celles gelées localement.

## 3. Tâche et corpus avant inférence

Seize paires de mots fixées dans `prepare.py` : couleurs, animaux, aliments, saisons et directions. Elles sont choisies par l'auteur, non aléatoirement ; aucune prétention à représenter un corpus textuel naturel ou indépendant de l'entraînement. Les prompts sont `The code is WORD. The code is`. La tâche d'évaluation est de restituer le mot explicitement assigné ; sa référence est sa tokenisation avec espace initial. L'argmax sur **tout** le vocabulaire sera évalué, sans réduire le choix aux deux références.

Pour chaque famille, A/B changent un seul token du mot assigné. Aplus remplace le premier `The` par `My`, conserve la réponse et la longueur. Le dernier token du prompt est identique. Vérifier avant gel, uniquement avec le tokenizer, les longueurs identiques, les références à un token et l'unique différence A/B. Ne jamais sélectionner ou remplacer une source sur les réponses de GPT-2. Toutes les 48 entrées et tous les échecs sont conservés.

Il s'agit d'une **nouvelle tâche**, pas des 16 Sudoku. Les familles textuelles ne sont pas appariées aux familles Sudoku ; aucune différence de score ne sera attribuée causalement à l'architecture seule. Pas de développement sur des sorties GPT-2 dans cette campagne et pas d'évaluation externe réservée : c'est une extension de portée interne avec adaptation préfixée.

## 4. État, contexte et continuation

Couper après le bloc 6 sur 12, sans recherche du bloc favorable. L'état modifiable z est le vecteur résiduel de dimension 768 du **dernier token** à ce point. Le contexte x est constitué des représentations natives de toutes les positions antérieures à l'entrée de chaque bloc restant. Il appartient au receveur. Les blocs causaux garantissent que ces positions ne dépendent pas du dernier token.

Pour reprendre, former à chaque bloc 7..12 la séquence avec positions antérieures natives du receveur et dernier vecteur courant. Exécuter le bloc complet. La continuation T conserve séparément la sortie d'attention et la sortie MLP du dernier token, soit 12 contributions pour six blocs. Éviter la soustraction « sortie moins entrée » et toute réassociation des résidus : les ajouts natifs FP32 sont rejoués dans l'ordre. La lecture est `lm_head(ln_f(z_final))`, appliquée avec la même forme matricielle que le chemin natif pour le contrôle bit à bit.

L'échange porte uniquement sur z, **pas sur l'ensemble de la mémoire du prompt**. Les autres positions des couches restantes sont un canal conservé vers l'entrée. Cet accès doit apparaître dans toute interprétation : échanger toute la mémoire supprimerait ce canal et poserait une autre question. Aucun cache donneur n'est transféré silencieusement.

Exiger sur les 48 entrées la parité exacte entre lecture native officielle, reconstruction par contributions natives et reprise sans perturbation. Les positions antérieures recalculées par l'adaptateur doivent rester natives, y compris sous intervention. Les sorties avec z et T fixés doivent être invariantes au contexte receveur. Arrêt sur une défaillance ; toute correction post-gel conserve les versions fautives et exige un amendement horodaté.

## 5. O et C

O conserve les 13 lectures : embeddings puis sorties de 12 blocs, avec le même lecteur final. Ce lecteur n'est pas un logit lens entraîné. Compter justesse, corrections et dégradations sans attribution mécaniste. Les sorties intermédiaires ne sont pas des tokens que GPT-2 a déjà émis.

C utilise A et B, soit 32 ancrages. Deux supports disjoints sont les deux moitiés de 384 coordonnées du vecteur. Directions gaussiennes indépendantes, graine 20261025, deux signes par direction. Norme de chaque delta : 0,1 × ||z||, mêmes normes pour les deux supports dans un ancrage. Le choix est exploratoire et **non calibré sur les réponses** ; aucun changement de dose ou de couche après examen. Les vecteurs peuvent être hors distribution. Contrôler la norme à tolérance relative 2e-6.

Conserver les cinq conditions Y00, Y10, Y11, Y01, Y1wrong du protocole : suite native ou adaptée, état natif ou perturbé ; suite mal appariée adaptée à l'autre support de même signe et norme. Sauver deltas, états finaux, contributions et logits complets. La perte est la NLL du mot cible sur un token, FP64 à partir des logits FP32. Rapporter aussi la justesse argmax globale.

Contrastes identiques : D=L10−L00, R=L10−L11, I=L10+L01−L11−L00, S=L1wrong−L11. Moyenne d'abord sur les quatre perturbations et deux ancrages par famille, puis sur les 16 familles. Bootstrap de familles entières, 20 000 tirages, IC descriptifs 95 % et ajustés 98,75 % pour les quatre contrastes. Marge exploratoire 0,01 nat/token. Utilité/spécificité seulement si les quatre bornes inférieures ajustées dépassent cette marge ; dommage moyen équivalent si tout son IC ajusté est dans ±0,01. Sinon résultat insuffisant ou non confirmé. Le mot « réparation » exige aussi de regarder le dommage d'exactitude, sans sélectionner uniquement les cas effectivement abîmés.

## 6. E et politiques de suite

Deux comparaisons : A/B et A/Aplus. Chacune exécute 2 contextes receveurs × 2 états donneurs × 3 suites (native gauche, native droite, régénérée), soit 12 conditions, 384 au total. Les mêmes z sont utilisés pour comparer les trois politiques. Conserver les diagonales et une lecture sans toute la suite, déjà présente dans O.

Pour A/B, h_ij=log p(mot_B)−log p(mot_A). Calculer u, v, w, g avec les formules historiques :

- u=0,5[(h_BA−h_AA)+(h_BB−h_AB)] ;
- v=0,5[(h_AB−h_AA)+(h_BB−h_BA)] ;
- w=h_BB−h_BA−h_AB+h_AA ; g=h_BB−h_AA.

Appliquer **sans changement** le diagnostic descriptif historique : deux sorties natives globalement exactes, g>0,1, |w/g|≤0,2, rapport u/g dans [0,1] ; entrée si rapport≥0,8, état si≤0,2, sinon deux sensibilités/mécanisme non modélisé. Déclarer non identifiable si une condition échoue et conserver les effets de toutes les familles. Le rapport u+v=g est une identité, pas un pourcentage causal. Ces seuils hérités ne sont pas validés pour le texte. Rapporter les motifs d'abstention.

Aligner séparément suite native du receveur et du donneur dans les échanges hors diagonale ; compter référence receveur, référence donneur et autres argmax. Les suites ne sont pas les mots imposés : ce sont les **contributions résiduelles continues** déjà calculées par le réseau. Les résultats ne sont pas forcés par une injection de tokens de sortie. Une suite figée peut demeurer suffisante sur plusieurs états sans que l'architecture entière appartienne à une classe « suite figée ».

Pour A/Aplus, pas de rapport de préférence entre références identiques ; rapporter NLL, changements d'argmax et variation totale des distributions. Pas de classement architecture AR/NCM par ces tâches différentes, pas de conclusion sur une diffusion non exécutée.

## 7. Gel, audit et arrêt

Sceller avant nouvelle inférence protocole, paramètres, familles, tokenizer, poids, scripts, principales sources importées et entrées historiques de comparaison. Gel local sans registre externe. Toute défaillance logicielle post-gel suspend le runner, conserve les sorties partielles et demande un amendement documenté avant reprise. Aucun résultat scientifique ne justifie un réglage rétroactif.

Audit distinct : recomptage des pertes en log-sum-exp scalaire, des contrats d'exactitude et de préférence, contrôle des deltas, des croisements, des diagonales et invariances, puis reprise de toutes les branches C et E des deux premières familles avec appels explicites aux sous-couches attention/MLP, sans utiliser le chemin `block.forward` de l'adaptateur. Exiger les mêmes états et logits. Refaire un contrôle natif officiel sur ces sources. Un seul agent programme et audite ; ces vérifications ne sont ni réplication externe ni vérité de mécanisme.

Arrêt au rapport Markdown après cette extension. Produire un complément au papier dans ce nouveau dossier, sans réécrire les résultats gelés de l'article précédent. Conserver l'éventuel échec de l'hypothèse AR proposée.

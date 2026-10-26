# Pythia-160M : lecture finale et continuation neuronale

Rapport du 25 octobre 2026. Mise en œuvre de la proposition 3 du directeur, suivant la [feuille de route](FEUILLE_DE_ROUTE.md). Les résultats ci-dessous concernent une extension sur WikiText-2 ; aucune campagne historique n'est remplacée. La revue par le sous-agent constitue un contrôle interne, sans indépendance humaine ou validation extérieure.

## État de la campagne

Les **128 séquences de développement et les 512 séquences réservées sont exécutées et auditées**. Sur le test, le lecteur au facteur propre à la corruption ne gagne aucune prédiction correcte face au facteur natif figé : **598/2 048 essais exacts dans les deux cas, zéro gain et zéro perte appariés**. Le self-repair de logit et la récupération de NLL restent faibles et variables ; leurs moyennes ne démontrent pas une récupération positive. Le protocole conserve la distinction entre variation du lecteur et adaptation de blocs aval, dont le chemin est absent dans ce graphe.

La première version du développement a échoué à son contrôle de parité. Les corrections [1.1](AMENDEMENT_LOGICIEL.md) et [1.2](AMENDEMENT_FP64.md) sont archivées et gelées avant test, sans relèvement des seuils. Les contrôles techniques passent sur développement et test. L'audit FP64 de développement utilise le script gelé ; l'audit FP64 de test utilise une implémentation supplémentaire par lecture des poids en mémoire mappée, ajoutée après résultats pour éviter le chargement concurrent de deux modèles. Cette différence d'audit est conservée explicitement.

## Correction méthodologique effectivement appliquée

L'objet testé est la distinction entre un lecteur dont le facteur de normalisation change et une continuation neuronale dont les contributions s'adaptent. Les deux objets reçoivent des noms et des métriques distincts. Le lecteur final est inclus dans `D` ; une différence obtenue seulement en changeant son facteur ne constitue pas une preuve d'adaptation des blocs.

Le [graphe officiel GPT-NeoX, version importée 4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/gpt_neox/modeling_gpt_neox.py) calcule la dernière couche comme `h_final = MLP(LN2(h)) + attention(LN1(h)) + h`, puisque `use_parallel_residual=true`. Une intervention après calcul d'une tête finale ne modifie pas l'entrée du MLP parallèle, ni les autres têtes. Aucune couche d'attention ou MLP ne suit cette dernière couche. Le chemin de compensation neuronale descendante visé est donc absent dans ce contrat ; la LayerNorm finale demeure une opération de lecture susceptible de varier.

Dans [Rushing et Nanda, §3.2](https://arxiv.org/html/2402.15390v2#S3.SS2), le texte cite L11H2 et L11H10, alors que la légende de la figure 5 mentionne L11H0. Les cibles du directeur, H2 et H10, sont conservées avant mesure ; H0 n'est pas ajouté pour rechercher un résultat favorable. La métrique reprise est la différence du logit cible centré moins la différence d'effet direct de la tête, avec le facteur de normalisation natif. Cette reprise ne revendique pas une invention du contrôle de normalisation.

## Ressources et partitions

Checkpoint officiel [EleutherAI/pythia-160m](https://huggingface.co/EleutherAI/pythia-160m), révision `50f5173d932e8e61f858120bcb800b97af589f46`. Le fichier `model.safetensors` fait 374 998 696 octets ; son SHA-256 correspond à celui de l'API officielle : `29d2e457a664e41c12c735f20a36dc0956a665f614a54ce5db21a32e75965270`. Le modèle chargé possède 162 322 944 paramètres. Les poids stockés sont convertis en FP32 pour le calcul CPU ; aucun apprentissage n'est exécuté.

Les données sont les fichiers Parquet officiels [Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext), configuration `wikitext-2-raw-v1`, partitions validation et test ; leurs empreintes sont conservées. Les expériences de l'article utilisaient The Pile : changer de corpus et ne retenir qu'une position par séquence interdit une comparaison directe avec ses pourcentages.

Parmi 1 602 lignes admissibles de validation et 1 797 de test après retrait des textes complets dupliqués, le tirage fixe 128 et 512 lignes, respectivement. Une ligne doit fournir au moins 33 tokens : les 32 premiers constituent le contexte, le 33e est la référence. Chaque ligne fournit deux donneurs distincts, fixés par indices circulaires `i+1` et `i+17` dans sa partition. Chaque corruption remplace le même site, la tête choisie à la position finale. Aucun choix ne dépend de la justesse ou du dommage.

Un audit supplémentaire de données relève zéro fenêtre de 33 tokens dupliquée à l'intérieur des sélections et zéro fenêtre partagée entre elles. Les lignes peuvent provenir d'un même article ; elles ne constituent pas des familles documentaires indépendantes. Les donneurs partagés introduisent également une dépendance entre essais.

## Conditions et mesures

Pour chaque tête et chaque donneur, le résidu natif `r0` et le résidu corrompu `rd` sont croisés avec trois facteurs de normalisation : natif `s0`, propre à la corruption `sd`, propre à l'autre donneur `so`. Les cinq sorties sont `D_s0(r0)`, `D_s0(rd)`, `D_sd(rd)`, `D_sd(r0)` et `D_so(rd)`. Le centrage du résidu est recalculé dans chaque condition ; seul le dénominateur est figé ou échangé.

Les contrastes NLL sont `D=L10−L00`, `R=L10−L11`, `I=L10+L01−L11−L00`, `S=L1wrong−L11`. Ici, `R`, `I` et `S` sont explicitement **des contrastes du lecteur**. Leurs analogues d'adaptation d'une continuation attention/MLP après la tête finale sont nuls par le graphe. Des signes positifs de récupération ou de spécificité d'échelle ne doivent donc pas entraîner une attribution à des blocs aval.

Les sorties conservent les pertes en FP64 calculées sur les logits FP32, les argmax sur tout le vocabulaire, les logits cibles centrés, les effets directs et les facteurs. Les cas défavorables sont inclus. Aucun filtrage des 2 % de positions à fort effet, aucun clipping de ratio et aucune sélection de tête supplémentaire ne sont effectués. Les résultats sont détaillés par tête et séquence. Les IC percentiles prévus rééchantillonnent 10 000 fois les séquences entières, donneurs et têtes conservés ensemble ; ils restent descriptifs et ne traitent pas la dépendance documentaire ou celle des donneurs.

## Gel, incident et correction conservés

Le premier gel local date du **25 octobre 2026 à 17:44:51 UTC**, avant tout forward du modèle dans cette campagne. Le premier développement produit 512 corruptions, soit 128 séquences × 2 têtes × 2 donneurs. Son temps chronométré est **58,58 s**, chargement et calcul des métriques compris, avant écriture des archives compressées. Ce temps n'est pas un benchmark de performance isolé : d'autres travaux se déroulent sur la machine.

Le contrôle échoue parce que la lecture manuelle FP32 atteint un écart natif maximal de **0,000244140625 logit**, au-dessus du seuil gelé de **0,0002**. L'écart maximal du rejeu corrompu est 0,0001220703125 ; le résidu natif recomposé est exact et le changement du MLP final vaut zéro sur les huit corruptions rejouées. L'audit distinct retrouve les 160 premières pertes exactement, ce qui ne répare pas l'échec de parité envers le modèle natif.

Le seuil n'est pas relevé. Les sources et observations sont archivées dans `archives/v1_native_parity_failure_20261025T174632Z/`. Le deuxième gel, **17:49:25 UTC**, remplace le lecteur manuel par le véritable noyau LayerNorm assorti d'un rapport d'échelle ; l'expression réelle du lecteur reste identique. L'audit distinct utilise une formule directe FP64. Les seuils, entrées, donneurs, métriques et têtes restent inchangés. La connaissance de ce premier développement est explicitement conservée ; ce développement n'est pas un test aveugle intact après correction.

## Développement après amendement : résultat disponible

Le rejeu 1.1 est achevé à **17:51:58 UTC**, en **117,59 s** selon le même périmètre. Il réussit les critères techniques inchangés : écart du lecteur natif **0**, écart maximal de rejeu corrompu **0,0001220703125**, erreur du résidu natif **0**, changement du MLP final **0**, écart maximal entre self-repair et effet de lecteur **0,0001471673**, inférieur à 0,0002. Huit corruptions ont été rejouées par le modèle complet. Le contrôle des JSON recalcule 2 048 contrastes scalaires, les cinq comptes d'exactitude et toutes les empreintes gelées. Il est distinct de l'audit FP64 décrit ci-dessous.

Sur les 128 séquences, 34 tokens suivants sont prédits exactement, soit 26,56 %. Les 512 essais répliquent quatre fois cette observation native : 136/512 prédictions exactes en Y00, 137/512 en Y10, 137/512 en Y11, 136/512 en Y01 et 137/512 avec l'autre facteur. Les effectifs 512 ne sont pas 512 séquences indépendantes. Aucun avantage d'exactitude d'une normalisation adaptée n'apparaît dans cette collection de développement.

Les moyennes, sans filtrage favorable, sont :

- Dommage NLL `D` : **0,00538727 nat**, IC descriptif 95 % [-0,00262985 ; 0,01262771].
- Récupération NLL du lecteur `R` : **0,000193031 nat**, IC [-0,00113207 ; 0,00140529].
- Interaction de lecteur `I` : **0,0000173970 nat**, IC [-0,000365242 ; 0,000259526].
- Spécificité de facteur `S` : **-0,000151234 nat**, IC [-0,000510931 ; 0,0000457204].
- Différence d'effet direct cible : **-0,0306665 logit centré** ; self-repair brut : **0,00363434**, IC [-0,00952175 ; 0,01685435].
- Effet du seul lecteur sur le logit centré : **0,00363625**, IC [-0,00952087 ; 0,01685961].
- Rapport moyen des facteurs natif/corrompu : **1,00021353** ; 257/512 rapports sont supérieurs à un.

La tête H2 a un dommage moyen de 0,000607012 nat et une récupération de lecteur négative de -0,000257306 nat ; H10 a un dommage de 0,01016753 nat et une récupération de lecteur de 0,000643367 nat. Cette variation ne déclenche aucun remplacement de tête. Les moyennes globales faibles et leurs intervalles ne démontrent pas une récupération générale ; les signes individuels ne sont pas transformés en taux de mécanismes corrects.

Le développement confirme une identité fonctionnelle et des effets de lecteur variables, sous la limite numérique annoncée. Il ne reproduit pas les forts effets conditionnels de l'article qui sélectionnait des positions à grand effet direct. Ce contraste de sélection était fixé avant exécution et demeure conservé.

## Sécurisation de la mesure et ouverture du test

L'amendement 1.2 est gelé le **25 octobre 2026 à 17:59:13 UTC**, avant tout forward de test. Il fait suite à des valeurs non finies dans une petite projection de mesure d'un pilote IOI exécuté en parallèle, dont la cause exacte reste inconnue et dépend des conditions. Aucun résultat non fini n'est observé dans Pythia : 8 896 512 valeurs sauvegardées et 512 mesures d'effet direct sont finies.

Seule la mesure d'effet direct passe en FP64, avec projection contiguë et centrage de l'unembedding en FP64. Le modèle et le lecteur expérimental restent FP32. Le recalcul du développement utilise les poids et états conservés, sans forward : changement maximal de l'effet direct **1,58e-7 logit**, moyenne signée **-4,88e-10**, aucune autre observation modifiée. Son coût est **0,43 s**. Les chiffres arrondis du développement ci-dessus restent inchangés ; les valeurs exactes actualisées figurent dans `results/validation_summary.json`.

L'audit gelé `verify.py --split validation` recompose indépendamment la formule de lecteur en FP64 sur les 32 premières observations, cinq lectures chacune : **160 pertes**, écart maximal **0,0001001766**, sous le seuil inchangé de 0,0002. Il recompose également les 2 048 contrastes et les cinq comptes d'exactitude. Les critères techniques de développement sont donc satisfaits avant l'ouverture du test. Aucun effet positif ne conditionne ce contrôle structurel.

## Test réservé : 512 séquences, sans adaptation neuronale aval

Le test est achevé le **25 octobre 2026 à 18:01:52 UTC**, en **63,49 s** selon le périmètre du runner. Les 512 séquences fournissent **2 048 corruptions** : deux têtes × deux donneurs par séquence. Les huit rejeux complets préspecifiés donnent un écart maximal de **0,0001831055 logit** ; le lecteur natif est exact, le résidu natif recomposé exact et le changement du MLP parallèle nul. L'écart maximal entre self-repair et effet du lecteur vaut **0,0001707396**, sous la limite 0,0002. Aucune cible n'a été remplacée après observation.

Les moyennes globales, toutes conditions conservées, sont :

- Dommage NLL `D` : **0,00239079 nat**, IC descriptif 95 % [-0,00160547 ; 0,00686917].
- Récupération NLL du lecteur `R` : **-0,000601093 nat**, IC [-0,00158942 ; 0,000140808].
- Interaction de lecteur `I` : **0,00000709485 nat**, IC [-0,000230012 ; 0,000147205].
- Spécificité de facteur `S` : **-0,000109216 nat**, IC [-0,000355497 ; 0,0000190420].
- Différence d'effet direct : **-0,0182510 logit centré**, IC [-0,0262613 ; -0,0106981].
- Self-repair brut : **0,000823937 logit centré**, IC [-0,00505181 ; 0,00670619].
- Effet du seul lecteur : **0,000823713 logit centré**, IC [-0,00505410 ; 0,00670547].
- Rapport moyen des facteurs natif/corrompu : **1,00013290** ; 1 035/2 048 rapports sont supérieurs à un.

Sur les 512 séquences natives, **151 tokens suivants sont exacts, soit 29,49 %**. Les comptes répliqués par tête et donneur sont Y00 **604/2 048**, Y10 **598/2 048**, Y11 **598/2 048**, Y01 **604/2 048**, Y1wrong **597/2 048**. La comparaison appariée confirme zéro gain et zéro perte de justesse entre lecteur natif figé et lecteur propre à la corruption. La justesse faible du modèle sur ces prochains tokens n'autorise aucun filtrage des seuls succès natifs.

H2 présente un dommage moyen de **0,00555188 nat**, une récupération de lecteur de **-0,000986300 nat** et un self-repair de **0,00345943 logit**. H10 présente respectivement **-0,000770298 nat**, **-0,000215886 nat** et **-0,00181155 logit**. Les IC de ces récupérations incluent zéro. Les deux têtes sont conservées ; la plus favorable n'est pas retenue comme résultat principal.

Les signes bruts sont variables : 1 053 valeurs de self-repair et 1 016 récupérations NLL de lecteur sont positives. Un export descriptif post-hoc compte 735 désaccords de signe entre ces deux métriques. Ce sont des estimands différents, avec des effets parfois proches de la précision numérique : ce désaccord ne mesure pas une erreur de la méthode publiée. L'origine structurelle du changement de lecture reste la même, sans contribution neuronale aval susceptible de s'adapter.

## Audit du test et statut de clôture

L'audit supplémentaire `audit_saved_readout_memmap.py` a été ajouté après les résultats. Il relit uniquement les poids utiles par blocs de 1 024 lignes, sans importer ou charger un modèle, et recalcule la formule directe FP64 du lecteur, les **mêmes 32 premières observations** et la **même tolérance de 0,0002** que l'audit gelé. Il passe : **160 pertes**, erreur maximale **0,0000922124** ; erreur maximale du contraste `R` **0,0001027708**. Il vérifie aussi **8 192 contrastes scalaires**, les cinq comptes d'exactitude, la finitude des états et l'intégrité du gel. Temps de cet audit : **1,52 s**. Son export est [test_memmap_verification.json](results/test_memmap_verification.json).

Le script gelé `verify.py --split test`, qui recharge le modèle complet pour le seul lecteur, **n'a pas été exécuté**. Il est remplacé pour cette livraison par cet audit supplémentaire, explicitement post-hoc ; aucun nouveau forward complet du réseau n'est exécuté. Une erreur initiale d'export du nouvel audit (`numpy.bool_` non sérialisable en JSON) est conservée avec sa version antérieure, puis corrigée par conversion explicite en booléen Python. Elle n'a modifié aucune observation scientifique.

La branche est clôturée sur les 640 séquences annoncées, avec ses limites et ses résultats faibles ou négatifs conservés. Le contrôle vérifie la séparation des estimands lecteur/continuation dans un cas structurel extérieur déjà étudié ; **il n'établit pas un nouveau mécanisme, une récupération générale ou la supériorité de notre méthode sur la mesure publiée**.

## Dépendances et coût d'intégration

Python 3.13, PyTorch 2.8.0, NumPy 1.26.4 et Transformers 4.57.1 ; CPU FP32, un thread, attention eager, aucun cache KV. Les dépendances Transformer proviennent du dossier isolé historique en lecture. La lecture Parquet utilise PyArrow 19.0.1 dans un nouveau dossier isolé : la tentative PyArrow 26 échouait parce qu'elle exige NumPy 2. L'environnement scientifique existant n'a pas été modifié. Le premier téléchargement des poids était tronqué ; sa reprise HTTP a abouti au fichier dont l'empreinte officielle est vérifiée.

Les estimations du directeur ne sont pas reprises comme durées mesurées. Le temps d'installation et le temps de téléchargement se distinguent du temps de calcul ; la pression mémoire et les autres tâches partagées empêchent d'interpréter ces durées comme une comparaison de méthodes.

## Portée et livrables

Ce contrôle établit l'exécution de mesures de self-repair d'effet direct au niveau du lecteur alors que les contributions attention/MLP pertinentes ne s'adaptent pas. Il ne démontre ni récupération moyenne de NLL sur une population, ni exactitude des classes générales du benchmark, ni valeur ajoutée face aux travaux antérieurs. Le résultat faible ou négatif sur WikiText-2 est conservé comme tel. La distinction entre lecteur et continuation exige que leur frontière soit déclarée ; elle n'est pas une classe d'architecture découverte automatiquement.

Les artefacts consultables sont [la feuille de route](FEUILLE_DE_ROUTE.md), [le protocole](protocol.json), [le gel](results/protocol_frozen.json), [les séquences et donneurs](data/sequences.json), [le runner](scripts/run.py), [l'audit gelé](scripts/verify.py), [l'audit supplémentaire de test](scripts/audit_saved_readout_memmap.py), [l'amendement de lecteur](AMENDEMENT_LOGICIEL.md), [l'amendement de mesure](AMENDEMENT_FP64.md), [le résumé du développement](results/validation_summary.json), [celui du test](results/test_summary.json), les fichiers `results/*_observations.json` et les résidus `results/*_states.npz`. [execution_status.json](results/execution_status.json) reflète l'état final, avec distinction entre audit gelé et audit supplémentaire. Le manifeste de livraison est distinct des trois gels historiques.

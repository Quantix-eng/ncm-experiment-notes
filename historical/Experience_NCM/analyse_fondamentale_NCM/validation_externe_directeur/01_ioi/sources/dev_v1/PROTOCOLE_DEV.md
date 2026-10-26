# Pilote IOI : dommage, récupération et contrôles de compensation

25 octobre 2026. Protocole fixé avant toute inférence de cette campagne. Gel local SHA-256, pas une prérégistration indépendante. Les données et le modèle sont extérieurs ; l'implémentation et l'analyse restent internes.

## Question et antécédents

Les contrôles supplémentaires distinguent-ils une récupération nécessitant une adaptation aval d'une tolérance ou d'un effet de lecture ? Le [self-repair de Rushing et Nanda](https://arxiv.org/html/2402.15390v2) est un antécédent direct, comprenant déjà contributions aval figées et normalisation finale figée. Leurs mesures et les nôtres sont comparées sans prétendre mesurer le même estimand. Le circuit [IOI de Wang et al.](https://arxiv.org/html/2211.00593v1) fournit des composants candidats indépendants, pas une vérité interne exhaustive. Leur compensation après suppression conjointe des trois Name Movers ne garantit pas notre effet après rééchantillonnage d'une seule tête.

## Données et partitions

MIB–IOI, révision épinglée dans `data/provenance.json`. 128 familles parmi `dev.json`, 512 parmi `test.json`, tirage sans remise fixé par la graine 2026102517. Trois variantes restent ensemble : native A, s2_io_flip B (réponse changée), s1_io_flip Aplus (réponse préservée). Toutes les métadonnées et tokenisations sont vérifiées sans inférence ; exclusion des réponses multi-token, labels inattendus, variantes de longueurs différentes, sources/variantes répétées et chevauchements exacts avec développement. Les exclusions sont exportées. Ce contrôle ne réserve pas des templates nouveaux : les templates peuvent être partagés entre partitions.

Les noms sont encodés avec un espace initial et doivent chacun former exactement un token. Un BOS GPT-2 50256 est ajouté explicitement ; ce choix est fixé pour cette extension, sans prétendre à une reproduction bit à bit du pipeline officiel MIB. Aucun exemple n'est sélectionné sur sa justesse native ou son dommage. Le test public est téléchargé pour le contrôle des entrées, mais aucune sortie test ne sera calculée avant la décision de développement.

## Modèle, frontière et interventions

GPT-2 small local, révision 607a30d783dfa663caf39e06633721c8d4cfcd7e, CPU FP32 un thread, eager attention, évaluation sans dropout ni cache KV. Intervention principale : sortie pré-projection de L9H9 à la dernière position, remplacée par les 64 composantes natives d'une autre source. La projection W_O officielle transforme ce remplacement en contribution résiduelle. Les autres têtes de la même couche restent natives. Frontière après toute l'attention de la couche 9, avant son MLP. Les lectures ne concernent que le prochain token, pas une révision de tokens déjà émis.

Deux donneurs différents : positions (i+37) mod n et (i+73) mod n dans la liste figée de la même partition. Aucun choix selon les noms, scores ou effets ; ils peuvent donc parfois transporter un nom partagé. Tous les contrôles mal appariés utilisent exactement le même site et une autre activation native. Les continuations recalculent le MLP de la couche d'intervention puis les couches restantes ; les positions antérieures sont natives et fixes à chaque couche. Le lecteur est la LN finale native recalculée, puis le lm_head.

Y00 = état natif + contributions aval natives ; Y10 = état perturbé + contributions natives ; Y11 = état perturbé + contributions adaptées ; Y01 = état natif + contributions adaptées ; Y1wrong = état perturbé + contributions adaptées à l'autre donneur. Les sommes suivent l'ordre FP32 du programme. Le sham doit être bit à bit natif, ainsi que les tokens antérieurs. Les résidus finaux et les suites sont conservés pour redécodage et audit.

Confirmation indépendante des contrastes : recalculer Y11 en figeant à leur valeur native les six Backup Name Movers strictement aval publiés : L10H1,H10,H6,H2 et L11H9,H2. Les backups L9H0/H7 ne sont pas aval de L9H9. Le témoin a la même taille et la même répartition par couche : L10H0,H3,H4,H5 et L11H0,H1. Le témoin n'est pas supposé inerte. Les têtes négatives L10H7/L11H10 ne sont pas choisies comme témoins neutres. L'effet différentiel ne couvre ni tous les composants ni tout self-repair.

## Observations et accès à l'entrée

O conserve les sorties natives des trois variantes. E complète les données C par le plan 2 entrées A/B × 2 états au point de coupure × 3 suites (native A figée, native B figée, recalculée), plus A/Aplus. Le « canal entrée » retenu contient les représentations natives de tous les tokens antérieurs ; ce n'est pas un changement total de l'entrée brute à mémoire égale. La suite figée du receveur peut véhiculer son information ; aucune classe d'architecture n'est déduite de ce contrôle. Aucune des six classes analytiques n'est imposée à GPT-2.

## Mesures, abstention et décision de passage au test

Primaire : NLL du nom correct sur le vocabulaire entier. D=L10−L00, R=L10−L11, I=L10+L01−L11−L00, S=L1wrong−L11. Conserver différence IO−S, exactitude à deux candidats et vocabulaire entier. Mesure publiée : SR=Δlogit_correct_centré−ΔDE_tête, où ΔDE projette la vraie différence donneur−receveur avec le facteur LN ORIGINAL figé, et non l'approximation −DE_natif. Il s'agit d'un self-repair sur le logit centré ; il ne mesure pas directement une récupération NLL.

Porte de développement principale, pratique et non confirmatoire : moyenne D ≥0,05 nat et au moins 25 % des familles avec D moyen >0,01 nat ; tous les contrôles logiciels doivent passer. Si échec, ne calculer aucune sortie réservée et ne changer ni tête principale, ni dose, ni donneurs pour obtenir un succès. Les 512 familles demeurent réservées à une campagne ultérieure amendée. Une absence de dommage ne réfute pas toute compensation de GPT-2.

Si passage : figer un protocole test avant ses sorties, conserver tous les cas, garder L9H9 primaire et L9H6/L10H0 secondaires sans remplacement opportuniste. Tester séparément les affirmations dommage utile, récupération, spécificité et blocage corroborant. Pour les quatre contrastes D,R,S,B, IC appariés bootstrap par famille à 98,75 % (Bonferroni quatre), 20 000 tirages ; marges respectives 0,05, 0,01, 0,01, 0,01 nat, fixées ici. B = perte sous blocage des backups − perte sous blocage témoin. Les marges sont des choix pratiques de cette campagne, pas universelles. I, mesures de logits et résultats par template sont descriptifs. Aucun rééchantillonnage par échange supposé indépendant. Les cas de désaccord entre métriques sont conservés sans désigner automatiquement une méthode antérieure erronée.

Une version sans contrôle figé ne peut établir la nécessité de l'adaptation ; une version sans S peut constater R mais doit s'abstenir sur la spécificité. Couverture, abstention et nombre d'affirmations corroborées seront séparés. Une moyenne positive ne certifie pas chaque famille. L'utilité scientifique supplémentaire n'est établie que si une conclusion que les contrôles existants ne permettaient pas déjà est corroborée ; sinon la livraison est une évaluation négative ou une réplication ciblée.

## Intégrité et arrêt

Figer protocole, scripts, entrées, sélection, tokenizer, poids et source GPT-2 importée avant pilote. Une assertion échouée arrête le runner ; conserver les versions et rédiger un amendement daté avant correction/reprise. Ne jamais présenter une correction logicielle comme prédiction nouvelle. Les audits sont internes. Les temps mesurés comprennent chargement/calcul explicitement indiqués ; les durées du directeur restent des estimations. Aucune nouvelle campagne NCM ou comparaison causale générale AR/parallèle n'est prévue.

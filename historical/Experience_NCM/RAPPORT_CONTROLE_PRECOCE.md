# Fiabilité précoce du NCM — faisabilité d'une politique d'abandon

**24 octobre 2026 — étude exploratoire locale, sur développement uniquement.**

## Question et portée

Peut-on économiser du calcul en abandonnant au niveau 9 les trajectoires les plus susceptibles d'échouer, sans perdre plus d'un point de taux de résolution ? Cette question porte sur l'utilité d'un signal avant la réponse finale. Elle est distincte du transfert de calibration, qui n'a pas été testé ici.

Un estimateur calibré avait déjà été construit dans H1. Le nouveau travail n'invente donc pas le principe de sonde de confiance. Il confronte ce signal au coût réel de sa collecte et à son utilisation dans le NCM. Après la sortie finale d'un Sudoku unique, le vérificateur exact reste une référence plus forte qu'un score probabiliste pour accepter une grille.

**Modèle unique :** NCM Sudoku hard, 25 457 600 paramètres générateur + 14 896 512 VQ-VAE, soit 40 354 112 paramètres. CPU FP32, batch 8, quatre threads, argmax déterministe. Aucun poids ni chemin public modifié.

## Protocole de faisabilité

Les 1 000 groupes d'apprentissage officiels, dont 137 échecs, servent au développement. Les anciens tests consultés ne sont pas présentés comme intacts ; aucun nouveau corpus n'a été généré. Les anciens scores OOF ont été inspectés avant la rédaction du [design local](experiments/metacognition/early_control/design-v1.json). Les nouvelles sondes, marges et règles conditionnelles ont été fixées avant leur profilage final ; il ne s'agit pas d'un pré-enregistrement public.

Trois régressions logistiques à pénalité fixe λ=1 sont ajustées en cinq plis groupés. Normalisation et PCA sont apprises uniquement hors du pli prédit :

- **Codes :** puzzle et résumés des probabilités des codes jusqu'au niveau 9, 1 026 variables avant suppression des constantes.
- **Interne seul :** 16 composantes PCA des moyennes/écarts-types des 512 canaux du dernier bloc du générateur, au niveau 9, avant normalisation finale.
- **Codes + interne :** combinaison des deux.

La référence **observable archivée** utilise les 1 341 résumés H1 et ses modèles OOF figés. Ses neuf décodages intermédiaires sont réellement calculés et chronométrés. Toutes les variables sont bornées au niveau courant ; labels et solutions sont exclus des entrées.

Chaque politique abandonne si le logit d'échec dépasse strictement un seuil. Le seuil de fonctionnement maximise les abandons sous une limite de **3 résolutions perdues sur 1 000** sur OOF. Une frontière optimiste autorise **10 pertes sur 1 000**, pour évaluer si la marge finale d'un point pourrait être plausible. Ces seuils sont sélectionnés sur les mêmes scores de développement que leurs résultats : ils sont optimistes, sans garantie indépendante.

Les critères candidats d'une future confirmation seraient une borne inférieure simultanée d'économie supérieure à **5 %**, contre poursuite fixe et référence observable compétitive, et une borne inférieure de différence de résolution supérieure à **−1 point**. Les marges sont pratiques. Une confirmation demanderait un dimensionnement sous des effets strictement favorables à ces marges, puis un nouveau corpus gelé avant mesure. Le pilote ne remplace pas ce dimensionnement.

## Moteur et contrôles

Le moteur isolé conserve au niveau 9 les codes, le latent cumulatif et les clés/valeurs du préfixe. Les trajectoires retenues sont regroupées en lots de huit ; seul le dernier lot est complété. Il s'agit d'un arrêt suivi d'une reprise réelle, pas d'une économie simulée après calcul de toutes les réponses.

Sur 32 puzzles d'apprentissage, codes, reconstructions, logits et réponses concordent bit à bit avec la référence. Les regroupements, permutations de lots et compléments du dernier lot conservent cette parité. Les caractéristiques concordent avec l'observateur passif antérieur. Le cas « aucune trajectoire retenue » est également exécuté : aucun appel aux dernières étapes. Les poids restent identiques et les caches sont nettoyés.

Le profilage porte sur 128 puzzles, cinq répétitions avec ordre des conditions mélangé ; les politiques réelles sont exécutées sur les 1 000 puzzles. La règle fixée prévoit trois passages si la porte optimiste est franchie, sinon un passage descriptif. Les coûts comptent extraction, score, préparation des lots, reprise, décodage final, vérification et remplissage. Le modèle, les coefficients et les prompts sont préchargés. Chargement, apprentissage et audit ultérieur sont exclus. Il s'agit d'un traitement de corpus sur un ordinateur partagé, pas d'une garantie de latence individuelle.

Les comparateurs comprennent le chemin public inchangé, le moteur avec poursuite systématique, les quatre sondes et une décision aléatoire indépendante des labels, appariée au nombre d'abandons de « codes + interne ».

## Résultats de développement

À la limite de trois résolutions perdues, codes abandonne **8** puzzles, codes + interne **9**, observable archivé **12**, interne seul **36**. Ce dernier écarte **33 échecs et 3 succès** : les 860 résolutions restantes représentent 86,0 % des 1 000 entrées, contre 86,3 % initialement. Les 36 abandons sont répartis sur les cinq plis, pas concentrés dans un seul modèle.

À la limite optimiste de dix pertes, les nombres d'abandons sont respectivement **32, 40, 26 et 56**. Même avant de compter la sonde, l'interne seul permet donc d'éviter la fin du calcul sur seulement 5,6 % du corpus, pas sur l'ensemble des 13,7 % d'échecs.

Les AUROC OOF sont 0,88489 pour codes, 0,89132 pour codes + interne, **0,90446 pour interne seul**, et 0,86672 pour l'observable archivé. Elles décrivent ce développement ; elles ne remplacent pas les AUROC des anciens tests et ne démontrent pas un avantage confirmé de la nouvelle sonde. L'AUROC ne garantit pas une utilité dans la petite zone de rejet autorisée, ni une probabilité bien calibrée. Les nouveaux logits ne sont pas présentés comme des probabilités calibrées.

## Coût et porte de faisabilité

Sans extraction ni score, la fin du calcul après le niveau 9 représente **40,96 %** du temps en médiane, avec un maximum observé de **46,35 %** sur les 80 lots chronométrés de cette condition. La numérotation 9/12 n'indique donc pas la fraction de calcul consommée.

En combinant ces fractions avec la meilleure frontière interne sous dix pertes, l'économie optimiste est **5,6 % × 40,96 % = 2,29 %**, ou **2,60 %** en utilisant la plus grande fraction restante observée. Avec le seuil effectivement exécuté, limité à trois pertes, elle est d'environ **1,47 %** avant tout surcoût. Le remplissage des lots réduit encore les possibilités : 36 abandons évitent quatre lots finaux sur 125, pas 36 inférences complètes.

**La porte de faisabilité de 5 % n'est pas franchie**, même avant les coûts de sonde, de séparation et de regroupement. C'est un filtre pratique sous hypothèse de coût comparable par exemple dans des lots de forme fixe, pas une borne mathématique universelle sur les temps muraux ni sur tous les observateurs possibles. Selon la règle prévue : un seul passage descriptif des politiques, aucun nouveau corpus, aucun dimensionnement confirmatoire lancé.

Sur les 1 000 puzzles, temps complets observés, dans l'ordre effectif :

- Codes + interne : **101,05 s**, 9 abandons, 860 résolutions vérifiées.
- Observable archivé : **77,15 s**, 12 abandons, 860 résolutions.
- Codes : **68,06 s**, 8 abandons, 860 résolutions.
- Référence publique fixe : **78,83 s** ; 863 résolutions de référence.
- Moteur avec poursuite systématique : **125,73 s**, 863 résolutions, toutes les sorties conformes aux archives.
- Aléatoire : **137,48 s**, 9 abandons, 856 résolutions ; 7 succès perdus, contre 3 pour la combinaison au même nombre d'abandons.
- Interne seul : **133,94 s**, 36 abandons, 860 résolutions.

**Ces différences de temps ne sont pas des gains ou ralentissements attribuables aux seules politiques.** La dérive entre passages est trop forte : les lots de poursuite sans sonde vont de 0,535 à 1,156 s dans le profilage par étapes. « Codes » semble économiser 13,66 % contre le fixe tout en n'évitant qu'un lot final sur 125 ; cela ne peut pas être expliqué par les seuls abandons. Le témoin de poursuite systématique est lui-même 59,5 % plus lent que le passage public fixe. Effets d'implémentation et variation de la machine peuvent contribuer ; leurs parts ne sont pas identifiées. Aucune accélération stable, aucun intervalle de performance, aucune non-infériorité confirmatoire ne sont revendiqués.

La référence observable riche n'est pas une implémentation minimale : elle construit aussi des éléments intermédiaires réutilisés par l'extracteur antérieur. Son coût mesuré ne prouve pas que toute politique observable serait intrinsèquement coûteuse. Le refus de poursuivre repose déjà sur la comparaison optimiste avec le calcul fixe, indépendamment de cette référence.

## Vérifications, anomalies et limites

L'[audit arithmétique](audit/metacognition/early-control-v1/verification.json) recalcule indépendamment les AUROC par comparaison directe des paires, toutes les frontières de seuils, les masques de poursuite, les résolutions perdues, les lots et leur remplissage, la décision de faisabilité et les empreintes. Les sorties de chaque trajectoire conservée concordent avec les archives propres. Les prétraitements et les cinq affectations de plis sont vérifiés. **31 tests logiciels passent**, dont deux nouveaux tests de frontière temporelle et d'absence réelle de poursuite. Aucun optimiseur n'atteint la limite de 500 itérations ; le plus grand gradient final absolu vaut environ 7,93e-6.

L'anomalie principale de cette campagne est **l'instabilité des temps muraux**, conservée et explicitée ci-dessus. Aucun écart de codes ou de réponse, modification de poids, réparation des grilles ou fuite de solution n'a été détecté par les contrôles effectués. La version du conducteur du pilote technique précède l'ajout du témoin aléatoire et de la règle conditionnelle de répétition ; son moteur numérique est identique à celui du profilage final, empreintes vérifiées. Le cas « tout abandonner » est effectivement exécuté dans le profilage et testé séparément.

Les modèles sont entraînés sur environ 800 puzzles par pli, pas sur un corpus neuf de déploiement. Un unique modèle réentraîné sur 1 000 pourrait avoir un autre comportement. Les seuils de développement et la sélection de la meilleure sonde sont optimistes. La sonde interne seule paraît intéressante dans cette zone de rejet, mais sa robustesse externe n'est pas établie. L'aléatoire est apparié à la combinaison codes + interne, pas aux 36 abandons de la sonde interne seule. Aucun contraste inférentiel entre ces politiques n'est annoncé.

La machine est un **Apple M1, 8 Go de RAM**, PyTorch 2.8.0 ; quatre threads PyTorch, pas une limitation de tous les processus système. Aucun pic mémoire n'est revendiqué. La marge disque observée était d'environ 8 Gio au départ ; l'arrêt n'est pas imposé par un manque d'espace. Les nouveaux artefacts sont de quelques mégaoctets, sans duplication des gros tenseurs historiques.

## Conclusion et suite recommandée

**Un signal interne exploitable pour classer certains échecs existe sur le développement ; cette politique d'abandon au niveau 9 ne justifie pas une confirmation visant 5 % d'économie à moins d'un point de perte de résolution.** Les échecs suffisamment séparables sont trop peu nombreux, et trop peu de calcul reste évitable. Il s'agit d'un résultat de faisabilité sur les représentations et la politique définies, pas d'une impossibilité générale de construire une confiance utile dans un NCM.

La confiance calibrée externe reste une question distincte, déjà abordée par H1. Ce pilote ne valide ni son transfert ni une conscience native de l'erreur. Déplacer le seuil, le niveau ou la marge après ces résultats ne transformerait pas ce lot en confirmation indépendante.

**Recommandation : clore cette variante économique et conserver le moteur comme outil validé.** Pour poursuivre l'idée de confiance, privilégier ensuite un usage où la fiabilité change une décision réelle : allocation de plusieurs candidats à budget égal, ou tâche dont la qualité finale n'est pas immédiatement vérifiable. Choisir une seule de ces questions, justifier architecturalement les variables, puis réserver et dimensionner une nouvelle évaluation avant toute sélection sur ses résultats. Ne pas lancer un grand corpus uniquement pour confirmer un objectif déjà peu plausible dans ce pilote.

Preuves et reproduction : [design](experiments/metacognition/early_control/design-v1.json), [développement](audit/metacognition/early-control-v1/development.json), [profilage](audit/metacognition/early-control-v1/profiling/report.json), [audit](audit/metacognition/early-control-v1/verification.json), [commandes](experiments/metacognition/early_control/README.md), [manifestes de clôture](audit/metacognition/early-control-v1/campaign-summary.json).

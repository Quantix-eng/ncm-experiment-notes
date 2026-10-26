# Extension du benchmark à un Transformer autorégressif réel

25 octobre 2026. **Une seconde architecture apprise a été exécutée et auditée : GPT-2 small sur une tâche textuelle contrôlée.** Le protocole O/C/E fonctionne dans le régime d'accès défini, sans étiquette de mécanisme imposée. Huit familles sur seize ont une préférence de réponse dominée par le contexte d'entrée ; les huit autres restent non identifiables en raison d'échecs natifs. Les rapports sont exclusivement Markdown.

Lire le [manuscrit consolidé, version 2](PAPER_MULTIMODEL.md), le [protocole gelé](PROTOCOLE.md), le [complément AR autonome en anglais](PAPER_ADDENDUM.md), les [résultats complets](results/summary.json) et l'[audit](results/verification.json). Le [papier précédent](../article_benchmark_coordination/PAPER.md) est conservé inchangé ; la nouvelle version ajoute l'expérience AR sans réécrire son historique.

## 1. Ce qui a été testé

GPT-2 small, 124 439 808 paramètres, poids préentraînés de `openai-community/gpt2`, révision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Aucun apprentissage. CPU FP32, un thread, mode évaluation, attention eager ; pertes calculées en FP64. Les versions effectives sont PyTorch 2.8.0, NumPy 1.26.4, Transformers 4.57.1. Les outils HF sont installés dans un répertoire isolé, sans remplacer les dépendances des campagnes antérieures. [Provenance officielle du modèle](https://huggingface.co/openai-community/gpt2).

Le [corpus](data/families.json) comprend 16 paires de mots, fixées avant toute inférence. Exemple de construction : `The code is cat. The code is` et sa variante avec `dog`. La tâche évaluée consiste à restituer le mot assigné. Chaque cible est un seul token avec espace initial ; la justesse utilise l'argmax sur tout le vocabulaire. Aplus remplace le premier `The` par `My`, avec même longueur et même référence. Les 48 prompts ont huit tokens. Aucun échec n'a été remplacé.

Le point de coupure est fixé après le bloc 6 sur 12. L'état échangé est le vecteur du **dernier token**, de dimension 768. Les représentations des positions antérieures restent celles du receveur à chaque couche. Les six blocs restants constituent la continuation, conservée comme douze contributions séparées d'attention et de MLP, ajoutées dans leur ordre natif. Les suites figées ne sont pas des tokens de réponse imposés.

Il s'agit d'une intervention à l'intérieur du calcul d'un prochain token, pas d'une réécriture du texte déjà émis. Le contexte natif conservé est un canal vers le prompt, analogue fonctionnellement à un accès à sa mémoire ; le vecteur échangé ne contient pas toute la mémoire du modèle. Les positions antérieures sont obtenues par une passe native préalable ; aucune économie de calcul n'est démontrée. Les couches AR et les niveaux NCM ne sont pas assimilés à un même axe temporel.

## 2. Gel et contrôles techniques

Le gel local du **2026-10-25T15:09:14Z** inclut 1 291 fichiers : poids, tokenizer, données, protocole, paramètres, runner, analyse, audit et principales sources importées. C'est un gel avant sorties, sans registre indépendant. Les critères n'ont pas changé après inférence et aucun amendement logiciel n'a été nécessaire.

La collecte complète a exécuté 48 passes natives avec parité exacte et shams, 128 essais C et 384 conditions E. L'adaptateur conserve le contexte antérieur natif par causalité ; il n'utilise aucun cache KV mutable. Les diagonales retrouvent le natif ; avec état et suite figés, changer le contexte receveur ne change pas la lecture finale. L'audit vérifie ces invariances, au lieu d'attribuer tout changement à l'entrée par défaut.

Temps du runner : 36,93 secondes, comprenant collecte et sauvegarde. Ce temps n'est pas une comparaison de vitesse avec NCM : tâches, nombres de positions, arithmétique et volumes d'artefacts diffèrent.

## 3. O : performances natives et lectures intermédiaires

- A : 12/16 réponses exactes.
- B : 8/16.
- Aplus : 13/16.
- Ensemble : **33/48**, dont 20/32 pour A/B.

Les deux ancres A et B sont simultanément exactes dans huit familles. Toutes les familles restent dans les statistiques d'effets ; les conditions de classification ne permettent pas de forcer un diagnostic sur les huit familles restantes.

La lecture finale appliquée aux états intermédiaires montre des corrections et des dégradations. Aucun mot cible n'est l'argmax à la coupure après six blocs. Ce constat ne mesure pas une absence d'information dans ces états. Pour B, 12/16 lectures sont exactes après le bloc 10, contre 8/16 à la fin. Cette variation concerne des lectures internes, pas un texte déjà généré que GPT-2 aurait révisé. Les treize lectures et toutes les transitions sont conservées, sans sélection d'un sommet favorable.

## 4. C : une petite amélioration moyenne de NLL ne suffit pas

Deux directions gaussiennes de supports disjoints, chacune sur une moitié du vecteur, sont testées avec deux signes sur les 32 ancrages A/B. Leur norme est fixée à 10 % de celle de l'état, sans calibration sur les réponses ni escalade après résultats. Les perturbations peuvent être hors distribution.

Les contrastes de NLL en nat/token et leurs IC ajustés à 98,75 %, obtenus par 20 000 rééchantillonnages des 16 familles entières, sont :

- D, dommage sous suite figée : **0,00181115**, IC [0,00141522 ; 0,00213673].
- R, récupération relative : **0,00781555**, IC [0,00088758 ; 0,01519674].
- I, interaction croisée : **0,00181959**, IC [0,00113743 ; 0,00242062].
- S, avantage de suite adaptée sur suite mal appariée : **0,00168670**, IC [0,00070116 ; 0,00268459].

Le dommage moyen est contenu dans la marge préfixée ±0,01 nat/token. Le critère exigeant les quatre bornes inférieures au-dessus de 0,01 n'est pas satisfait. Décision conservée : `mean_frozen_damage_equivalent_within_prespecified_margin`. Un effet peut être positif et inférieur à la marge pratique ; le signe seul ne remplace pas le critère.

Les exactitudes sont **80/128 pour Y00 et Y10**, **78/128 pour Y11, Y01 et Y1wrong**. Les 128 requêtes ne sont pas des unités indépendantes. Un export descriptif apparié ajouté après les résultats confirme que Y10 ne change aucun argmax, tandis que chacune des trois autres conditions perd deux réussites natives sans gain. La NLL moyenne baisse sous régénération, alors que deux réponses correctes sont perdues. Ni la modification de la continuation, ni le signe moyen de R ne prouvent donc une réparation utile selon le critère fixé.

Cet [export post-hoc](results/paired_descriptive.json) par [postprocess.py](postprocess.py) n'était pas dans le gel initial. Il repose exclusivement sur les argmax déjà conservés, n'effectue aucune inférence et ne modifie ni seuil, ni contraste, ni diagnostic. Son statut est consigné dans le journal.

## 5. E : le contexte peut dominer dans un modèle AR

Sur toutes les familles, les effets moyens descriptifs de préférence B contre A sont :

- Contexte receveur u : **3,76857 nat**, IC 95 % [1,23271 ; 6,39620].
- État échangé v : **0,30567 nat**, IC [0,19423 ; 0,42402].
- Interaction w : **0,01608 nat**, IC [-0,21121 ; 0,22946].

L'étendue native moyenne g vaut 4,07424 nat et u+v=g par identité algébrique. Ces rapports ne sont pas des proportions de calcul, des ressources informationnelles ou des coefficients causaux de mélange.

Avec les critères historiques inchangés — deux ancres exactes, g>0,1, interaction relative au plus 0,2, ratio dans [0,1], seuil d'entrée 0,8 — les diagnostics sont :

- **8/16 familles : `input_dominant_response_preference_in_this_test`.**
- **8/16 : `non_identifiable_native_anchor_failure`.**

Les huit familles diagnostiquées sont précisément celles dont les deux ancres sont exactes. Ce n'est pas une sélection rétrospective : cette condition était gelée et l'effectif total reste seize. Les effets des seize familles, y compris les négatifs, sont publiés. Aucune vérité de mécanisme interne n'est attribuée au modèle.

Pour les 32 échanges réciproques hors diagonale A/B :

- Suite régénérée : **19/32** références du receveur, 10/32 du donneur, 3 autres.
- Suite native du receveur figée : **18/32** références du receveur, 11/32 du donneur, 3 autres.
- Suite native du donneur figée : **9/32** références du receveur, 20/32 du donneur, 3 autres.

Le comptage apparié post-hoc donne un gain et aucune perte de justesse pour la régénération contre la suite figée du receveur : `copy-013`, receveur B. Cela décrit un cas de suffisance différente entre suites, sans établir une loi générale de réparation ou de spécificité. Le receveur n'est pas suivi parfaitement avec une suite figée ; la régénération ne le garantit pas non plus. Les deux témoins gauche/droite figurent séparément dans les résultats complets, sans les confondre avec l'alignement receveur/donneur.

Pour A/Aplus, les deux références sont identiques. Les suites régénérées, natives du receveur et natives du donneur donnent respectivement 25/32, 26/32 et 25/32 réponses exactes. Les différences de NLL, de probabilités et d'argmax sont conservées ; aucun ratio entre deux solutions identiques n'est calculé. Le changement de formulation peut changer la performance sans changer la référence attendue.

## 6. Conséquence pour la comparaison multi-modèles

La portée s'étend désormais de témoins construits et d'un NCM Sudoku à **deux architectures apprises et deux tâches**. L'adaptation à GPT-2 apporte des contrôles effectifs sur les tenseurs FP32 et conserve les abstentions dues aux performances natives. Elle montre aussi qu'une préférence dominée par le contexte est possible dans un Transformer AR lorsque ce canal reste disponible après échange d'un état partiel.

La proposition « AR = frozen_recipient_native, NCM = input_dominant, diffusion = mélange » ne peut donc pas servir de vérité de calibration. Ses termes ne sont pas du même type : une politique de suite, un diagnostic de préférence et une hypothèse sur un mécanisme. Les politiques figées sont exécutables sur NCM comme sur GPT-2. Les diagnostics décrivent le **modèle sous une intervention et un accès donnés**, pas l'architecture indépendamment de ces choix.

Les 8/16 GPT-2 et 15/16 NCM ne constituent ni un classement de modèles ni un test causal de l'architecture : tâches, performances natives, états, profondeurs et canaux d'entrée diffèrent. L'irréversibilité des tokens déjà émis en décodage AR append-only reste vraie, mais ne prédit pas la dépendance du calcul interne du prochain token. Aucune diffusion n'a été exécutée ; aucun diagnostic ne lui est attribué.

## 7. Audit, limites et suite justifiée

L'audit gelé retrouve exactement **1 024 recompositions** (640 conditions C et 384 E), **16 régénérations C et 16 E** par appels explicites aux sous-couches sur les deux premières familles, **six passes natives officielles**, et **640 pertes** recalculées par log-sum-exp scalaire. Les comptes de justesse, effets, seuils, normes, diagonales et invariances sont contrôlés. Les fichiers des manifestes NCM et de l'article précédent, ainsi que les 1 291 fichiers gelés de cette campagne, sont inchangés. C'est un audit logiciel par le même agent, pas une réplication externe.

Les seize paires ont été choisies manuellement et l'étude ne possède pas de test externe secret. Les IC bootstrap sont descriptifs sur cette collection restreinte ; aucune généralisation à tout le langage n'est établie. La tâche a une référence de copie définie, mais GPT-2 n'a pas été entraîné ici à suivre cette consigne ; les échecs observés restent des limites du dispositif. Une seule profondeur et une seule dose ont été testées. Les états croisés peuvent être hors distribution conditionnellement au contexte. Les seuils hérités ne sont pas des critères validés d'utilité sur texte.

La prochaine étape la plus informative serait un **contraste à tâche commune**, avec calibration sur développement indépendant, puis évaluation gelée : par exemple deux petits modèles appris AR et parallèles sur une tâche de contraintes commune. Il faudrait aussi comparer explicitement l'accès au contexte retenu et l'échange de la mémoire complète ; leur distinction est logique avant d'être architecturale. Les méthodes d'attribution indépendantes et l'évaluation extérieure restent absentes. Aucune de ces campagnes supplémentaires n'a été exécutée ici.

La diffusion n'est pas priorisée sur une étiquette supposée « mélange ». Son futur adaptateur devra définir état, conditionnement et bruit restant, puis traiter l'aléa partagé et le gel des suites sans forcer une classification. Aucun résultat de ce type n'est revendiqué dans cette livraison.

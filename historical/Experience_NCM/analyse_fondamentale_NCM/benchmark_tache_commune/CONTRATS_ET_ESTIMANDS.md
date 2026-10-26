# Ce que signifie bloquer l'accès au receveur

Note du 25 octobre 2026, rédigée après résultats pour expliciter les objets déjà fixés au [protocole](PROTOCOLE.md). Aucun nouveau seuil ni calcul de réseau.

## Le facteur causal dépend de la frontière choisie

Au point de coupure, écrivons l'état H=(P,Q), avec P les sept positions de contexte et Q les quatre requêtes. Pour les variantes i,j et la continuation neuronale F, les deux contrats régénérés sont :

- R : Y_ij=D(F(P_i,Q_j)).
- M : Y_ij=D(F(P_j,Q_j)).

Le contrat R mesure l'effet d'un préfixe représenté dans P à Q fixé. Il ne mesure pas l'effet total de remplacer l'entrée brute, qui aurait aussi changé Q. Le contrat M remplace les deux parties. Aucun tenseur calculé depuis i n'y entre sous régénération.

Le mot « entrée » dans le diagnostic désigne donc un **canal de conditionnement retenu**, pas une cause abstraite identique dans toute architecture. Le mot « état » désigne une partie de la mémoire dans R et toute la mémoire dans M. Un changement de diagnostic entre ces contrats ne démontre pas un changement des poids ou du mécanisme appris ; l'estimand expérimental a changé.

## Invariance attendue dans M

Sous M, Y_ij=Y_j. Si h_ij est une préférence B/A, h_AA=h_BA et h_AB=h_BB. Alors, pour les contrastes du protocole :

u=0 ; w=0 ; v=g=h_BB−h_AA.

L'étiquette état-dominant découle de cette identité lorsque les autres critères sont satisfaits. L'expérience vérifie que le logiciel réalise bien ce contrat sur des poids appris et des réponses correctes ; elle ne découvre pas l'unicité d'un algorithme cognitif parmi tous les programmes possibles.

## Une suite figée peut réintroduire l'information du receveur

Une continuation native T_i a été calculée lors d'une passe complète du receveur. Tester D(H_j+T_i), avec les contributions ajoutées dans l'ordre, transmet donc de l'information provenant de i par T_i. Ce contrôle n'est pas la même intervention que régénérer F depuis H_j.

Dans M, le choix `recipient_native` change la suite lorsque le receveur change. Le contrôle « pas d'effet du receveur à état et suite fixés » concerne plutôt `fixed_left` ou `fixed_right`, où la même suite est conservée. Confondre ces alignements donnerait une fausse violation du chemin bloqué.

Cette distinction explique pourquoi des suites figées peuvent produire la réponse du receveur alors que les suites régénérées produisent celle du donneur. Elle vaut pour les modèles synthétiques, GPT-2 et NCM, sans justifier une classe d'architecture « suit sa suite ».

## Observation commune et interventions différentes

Les trajectoires natives sont les diagonales des deux contrats et y sont identiques. Les contrats n'ajoutent pas deux nouveaux mécanismes à classer à partir de ces observations : ils définissent deux interventions futures. Leur effet différent illustre pourquoi une spécification du régime d'accès est nécessaire avant toute comparaison.

Des travaux sur les [abstractions causales](https://papers.nips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html) formulent précisément les alignements et interventions étudiés. Notre contrôle est plus restreint : il établit un chemin absent annoncé et des dépendances fonctionnelles de sorties. Aucune nouveauté de cette identité causale n'est revendiquée.

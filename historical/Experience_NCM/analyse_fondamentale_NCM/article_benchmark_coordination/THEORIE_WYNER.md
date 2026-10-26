# Coordination conditionnelle : ce que Wyner permet d'affirmer sur NCM

Note théorique du 25 octobre 2026, accompagnant [le papier](PAPER.md). Définitions établies, dérivations élémentaires et analyses auxiliaires explicitement séparées des observations historiques. Aucune nouveauté de théorème n'est revendiquée. Les références primaires sont données au fil du texte ; un second agent a vérifié les hypothèses et les preuves, sans constituer une revue scientifique externe.

## 1. Conclusion immédiate

Le résultat « 15/16 familles dominées par l'entrée » **ne prouve pas** que les indices réalisent l'information commune minimale de Wyner ni qu'ils compensent une perte de coordination causée par le pooling. Il décrit une préférence de sortie sous un ensemble d'interventions et un seuil fixés. Il n'est pas un estimateur de bits.

Le lien défendable est le suivant : l'entrée est une information latérale partagée accessible aux prédicteurs parallèles, ce qui peut contribuer à coordonner leurs distributions. Pour obtenir une borne conditionnelle non triviale sur une ressource supplémentaire au-delà de l'entrée complète, il faut spécifier une distribution non dégénérée de sorties à entrée fixée et vérifier les restrictions d'architecture. Ces conditions ne sont pas réunies dans la génération Sudoku argmax actuelle. Une information commune non conditionnelle, sur une population de puzzles, peut néanmoins être non triviale pour un modèle déterministe.

Le contrôle de suite figée reste déterminant : la suite native du receveur donne déjà 32/32 solutions du receveur après échange d'état, comme la suite recalculée. Les observations ne démontrent donc pas une adaptation nécessaire au nouvel état. Elles ne manipulent pas davantage le pooling.

## 2. Définition et ressource bornée

Pour une loi discrète de Y=(Y_1,...,Y_m), une information latérale S disponible à tous les producteurs et des logarithmes de base 2, nous utilisons l'extension conditionnelle multivariée :

\[
C_W(Y_1;\ldots;Y_m\mid S)
=\inf_{U:\;p(y\mid u,s)=\prod_i p(y_i\mid u,s)} I(Y;U\mid S).
\]

L'indépendance requise est **mutuelle**, pas seulement deux à deux. La version multivariée est étudiée par [Liu, Xu et Chen](https://arxiv.org/abs/1010.3613). La version conditionnelle à deux variables et sa décomposition selon l'information latérale sont définies par [Graczyk, Lapidoth et Wigger](https://www.isiweb.ee.ethz.ch/papers/docu/rgra-alap-mwig-2022-2.pdf), définition 2.1 et proposition 2.2. La formule ci-dessus combine explicitement ces formulations ; elle ne doit pas être présentée comme une invention.

Supposons une architecture à producteurs locaux `Y_i=f_i(U,S,E_i)` où les hasards privés E_i sont mutuellement indépendants conditionnellement à (U,S), et où aucun calcul global ultérieur ne recouple les sorties. U est alors une variable admissible dans l'infimum :

\[
C_W(Y_1;\ldots;Y_m\mid S)\le I(Y;U\mid S)\le H(U\mid S).
\]

Si U est un message discret de longueur fixe B bits, B doit donc être au moins C_W. Il faut compter **toutes** les ressources partagées dans U ou S. La dimension d'un tenseur, son nombre de valeurs FP32 ou la taille syntaxique d'un codebook ne mesurent pas à eux seuls son entropie utile. Pour un préfixe de codes de longueur L sur 16 symboles, 4L bits est un plafond syntaxique, pas une quantité mesurée ni un coût minimal atteint.

L'interprétation opérationnelle de [Wyner (1975)](https://doi.org/10.1109/TIT.1975.1055346) porte notamment sur un taux asymptotique de synthèse approchée. Il faut la distinguer de la synthèse exacte à taille finie étudiée par [Kumar, Li et El Gamal](https://arxiv.org/abs/1402.0062). Les calculs exacts élémentaires ci-dessous ne prouvent pas qu'un réseau atteint un taux opérationnel optimal.

L'entrée structurée S n'est pas automatiquement le hasard commun **indépendant de l'entrée** d'un problème de synthèse de canal. Cette distinction de budgets est essentielle dans [Cuff (2013)](https://arxiv.org/abs/1208.4415). Ici, appeler le préfixe « information latérale partagée » évite de confondre ces ressources.

## 3. Exemple exact : la parité uniforme

Soit n≥2, et Y uniforme sur les 2^(n−1) chaînes binaires de parité paire. Alors H(Y)=n−1 bits et chaque marginale est uniforme, donc :

\[
TC(Y)=\sum_i H(Y_i)-H(Y)=n-(n-1)=1\text{ bit}.
\]

Pour n≥3, toute paire de bits est indépendante : ses quatre valeurs ont la même fréquence. L'absence de dépendance deux à deux ne suffit donc pas à une production mutuellement indépendante.

Si la loi conditionnelle sachant U=u est un produit, posons p_i(u)=Pr(Y_i=1|u). Le support global ne contient que des parités paires ; chaque composante de mélange de probabilité positive doit aussi y rester. Ainsi :

\[
1=E[(-1)^{\sum_iY_i}\mid U=u]=\prod_i(1-2p_i(u)).
\]

Chaque facteur est dans [-1,1]. Un produit égal à 1 exige que chacun ait une valeur absolue de 1, donc chaque p_i(u) vaut 0 ou 1. Tous les bits sont déterministes conditionnellement à U. Par conséquent H(Y|U)=0 et I(Y;U)=H(Y)=n−1. Le choix U=Y atteint cette valeur :

\[
C_W(Y_1;\ldots;Y_n)=n-1\text{ bits}.
\]

Ce calcul est une application élémentaire de la définition et non un nouveau théorème. Le script auxiliaire énumère n=2..10, contrôle les entropies, la corrélation totale et les informations de paires, et vérifie qu'un support cartésien contenu dans la parité ne peut varier sur aucune coordonnée. L'énumération illustre la preuve générale ; ce n'est pas une optimisation numérique de C_W.

### Contre-exemple au transfert vers un décodeur global

Tirer n−1 bits R_i indépendants, sans message commun aléatoire, puis appliquer un décodeur global :

\[
D(R_1,\ldots,R_{n-1})=(R_1,\ldots,R_{n-1},\bigoplus_{i=1}^{n-1}R_i).
\]

La sortie est exactement uniforme sur la parité paire. Les producteurs de graines n'utilisent aucun hasard commun, mais le décodeur réalise l'interaction nécessaire. Cela ne contredit pas C_W=n−1 : l'architecture ne satisfait pas la restriction de producteurs finaux locaux indépendants. Une borne de coordination des sorties ne peut donc pas être appliquée sans contrôle à une architecture dont le lecteur est global.

## 4. Exemple exact : deux solutions du même puzzle

Fixons une base ambiguë B=b ayant exactement deux solutions s_A et s_B. Soit V une branche de probabilité p, et Y=s_V. Le choix U=V donne C_W(Y|B=b)≤h_2(p).

Si au moins deux cases i et j diffèrent entre les solutions, chacune détermine V. Donc I(Y_i;Y_j|B=b)=h_2(p). Pour tout U admissible, l'indépendance conditionnelle donne la chaîne Y_i–U–Y_j et :

\[
I(Y_i;Y_j\mid B=b)\le I(Y_i;U\mid B=b)\le I(Y;U\mid B=b).
\]

En minimisant, C_W≥h_2(p), d'où égalité. Le mélange uniforme p=1/2 exige **un bit**, dans le modèle de production locale défini. Un produit non dégénéré sur deux cases différentes donnerait aussi une combinaison hybride hors des deux solutions.

Cette loi cible n'est **pas** celle échantillonnée par le NCM de l'étape 3a. Le modèle recevait les puzzles uniques A et B, pas la base ambiguë avec un échantillonnage uniforme des deux solutions. Le script vérifie les deux solutions des 16 bases conservées et les combinaisons hybrides ; l'attribution d'un bit est analytique pour la loi hypothétique spécifiée.

Si un indice partagé J révèle la branche, C_W(Y|B,J)=0. Cela ne mesure pas le calcul nécessaire pour exploiter cet indice. Produire toujours la même solution valide d'une base ambiguë aurait aussi un besoin nul de choix aléatoire : mesurer la diversité est donc indispensable.

## 5. Pourquoi le cas déterministe actuel donne une borne triviale

Pour un puzzle unique complet P, sa solution Y=s(P) est une fonction déterministe. Sachant P=p, la loi est une masse de Dirac, qui est un produit de masses de Dirac par case. U constant est admissible :

\[
C_W(Y_1;\ldots;Y_{81}\mid P)=0.
\]

Avec les poids, conventions et matériel fixés, la génération argmax fournit également des codes grossiers G=g(P) et fins F=f(P). Ainsi :

\[
I(G;F\mid P)=0.
\]

Ces zéros ne signifient ni que les indices n'ont aucun rôle causal ni que résoudre Sudoku est facile. La difficulté de calculer s(P) n'est pas un coût d'information conditionnelle supplémentaire une fois P donné. Une théorie de complexité ou de communication computationnellement contrainte poserait une autre question.

L'information non conditionnelle I(G;F), sur une distribution de puzzles, peut être positive parce que les deux variables dépendent de P. Même l'information conditionnelle change si l'on ne conditionne que sur la famille de base et si l'indice discriminant reste aléatoire. Il faut toujours écrire le conditionnement.

## 6. Trois processus qu'il ne faut pas confondre

**Encodage du tokenizer gelé.** Les codes Q(Y) sont des fonctions d'une réponse fournie à l'encodeur. Leurs dépendances entre positions ou niveaux ne satisfont pas automatiquement une hypothèse de production indépendante. On ne peut pas appliquer au tokenizer la factorisation du générateur sans mesure ou preuve distincte.

**Génération des codes.** Le chemin exécuté prend l'argmax. Dans une autre campagne utilisant des tirages privés indépendants, le modèle parallèle pourrait satisfaire :

\[
p(K_{k,1:m}\mid P,Z_{k-1})=\prod_i p(K_{k,i}\mid P,Z_{k-1}).
\]

Il faudrait vérifier le sampler, l'absence de bruit commun supplémentaire et toute contrainte de tirage. Une attention déterministe entre requêtes calcule les paramètres de distributions ; elle ne couple pas automatiquement les tirages une fois les conditions fixées. L'entrée complète et l'état antérieur sont tous deux des ressources communes : on ne peut pas attribuer une borne au seul état grossier si le préfixe reste disponible.

**Reconstruction finale.** L'interpolation, les résidus convolutifs et le décodeur global peuvent faire dépendre un chiffre de plusieurs codes. La factorisation des codes n'entraîne pas celle des chiffres finaux. Le contre-exemple de parité globale montre pourquoi l'hypothèse de localité n'est pas un détail.

Ces distinctions sont cohérentes avec la formulation [NCM](https://arxiv.org/html/2610.14162v1#S3.SS2) ; leur réalisation dans cette campagne est vérifiable dans les [sources exécutées](../benchmark_ncm_reel/runtime.py).

## 7. Ce que mesure l'analyse auxiliaire des codes

L'analyse est post-hoc sur traces déjà connues, avec script et objets de mesure scellés avant le calcul auxiliaire. Pour chaque famille F, V choisit uniformément l'entrée unique A ou B, puis R_k est le code généré déterministe. La distribution a exactement deux atomes par famille :

\[
I(V;R_k\mid F=f)=\begin{cases}1\text{ bit}&R_k(A)\ne R_k(B),\\0&\text{sinon.}\end{cases}
\]

Résultats : le niveau 5, de longueur 9, reste au code 11 et distingue 0/16 familles. Les niveaux 8 et 9 distinguent chacun les entrées dans 16/16 familles. Les préfixes concaténés 1..9 et les suffixes 10..12 distinguent chaque branche, d'où I(R_1:9;R_10:12|F)=1 bit pour cette distribution finie.

Ce bit est un résultat de deux entrées connues et distinguables. Une table mémorisant les deux vecteurs peut les discriminer ; cela ne prouve ni une lecture sémantique réutilisable, ni la généralisation à d'autres familles, ni une information commune minimale consommée par le réseau. La différence de code pourrait simplement refléter l'indice d'entrée. Il ne s'agit pas d'un estimateur fiable de l'information mutuelle des grands vecteurs sur une population de Sudoku.

Cela exclut néanmoins le raccourci « l'entrée domine parce que toute distinction A/B a disparu des codes grossiers » pour ces traces. La distinction des codes ne garantit pas à elle seule celle de leur reconstruction accumulée z_9 : les tables et transformations peuvent être non injectives. La dominance causale fonctionnelle et la présence d'une information statistique sont deux questions distinctes. Les [valeurs par niveau et famille](results/auxiliary.json) restent disponibles.

## 8. Pourquoi le pooling ne peut pas être ajouté comme cause établie

Le résultat historique concerne une symétrie d'un opérateur précis : moyenner des embeddings additifs sur des fenêtres de mêmes histogrammes efface leur ordre brut. Le niveau concerné a une longueur de **9 positions, au niveau 5**. Le point de coupure de l'étape 3a est le **niveau 9, avec 37 positions**. Ces deux nombres neuf ne désignent pas le même objet.

La normalisation, la hiérarchie résiduelle et les autres niveaux ne sont pas couverts par l'identité brute. Une observation d'un code constant n'établit pas l'absence d'information pertinente dans toute l'accumulation. Les données auxiliaires montrent précisément une distinction A/B ailleurs dans le préfixe.

L'encodeur de solution n'est pas appelé dans la génération native. La [réparation historique du pooling](../correction_pooling/RAPPORT.md) modifiait des cibles d'encodage sans améliorer les générations inchangées. Il faut aussi distinguer cet opérateur de la réduction spatiale des états utilisés comme requêtes par le générateur. L'étape 3a n'a modifié aucun de ces opérateurs.

Pour établir « pooling → perte de coordination → recours compensatoire aux indices », il manque donc : une mesure de perte pertinente, une intervention sur l'opérateur causal correspondant, et une interaction préfixée entre cette intervention et l'accès à l'information d'entrée, à distribution et budgets contrôlés. Les 15/16 diagnostics ne remplacent aucune de ces étapes.

## 9. Programme expérimental non exécuté

Une prochaine campagne pourrait utiliser les bases ambiguës, mais il faudrait d'abord fixer une loi cible de diversité (par exemple les deux solutions équiprobables), un sampler et des graines réservées. Rapporter probabilités de branche, validité, sorties hybrides et diversité ; une solution valide répétée ne suffit pas à reproduire cette cible.

Un témoin à lecteurs effectivement locaux rendrait la borne directe, tandis qu'un lecteur global testerait la part de coordination qu'il réalise. Les budgets de signal commun, accès au préfixe et couplage du lecteur doivent être explicités ; aucun taux optimal n'est déduit de la seule précision.

Pour mesurer I(G;F) du tokenizer gelé, il faudrait un corpus nouveau, des variables de dimension contrôlable, une estimation de biais et une évaluation séparée par famille. Les deux atomes par famille de l'analyse présente ne remplissent pas ces conditions. Pour attribuer un rôle au pooling, sa modification doit correspondre au chemin réellement exécuté ; perturber un encodeur absent de l'inférence ne teste pas ce chemin.

**Formulation retenue pour le papier :** les résultats sont compatibles avec l'entrée comme signal commun de conditionnement ; ils ne montrent ni que ce signal atteint l'information commune de Wyner ni qu'il compense une perte de coordination due au pooling. Le lien théorique rend une expérience future définissable, sans transformer une interprétation compatible en mécanisme démontré.

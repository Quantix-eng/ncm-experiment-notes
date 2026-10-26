# Rapport — estimands et comparaison de méthodes

25 octobre 2026. Spécification analytique pour les nouvelles campagnes. Aucun résultat empirique n'est ajouté ici.

## 1. Les objets à conserver séparément

Pour un receveur `x`, la bonne réponse `y`, l'autre nom `s` et des logits `l` :

- `NLL(l,y) = logsumexp(l) − l[y]` ;
- `LD(l) = l[y] − l[s]` ;
- `centred(l,y) = l[y] − mean_v l[v]` ;
- exactitude à deux candidats : `l[y] > l[s]`, avec départage déclaré ;
- exactitude vocabulaire : `argmax_v l[v] = y`.

L'une de ces mesures ne remplace pas les autres. Une hausse du logit correct peut coexister avec une hausse plus forte des concurrents, donc une NLL dégradée. Une exactitude à deux noms peut masquer un mot tiers mieux classé. Ces possibilités suivent directement des définitions.

L'effet d'ablation classique compare la sortie native et la sortie perturbée avec continuation régénérée. Il mesure la conséquence de l'intervention sur le système complet. L'attribution directe projette la contribution d'un composant vers une direction du lecteur ; elle ne mesure pas tout son effet médié par les blocs suivants.

## 2. Calcul exact de la mesure publiée

Rushing et Nanda définissent `SR = Δcentred_logit − ΔDE_head`, avec normalisation originale pour l'effet direct. La comparaison doit respecter cette convention. [Self-Repair, §2.1 et §3.1](https://arxiv.org/html/2402.15390v2#S2.SS1)

Pour spécifier une implémentation GPT-2 sans ambiguïté, notons :

- `r0` : résidu final natif à la position évaluée ;
- `c0,c1` : contributions projetées de la tête native et donneuse, dans l'espace résiduel ;
- `P(c)=c−mean(c)` : projection de centrage sur les dimensions cachées ;
- `sigma0=sqrt(mean(P(r0)^2)+eps)` : dénominateur LayerNorm natif ;
- `gamma,beta` : paramètres de la normalisation finale ;
- `w_y = W_U[y] − mean_v W_U[v]` : direction du logit centré.

L'effet direct du remplacement, dans cette convention, est :

`delta_DE = dot(w_y, gamma * P(c1−c0) / sigma0)`.

La mesure publiée réimplémentée pour le remplacement est :

`SR = centred(l11,y) − centred(l00,y) − delta_DE`.

`c1` est l'activation donneuse réellement substituée. L'approximation `delta_DE = −DE(c0)` n'est pas utilisée : un donneur IOI peut contribuer lui-même au nom correct ou à son concurrent. Le biais partagé de la projection d'attention n'est pas ajouté une fois par tête ; la décomposition doit reconstruire le bloc avec ce biais une seule fois.

Une version fondée sur `w_y−w_s` produit un objet analogue pour le logit-difference, à étiqueter comme **adaptation de la mesure au contraste de noms**, distincte de la reproduction du logit centré. Un rapport ne doit pas changer de convention entre méthodes ou exemples.

## 3. Séparer la lecture finale des contributions aval

Définissons notre lecteur à facteur figé :

`F0(r) = W_U (gamma * P(r) / sigma0 + beta) + bias_U`.

Le centrage `P(r)` porte toujours sur le résidu courant. Figer le seul dénominateur n'impose pas de conserver la moyenne native. Pour un résidu final régénéré `r11`, une identité utile de vérification est :

`SR = SR_residual + SR_reader`,

`SR_residual = centred(F0(r11),y) − centred(F0(r0),y) − delta_DE`,

`SR_reader = centred(l11,y) − centred(F0(r11),y)`.

Cette décomposition est une spécification algébrique du présent adaptateur, pas une revendication de théorie nouvelle. Si les contributions aval sont figées exactement et que seul le remplacement de tête change `r`, `SR_residual` doit être nul à l'erreur numérique près. Une valeur non nulle de `SR_reader` reste compatible avec aucune adaptation de bloc aval. C'est précisément la distinction qu'une conclusion de compensation nécessaire doit respecter.

## 4. Contrastes croisés

Avec `z0` l'état natif, `z1` l'état perturbé, `T0` la continuation native et `T1` celle adaptée à la perturbation :

- `Y00 = D(z0,T0)` ;
- `Y10 = D(z1,T0)` ;
- `Y11 = D(z1,T1)` ;
- `Y01 = D(z0,T1)` ;
- `Y1wrong = D(z1,Tother)`.

La continuation est une suite de contributions figées, ajoutées dans l'ordre de l'implémentation. Le lecteur final et sa liberté de normalisation doivent être déclarés pour chaque bras. Avec `Lab=NLL(Yab,y)` :

`damage = L10−L00` ; `recovery = L10−L11` ;

`interaction = L10+L01−L11−L00` ; `specificity = L1wrong−L11`.

Les autres métriques ont leur propre signe et leur propre unité ; on ne les combine pas dans une moyenne de scores hétérogènes.

Une continuation adaptée qui réussit sans dommage figé suffisant n'établit pas une récupération utile. Un résidu qui change n'établit pas une amélioration. Une valeur positive de spécificité ne suffit pas à localiser le mécanisme : une suite mal appariée peut aussi être spécialement nuisible.

Les différents donneurs de `T1` et `Tother` interviennent au même site. Ce contrôle réduit le changement de support, sans égaliser automatiquement leur norme ni garantir qu'ils restent dans une distribution d'états naturels. Les normes sont conservées et leur traitement est fixé avant test.

## 5. Corroboration indépendante de l'affirmation

Soit `B` un ensemble de backups strictement aval, annoncé depuis la littérature. `Y_B` régénère la continuation perturbée, mais rejoue les contributions natives des composants de `B`. `Y_C` réalise le même blocage sur un ensemble témoin annoncé, de mêmes cardinalité et distribution par couche. Posons :

`block_effect = NLL(Y_B,y)−NLL(Y11,y)` ;

`control_effect = NLL(Y_C,y)−NLL(Y11,y)` ;

`selective_block_effect = block_effect−control_effect`.

Une perte de récupération lorsque `B` est empêché de s'adapter, supérieure au témoin selon la marge fixée, corrobore sa contribution dans ce contrat. Elle ne prouve ni l'exclusivité de `B`, ni une compensation universelle, ni un calcul cognitif particulier. Le blocage peut avoir des interactions : rapporter les sorties et pas uniquement le contraste final.

Une absence d'effet invalide l'attribution restreinte à `B` telle qu'annoncée, sans éliminer les contributions des MLP, d'autres têtes ou du lecteur. Les données de ce test ne servent pas à remplacer `B` par un nouvel ensemble favorable dans la même évaluation réservée.

## 6. Comparaisons équitables et valeur ajoutée

Les méthodes numériques existantes reçoivent leur nom et leur estimand. On ne transforme pas rétroactivement une attribution directe en prédiction de récupération de NLL. L'évaluation compare : ablation et attribution directe ; self-repair publié ; protocole complet ; protocole sans `Y10` ; protocole sans `Y1wrong`.

Les deux variantes amputées ne sont pas des méthodes indépendantes de la littérature ; elles testent l'utilité marginale de nos contrôles. EAP-IG est une méthode de localisation, donc reste hors d'une comparaison de précision d'attribution tant qu'une sortie commune n'est pas spécifiée. Cette séparation suit les deux pistes de MIB. [MIB](https://proceedings.mlr.press/v267/mueller25a.html)

Une valeur ajoutée exige une conclusion supplémentaire précise, annoncée par la méthode, puis corroborée par une intervention non utilisée pour la produire. Un simple désaccord numérique, une reproduction de résultat connu ou un taux d'abstention élevé ne satisfait pas ce critère. Si aucune conclusion supplémentaire n'est corroborée, le rapport doit indiquer **valeur ajoutée scientifique non établie**.

Les effets sont agrégés au niveau des familles sources, donneurs et variantes conservés ensemble. Les résultats par template et les échecs natifs restent visibles. Les marges, seuils de dommage et règles d'abstention effectifs appartiennent au protocole gelé de la campagne, sans importer automatiquement ceux des anciens puzzles Sudoku.

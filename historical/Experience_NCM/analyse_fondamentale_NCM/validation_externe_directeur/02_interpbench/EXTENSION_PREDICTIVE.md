# Extension à des prédictions interventionnelles réservées

25 octobre 2026. Spécification préparatoire ; elle n'a pas été évaluée sur les quatre tâches réservées.

## Objet évaluable

Une méthode reçoit un contrat d'accès, une source, des observations autorisées et la description d'une intervention future. Elle retourne l'une des sorties suivantes, avec l'identifiant exact de la requête : `abstain`, `unchanged_defined_outputs`, `changed_at_positions(S)` ou `categorical_prediction(values, mask)`. Une affirmation n'est jamais universelle : elle porte sur cette requête nommée. Une phrase telle que « cette tête est inutile » n'est pas admissible comme prédiction sans préciser variable, support, donneur, lecteur et métrique.

Le score primaire pour une prédiction complète est la proportion d'affirmations émises contredites par la sortie du programme externe sur son masque de positions définies. Le score d'invariance exige l'identité de toutes ces positions. Une méthode qui prédit seulement un changement ne gagne pas le score d'une prédiction catégorielle complète. Les types sont rapportés séparément. La couverture est le nombre d'affirmations émises divisé par le nombre de requêtes autorisées ; aucune précision conditionnelle ne se lit sans sa couverture.

## Trois accès et deux banques

O offre uniquement entrées, références et trajectoires natives. C ajoute les cinq sorties croisées sur des donneurs de proposition, au même site. E ajoute les interventions entrée/état prévues et les ablations de composants annoncés. Ces banques servent à former les hypothèses, pas à les valider.

Une seconde banque emploie des sources/familles et donneurs distincts pour la falsification. Le résultat de son programme externe reste inaccessible jusqu'au gel des prédictions de toutes les méthodes. Les branches d'une même source, les variations de longueur et tous ses donneurs restent groupés. Les effectifs, le nombre de donneurs, les variables choisies, les tolérances continues éventuelles et les règles d'abstention doivent être scellés dans une version exécutable avant génération de ces résultats.

Le pilote actuel ne fixe pas ces règles à partir de ses sorties favorables. Il teste seulement deux invariances annoncées par les programmes externes. Choisir aujourd'hui un seuil sur les faibles effets des deux tâches triviales ne donnerait aucune justification confirmatoire sur les quatre tâches réservées.

## Oracle et comparaisons

L'oracle est le programme RASP/Tracr avec son alignement publié, exécuté sous le même échange de variable native. Son identité n'est ni une classe de notre bibliothèque, ni une réponse produite par notre diagnostic. Pour les têtes non alignées, l'oracle SIIT prédit l'invariance ; pour une variable alignée, il fournit une sortie contrefactuelle. Dans les deux cas, on mesure également la conformité du réseau au programme.

IIA et SIIA fournissent cette conformité réseau/oracle et un antécédent indispensable. Une méthode de prédiction nouvelle doit être comparée à d'autres méthodes produisant le même objet, avec le même accès déclaré, ou présentée comme une nouvelle couche d'évaluation au-dessus de ces mesures. Aucun taux de « classification correcte » ne doit être inventé pour IIA, EAP ou une méthode de localisation qui n'émet pas les mêmes propositions.

Une ablation de contrôle enlèvera séparément les observations figées et les donneurs mal appariés, en gardant les mêmes requêtes finales. Le bénéfice exigera une diminution des contradictions à couverture comparable, ou une couverture accrue à risque comparable, par tâche. Une différence numérique ou une simple abstention accrue ne suffit pas à établir une amélioration scientifique.

## Sorties indéfinies et hors domaine

Le masque des positions définies est recalculé par l'oracle pour chaque intervention. Une classe arbitraire produite à une position `None` du programme ne devient pas une erreur du diagnostic. Les changements du masque sont conservés comme résultat distinct. BOS/PAD sont exclus selon les conventions officielles.

Les échanges de têtes entières et de MLP natifs respectent la granularité annoncée. Des interventions Q/K/V, bruitées ou nulles demanderaient une nouvelle validation du domaine oracle/réseau. Aucun succès du présent dispositif ne leur est automatiquement transféré.

## Critère d'achèvement

Une évaluation est achevée seulement quand les prédictions sont gelées avant leurs réponses réservées, que les sorties programme/réseau sont conservées, que couverture et contradictions sont rapportées ensemble et que les limites d'accès sont explicites. Cette spécification ne représente ni une implémentation accomplie de toutes les méthodes, ni une validation déjà obtenue.

# Journal — composition terminale

## Avant les nouvelles observations

- Lecture de l'analyse du directeur, du protocole terminal historique, du rapport et des sources d'inférence.
- Question : les effets de deux substitutions sémantiques se maintiennent-ils après décodage, au-delà des échecs déjà présents pour les substitutions isolées ?
- 32 sources fixées : indices 512 à 543 du corpus terminal conservé, distinctes des indices 0 à 127 de l'ancienne expérience causale. Les performances natives de ce corpus ont déjà été analysées ; absence de nouvel échantillon indépendant explicitement déclarée.
- Cinq paires tirées par source à partir des seuls indices, une par configuration. 64 doubles écritures et 16 écritures isolées par paire. Aucun ajustement de la table historique.
- Seuils exploratoires de dégradation : rétention conditionnelle < 95 %, ou perte nette > 5 points par rapport à la conjonction des contrats isolés. Seuils de décision pratique, sans justification confirmatoire indépendante.
- Protocole, scripts, sources et entrées épinglés dans `results/protocol_frozen.json` avant le lancement du runner. La vérification des empreintes est répétée avant et après les expériences.
- Aucune étape 2 ou 3 lancée : arrêt demandé après le pilote.

## Après exécution et audit

- Runner terminé à 12:37:57 UTC : 32 sources, 10 240 doubles éditions, 2 560 isolées, sans changement des critères ni des scripts gelés.
- Deux cibles : 5 644/10 240 ; contrat complet : 5 576/10 240 ; rétention conditionnelle : 5 206/5 504. Le seuil local de rétention est franchi de peu ; l'intervalle descriptif global recouvre 95 %.
- 210 pertes parmi 1 182 conjonctions isolées réussies entre voisines ; 298 pertes globales, contre 370 gains. Pas d'effondrement global établi de la moyenne.
- Vérification scalaire, intervalles groupés par puzzle et 400 rejeux singleton passés. Empreintes scientifiques historiques et gel local inchangés.
- Analyse descriptive supplémentaire après lecture des résultats : séparation des chiffres identiques et différents, conservée dans `audit_target_equality.py` et `results/posthoc_target_equality_audit.json`. Entre voisines, 121/130 contrats perdus avec chiffres identiques, 89/1 052 avec chiffres différents. Aucun nouveau critère, aucune nouvelle inférence neuronale.
- Décision : ne pas lancer de projet architectural sur cette règle en l'état. Arrêt après l'étape 1 ; rapport et observations conservés. Aucun changement du manuscrit.

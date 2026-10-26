# Journal concis

- Demande : approfondir les explications positives et les expériences discriminantes, sans présupposer un mécanisme à confirmer. Les campagnes antérieures restent inchangées.
- Audit du code : les argmax du décodeur intermédiaire ne sont pas réinjectés dans le générateur ; le décodeur comporte lui-même de l'attention. Distinction entre argmax des têtes et plus-proche-voisin de l'encodeur.
- Exploration des traces existantes : comparaison génération/codec oracle, puis formalisation de la réanalyse avec bootstrap et vérification des entrées. Les données avaient déjà été examinées ; ce n'est pas un test aveugle.
- Correction de vérification, avant rejeu réussi : les configs et poids sont épinglés dans input_manifest.json, lui-même contrôlé par final_manifest.json, plutôt que directement dans ce dernier. Aucun résultat ni critère scientifique changé.
- Rejeu oracle réussi sur 16 solutions : codes et douze reconstructions identiques, sans appel aux blocs/têtes du générateur. Les expériences de compensation, localisation et transfert restent proposées, non exécutées.
- Vérification de la recette : présence du fine/nested dropout dans configuration, code et description de l'article. Ne pas supposer une supervision exclusivement finale, ni déduire l'historique précis d'entraînement de la seule configuration.

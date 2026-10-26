# B0 — oracle externe Craft et masques RASP, sur développement

26 octobre 2026 UTC (date effective portée par le gel). Ce plan est écrit après A1. Les sources, critères et données sont scellés avant les nouveaux calculs.

L'oracle suit les appels officiels `static_validate`, `extract_rasp_graph`, `infer_bases`, `add_craft_components_to_rasp_graph` puis `craft_graph_to_model`, en s'arrêtant avant l'assemblage JAX. Les fonctions des programmes sont extraites sans modification de leurs corps depuis les cas publiés ; les classes d'entraînement sont seules écartées. L'oracle ne reçoit aucune sortie de notre diagnostic ni de réseau appris.

Les embeddings représentent exactement les directions token, indice et constante un du compilateur. BOS n'a pas d'indice ; les tokens ont des indices commençant à zéro. Le masque causal déclaré par la configuration est appliqué aux têtes comme dans l'assembleur final. L'emplacement de chaque moitié de bloc suit l'énumération attention/MLP du code d'assemblage.

Pour les cas 8 et 33 seulement : contrôler 128 sources chacun, réponses natives Craft contre RASP puis contre les références antérieures. Deux donneurs par source, distincts et de même longueur utile, sont tirés par les seuls identifiants/données. Sites fixés : L0H0, MLP0 et L1H0. Les têtes non alignées ont la prédiction d'invariance du programme ; MLP0 échange tous les noeuds produits par le bloc aligné. Le RASP indépendant est évalué en substituant les valeurs natives des expressions correspondantes ; le Craft indépendant échange la contribution du composant.

Exiger une identité catégorielle Craft/RASP sur toutes les positions RASP définies pour chaque intervention ; arrêter au premier désaccord, sans réseau réservé. Rapporter séparément la conformité du réseau appris à cet oracle, y compris ses divergences. Aucun échec du réseau à suivre son programme n'est attribué au diagnostic.

Le masque est obtenu par l'évaluateur RASP officiel et sa propagation de `None`, jamais par le modèle appris. Deux contrôles génériques fixés vérifient un décalage avec première valeur indéfinie, puis la propagation de cette valeur à travers une transformation. Ils ne valident pas encore les quatre programmes réservés sous toutes les interventions.

Les quatre cas réservés peuvent être compilés pour exporter leur schéma, sans appliquer leur réseau ou calculer les réponses du jeu réservé. L'alignement statique devra être comparé aux arêtes publiées et les labels auxiliaires sans expression RASP explicite seront signalés ; aucune prédiction de masque ne sera fabriquée pour eux.

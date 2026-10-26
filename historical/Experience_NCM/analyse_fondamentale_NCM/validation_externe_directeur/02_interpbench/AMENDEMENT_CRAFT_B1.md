# Amendement B1 — import des seules étapes NumPy

25 octobre 2026 UTC. B0 a échoué avant tout calcul de modèle ou programme : l’initialiseur du paquet `tracr.compiler` importe l’assembleur JAX et requiert chex. Les sous-modules utilisés jusqu’à Craft n’en ont pas besoin. B1 crée donc le namespace Python `tracr.compiler` avec le chemin officiel, puis importe les sous-modules officiels inchangés. Aucun compilateur ou oracle numérique n’est réécrit. Les anciens fichiers B0 et son échec sont conservés.

Les données, variables, critères et masques de B0 restent identiques. Le code nouveau et ses imports sont gelés avant exécution. Une vérification séparée conserve le fait qu’aucun module jax/haiku/chex n’est importé. L’oracle ne reçoit aucune sortie de diagnostic.

Erratum de date : la première phrase de PROTOCOLE_CRAFT_B0.md porte à tort le 26 octobre ; le gel effectif UTC, conservé, date du 25 octobre 2026 à 21:11:24.098180. Cette coquille n’a changé aucun calcul ni critère.

# Recherche d'un effet positif : alphabet terminal et décodage certifié

- Explorer seulement les anciens train/validation ; conserver les pistes écartées dans le journal. Aucun résultat favorable ne sera garanti à l'avance.
- Hypothèse issue de cette exploration : le niveau terminal porte une identité de chiffre largement lisible par simple table. Tester la causalité par substitution ciblée, puis comparer au bruit local de même norme.
- Application candidate : accepter la grille lue par table seulement si contraintes et indices sont satisfaits ; sinon utiliser le décodeur natif. Sur un puzzle à solution unique, cette règle ne peut pas dégrader la réussite exacte, indépendamment de la qualité de la table.
- Faire un pilote sur validation, puis geler mapping, interventions, métriques et marges avant génération d'un nouveau corpus. Vérifier unicité et exclusions de familles déjà étudiées.
- Mesurer séparément lecture prédictive, effet causal, couverture de certification, qualité et temps réel. Une économie de décodeur ne garantit pas une accélération de toute la génération.
- Livrer rapport, article, code exécuté, traces, contrôles et manifestes. Préserver toutes les campagnes précédentes. Les jeux déjà examinés restent exploratoires.

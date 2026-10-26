# Contribuer

Utiliser l'environnement et les actifs épinglés du README. Garder les poids officiels inchangés et le CPU FP32 comme référence. Exécuter les tests puis la comparaison au code original après toute modification numérique ; conserver les frontières des lots.

Ajouter une entrée concise à `FEUILLE_DE_ROUTE.md` : modification, protocole, résultat, décision. Un gain de vitesse doit préciser matériel, versions, batch, threads, démarrage inclus ou exclu et répétitions. Vérifier les logits et les codes, pas seulement les réponses. Ne pas assouplir un seuil après avoir vu les résultats sans déclarer un nouveau protocole.

Les expériences MPS restent explicitement expérimentales. Ne pas introduire de fallback CPU silencieux, quantification ou réparation des réponses dans le chemin strict. Ne pas committer poids, données, environnement virtuel ou réponses individuelles ; publier les téléchargements épinglés et les résultats agrégés.

La référence CPU historique est `254bb49`. Le dépôt amont nécessaire aux comparaisons doit rester au commit indiqué dans `UPSTREAM.md`, sans modifications.

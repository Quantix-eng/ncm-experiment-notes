# Contrôle post-hoc du régime de dropout

Motif : au niveau 5, les statistiques EMA sauvegardées sont positives pour tous les codes, alors que l'encodage déterministe de solutions n'en utilise qu'un sur hard. La configuration d'évaluation indique un dropout d'encodeur de 0,3 ; celui-ci est désactivé par eval(). Il peut rompre l'invariance des moyennes par ligne démontrée sans dropout.

Test exploratoire sur les 64 premières solutions de train, pour chaque tokenizer :
- Garder tous les poids gelés et les modules VQ/decoder en évaluation.
- Comparer encoder.drop en évaluation à encoder.drop actif à sa probabilité configurée, sur quatre graines 73130–73133.
- Quantifier utilisation/perplexité des codes du niveau 5, variabilité du résidu sous-échantillonné et différence des codes entre répétitions.
- Vérifier l'identité du state_dict avant/après : aucune mise à jour EMA ni entraînement.

Ce test constitue une intervention sur le régime d'encodage d'un checkpoint gelé. Il ne reconstitue pas l'entraînement original et ne démontre pas que le dropout est l'unique cause historique de la répartition des codebooks.

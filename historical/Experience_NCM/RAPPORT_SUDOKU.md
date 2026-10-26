# Sudoku sur M1 — résultat du 22 octobre 2026

**Parité bit à bit : 384/384 grilles. Accélération mesurée : 2.39×.**

CPU FP32, batch 8, quatre threads. Génération et décodage : 94.88 s pour l’original contre 39.62 s avec cache, soit 9.7 grilles/s. Chauffe exclue ; ordre original/optimisé alterné. Mesures locales, pas une garantie de débit sur une autre machine.

- Bug des instantanés intermédiaires corrigé : copies indépendantes, reconstruction finale inchangée.
- Contexte du puzzle calculé une seule fois ; accumulateur multiscalaire réutilisé au décodage.
- Poids, FP32, température nulle et argmax conservés. Aucun correcteur de Sudoku ajouté.
- Codes, représentation finale, logits et réponses comparés avec `torch.equal`, sans tolérance.

- easy : 127/128 résolutions exactes, comme l’original ; 2.20× plus rapide sur cet essai.
- medium : 125/128 résolutions exactes, comme l’original ; 2.54× plus rapide sur cet essai.
- hard : 110/128 résolutions exactes, comme l’original ; 2.61× plus rapide sur cet essai.

**Limite :** 128 indices répartis dans chacun des trois jeux officiels, soit 384 grilles sur 6 000. Pas de validation exhaustive ni de garantie entre CPU/GPU ou versions différentes. La référence est le code original exécuté sur CPU, pas une reproduction des sorties CUDA des auteurs. Avant publication, étendre la comparaison avec `--n 0`.

Trois tests de régression passent. Paquet installable, licence/provenance conservées, suivi Git local ; rien publié. Les mesures RSS du processus de comparaison comprennent deux modèles et ne décrivent pas la mémoire d’un déploiement unique.

Preuves : [parité et indices](audit/sudoku-parity.json), [synthèse chiffrée](audit/sudoku-summary.json), [journal](FEUILLE_DE_ROUTE.md), [installation et reproduction](README.md).

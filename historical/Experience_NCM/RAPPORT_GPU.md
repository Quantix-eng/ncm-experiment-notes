# GPU Sudoku — 22 octobre 2026

**MPS fonctionne sans repli CPU ; la version CPU `254bb49` reste la référence et le choix par défaut.**

Interpolation adaptative remplacée par sélection/somme/division sur Metal ; cache du contexte conservé. FP32, poids et règles de décodage inchangés. Aucun module CPU de référence modifié.

## Fidélité

- Batch 8 : 384 cas, 0 réponses différentes et 0 grilles avec codes différents. Écart maximal de probabilité à conditionnement identique : 0.000103 ; KL maximale 2.18e-08.
- Batch 32 : 384 cas, 0 réponses différentes et 0 grilles avec codes différents. Écart maximal de probabilité à conditionnement identique : 7.77e-05 ; KL maximale 1.91e-08.

Les mêmes 128 indices par difficulté sont testés aux deux batchs : **384 grilles distinctes**, pas 768. Diagnostic : softmax des logits à T=1, mesuré en FP64 ; l'inférence reste FP32 avec argmax à T=0. Le seuil exploratoire d'écart absolu `1e-5` est dépassé aux deux batchs et n'est pas assoupli après observation. Identité des réponses observée ne signifie pas identité des distributions.

## Vitesse et mémoire

Médianes de cinq répétitions après chauffe ; ratios CPU optimisé / MPS (>1 favorise MPS).

- easy : batch 1 : 0.22× ; batch 8 : 0.82× ; batch 32 : 1.69×.
- medium : batch 1 : 0.25× ; batch 8 : 0.89× ; batch 32 : 0.30×.
- hard : batch 1 : 0.21× ; batch 8 : 0.84× ; batch 32 : 1.21×.

Ces mesures sont sensibles à l'état de la machine. Les séries CPU puis GPU ne sont pas appariées ; un contrôle medium sur trois paires alternées, libération des caches incluse, est conservé séparément. Les allocations du pilote sont relevées en fin de série, pas un pic mesuré.

- Contrôle medium batch 8, coût de libération inclus : CPU 0.57 s, MPS 1.21 s ; pilote après libération ≤ 1.14 Go.
- Contrôle medium batch 32, coût de libération inclus : CPU 3.15 s, MPS 2.26 s ; pilote après libération ≤ 1.16 Go.

Contrôle du pooling sur entrée aléatoire : écart maximal MPS/CPU 2.38e-07. Adaptateur seul sur CPU, 8 grilles par difficulté : aucune réponse différente, écart maximal de probabilité du générateur 6.68e-06. Cela distingue une contribution du pooling des autres écarts de backend, sans établir un budget d'erreur exhaustif.

**Décision :** conserver le CPU pour la publication actuelle. La piste Metal est reproductible mais reste expérimentale ; elle nécessite un bénéfice stable et une validation élargie de fidélité avant promotion. Le corpus complet de 6 000 grilles n'a pas été évalué ici.

Preuves : [batch 8 et benchmarks](audit/sudoku-mps.json), [batch 32](audit/sudoku-mps-b32.json), [contrôle mémoire](audit/sudoku-mps-memory.json), [contrôle du pooling](audit/sudoku-mps-pooling-control.json), [protocole](experiments/README.md), [journal](FEUILLE_DE_ROUTE.md).

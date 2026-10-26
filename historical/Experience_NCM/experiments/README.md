# Expérience GPU Sudoku

Référence CPU : commit `254bb49`. Ses modules de calcul sont contrôlés octet par octet avant exécution. Le chemin CPU par défaut n'est pas modifié.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 .venv/bin/python experiments/sudoku_mps.py
```

FP32 uniquement, poids officiels, trois difficultés, 128 indices répartis par difficulté. Pooling adaptatif remplacé uniquement dans l'instance expérimentale : les intervalles sont préparés à l'initialisation, puis sélection/somme/division s'exécutent sur MPS, sans transfert CPU dans l'inférence. Cache du préfixe conservé.

Contrôles : réponses et codes en génération libre ; distributions de chaque niveau en imposant les codes CPU précédents ; distributions du décodeur à codes identiques. Softmax de diagnostic à **T=1**, calculé en FP64 sur CPU pour mesurer les différences des logits FP32 sans ajouter un second écart de backend. Le Sudoku reste décodé à **T=0 par argmax** : ce diagnostic ne constitue pas un échantillonnage du modèle. Cette mesure est distincte de l'inférence GPU.

Critères exploratoires de fidélité : aucune réponse différente ; maximum d'écart absolu de probabilité ≤ `1e-5`, distance de variation totale ≤ `1e-4`, KL(CPU‖MPS) ≤ `1e-6` sur les distributions conditionnées identiquement. Ces seuils sont des tolérances de recherche, pas une preuve d'équivalence universelle ni un critère de significativité statistique. Une divergence de codes est signalée même si la réponse finale reste identique. Toute promotion exige une validation élargie.

Vitesse : batchs 1/8/32, deux chauffes puis cinq répétitions synchronisées ; préfixes déjà sur le bon appareil. Temps incluant génération et décodage ; transferts initiaux et analyses exclus. Comparaison avec le CPU optimisé, pas avec le code amont plus lent. Allocations MPS relevées en fin d'essai ; pas assimilées à un pic.

Contrôles complémentaires : `mps_memory_benchmark.py` mesure trois paires alternées sur medium, avec le coût de `empty_cache()` inclus ; `pooling_control.py` sépare l'effet de l'adaptateur sur CPU et son erreur sur entrée aléatoire MPS. `summarize_mps.py` régénère le rapport à partir des fichiers JSON. Pour comparer au CPU8 archivé, les réponses locales doivent correspondre au SHA-256 du rapport du commit `254bb49` ; l'expérience batch 32 effectuée ici a vérifié cette correspondance.

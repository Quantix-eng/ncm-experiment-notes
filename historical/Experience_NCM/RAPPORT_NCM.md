# NCM sur M1 — audit du 22 octobre 2026

> Audit initial historique sur TinyStories, antérieur aux correctifs. Pour l'état actuel sur Sudoku : [README](README.md) et [journal](FEUILLE_DE_ROUTE.md).

**Verdict : TinyStories fonctionne sur ce Mac avec une marge mémoire raisonnable. Le CPU est actuellement plus rapide que notre chemin GPU hybride.**

## Travail effectué

- Dépôt cloné et laissé intact : `upstream-ncm/`, commit `c70274273f489abc73832b2b98584a8483b6b203`.
- TinyStories complet téléchargé : **578,23 Mo**, générateur, VQ-VAE, configuration et BPE. Révision Hugging Face `d342ce8aaea58830eef3b8ad473664673e41f692`. Tailles vérifiées, SHA-256 des quatre fichiers LFS conformes : `audit/download-manifest.json`.
- Environnement isolé `.venv/` : **Python 3.13.1 ARM64, PyTorch 2.8.0**. Chargement tensoriel `weights_only=True` et correspondance stricte des poids réussis ; dépendances verrouillées dans `audit/requirements-lock.txt`.
- Machine vérifiée : M1, CPU/GPU 8 cœurs, **8 Go unifiés au total**, macOS 14.0. Environ **8–9 Gio de disque libres après installation** ; valeur fluctuante.

## Architecture et logiciels

NCM génère des codes discrets multiscalaires, puis les décode avec un VQ-VAE. « Concept » est ici une désignation technique ; aucune capacité métacognitive n’en découle.

TinyStories totalise **125,04 M de paramètres** : générateur 88,61 M + VQ-VAE 36,43 M. Douze niveaux `[1,8,16,24,32,40,52,64,76,92,108,128]`, premier niveau tiré de l’unigramme fourni, puis 11 passes du générateur. Les 641 codes cumulés donnent **128 tokens BPE**. Ce checkpoint produit des débuts d’histoires en anglais ; il n’est pas entraîné comme assistant conversationnel.

Code : PyTorch/SDPA, RoPE, normalisation Q/K, convolutions/interpolations ; Hydra/OmegaConf, NumPy, regex. Transformers/Hugging Face servent aux autres tokenizers et à l’évaluation ; ROUGE/sacreBLEU aux métriques. Aucun kernel CUDA externe obligatoire dans le chemin testé. Aucun portage MLX/Core ML/GGUF fourni ; dépendances amont peu contraintes.

## Résultats locaux

FP32, batch 1, quatre threads CPU, une chauffe puis trois générations ; synchronisation MPS. Temps incluant génération et décodage tensoriel, excluant chargement et conversion en texte.

- **CPU : médiane 0,80 s**, plage 0,70–1,18 s ; pic RSS **0,78 Go**.
- **MPS natif : échec** du pooling adaptatif avec dimensions non divisibles.
- **MPS hybride : médiane 1,42 s**, plage 1,41–1,44 s ; pic RSS **0,83 Go**. Seule l’interpolation `area` passe explicitement sur CPU. Allocations du pilote MPS en fin d’essai : **1,16 Go**, pas un pic mesuré.

Paramètres et buffers : **648 Mo** en mémoire CPU. RSS et allocations MPS ne s’additionnent pas directement en mémoire unifiée. Les sorties ont 128 tokens et des logits finis, mais présentent des incohérences. Trois essais prouvent la faisabilité, pas l’équivalence de qualité ni un débit stable. Les accélérations publiées concernent un **H100 80 Go, batch 16**.

## Problèmes à anticiper

1. **CUDA imposé** dans les scripts officiels : remplacement des `.cuda()`, du lancement multi-GPU et des chargements CUDA nécessaire. Notre sonde les contourne. Déplacer séparément VQ-VAE et générateur : le premier est stocké hors de l’arbre des modules.
2. **Pooling MPS incompatible confirmé** : un changement de `device` seul ne suffit pas. Le repli CPU explicite fonctionne, mais introduit des transferts.
3. **LM1B utilise FP64** dans son post-traitement, incompatible MPS : exécuter sur CPU ou valider une conversion FP32. Préserver ses pénalités de décodage.
4. **Bug reproduit** : `idx_to_zhats` partage le même accumulateur entre les instantanés `z_hats`. Cloner avant toute étude des représentations intermédiaires ; décodage final testé exploitable.
5. **Publication incomplète** : OpenWebText annoncé mais uniquement `owt/TODO.md` côté poids ; aucun pipeline d’entraînement ni tests automatisés dans le dépôt inspecté.

## Plan priorisé

1. **Référence CPU FP32, batch 1** ; corriger l’aliasing pour les expériences. Libérer du disque avant données et évaluateurs.
2. **Pooling Metal équivalent**, validé numériquement sur entrées identiques ; comparer ensuite CPU/MPS, batchs 1/2/4, répétitions, mémoire et swap.
3. **Optimisations ciblées** : éviter le tri inutile lorsque `top_p=1`, puis essayer l’autocast FP16. `.half()` seul peut échouer : accumulateurs/masques créés en FP32. Mesurer la qualité sur plusieurs graines ; même graine ≠ mêmes échantillons entre backends.
4. **Évaluer séparément** : décharger NCM avant GPT-2 large (~774 M de paramètres). Reproduire le protocole avant toute revendication de scores. Quantification/MLX seulement si le profilage le justifie ; entraînement complet hors première étape.

Autres checkpoints complets, tailles distantes : Sudoku easy **162 Mo**, XSum **654 Mo**, LM1B small **988 Mo**, WMT14 **1 126 Mo**, LM1B medium **1 866 Mo**, large **3 499 Mo**. Non téléchargés ; tailles disque, pas pics RAM.

## Reproduction et preuves

```sh
.venv/bin/python scripts/probe_ncm.py --device cpu --output audit/probe-cpu.json
.venv/bin/python scripts/probe_ncm.py --device mps --area-cpu --output audit/probe-mps-hybrid.json
```

Les JSON contiennent mesures et textes ; `audit/probe-mps.log` conserve l’échec natif. MPS est masqué dans le bac à sable de l’agent mais vérifié disponible hors de ce contexte. Aucun fallback global silencieux activé.

Sources : [article v1, §4.6 et annexes C/I](https://arxiv.org/html/2610.14162v1), [dépôt](https://github.com/HazyResearch/ncm), [poids à révision fixe](https://huggingface.co/hazyresearch/ncm-checkpoints/tree/d342ce8aaea58830eef3b8ad473664673e41f692), [bug #11](https://github.com/HazyResearch/ncm/issues/11), [discussion timings #9](https://github.com/HazyResearch/ncm/issues/9), [PyTorch MPS 2.8](https://docs.pytorch.org/docs/2.8/notes/mps.html). Les témoignages des issues ne sont pas nos propres reproductions.

# Lecture et reproduction de C2/C3

Le [rapport](RAPPORT_C2.md) fait autorité pour cette extension ; le premier `RAPPORT.md` décrit le pilote historique. B0/B1 et leurs échecs sont conservés. L'évaluation ne couvre pas tous les composants et n'est pas une réplication humaine indépendante.

## Artefacts

- `data/c2_plan.json` : sites, auxiliaires exclus et trois donneurs distincts par famille.
- `results/c2_frozen.json` : sources, scripts et paramètres avant forwards réservés.
- `results/c2_predictions_frozen.json` : prédictions avant les réponses du troisième donneur.
- `results/c2_probes.json`, `c2_falsification.json`, `c2_E.json` : observations, masques et conditions E.
- `results/c2_probe_logits.npz`, `c2_future_logits.npz` : logits FP32, sans pickle.
- `results/c2_summary.json` : toutes les tâches, tous les seuils et risques distincts.
- `results/c2_verification.json` : audit scalaire et intégrité.
- `results/c3_padding.json`, `c2_mask_and_c3_audit.json` : contrôles post-hoc sans remplacement de C2.

## Audit sans nouveau réseau

Depuis ce dossier, `../../.venv/bin/python verify_c2.py` recalcule les contrôles de score, d'argmax et d'intégrité. `../../.venv/bin/python audit_masks_c2.py` refait la clarification post-hoc des masques et le contrôle C3. Ces scripts écrivent leurs rapports d'audit déterministes, sans modifier les observations.

## Recalcul des expériences

Phases exécutées : `reserved_c2.py --phase prepare`, puis `--phase probes`, puis `--phase falsify`, puis `analyse_c2.py`. C3 est exécuté séparément par `padding_c3.py`. Ne pas relancer ces phases dans le dossier de référence : elles écriraient les fichiers d'expérience. Pour rejouer, copier d'abord le dossier dans un nouveau dossier frère, conserver ses sources/dépendances et manifestes, puis créer ses nouveaux gels et sorties. Ce rejeu est une reproduction informatique, pas une nouvelle validation réservée.

Les dépendances optionnelles restent isolées dans `deps`, avec les bibliothèques Transformer historiques en lecture. Les modules réellement importés sont décrits dans `results/environment_actual_imports.json` ; les anciennes métadonnées d'installation conservées ne doivent pas être confondues avec la version du module chargé.

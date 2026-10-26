# Amendement logiciel 1.2 : mesure d'effet direct en FP64

25 octobre 2026, après le développement 1.1, avant tout forward de test.

Le pilote IOI parallèle a rencontré des valeurs non finies dans une petite projection FP32 malgré des entrées finies. La cause exacte n'est pas établie et le phénomène n'est pas reproduit dans toutes les conditions. Ce constat ne permet pas d'attribuer un défaut général à PyTorch ou à Pythia.

L'audit de Pythia ne trouve aucune valeur non finie : 8 896 512 éléments dans les états/logits sauvegardés et 512 mesures d'effet direct sont finis. Pour réduire le risque avant le test réservé, la **mesure d'effet direct uniquement** utilise désormais une projection de tête contiguë en FP64, un centrage de l'unembedding en FP64 et un produit scalaire FP64. Le modèle, ses projections officielles et ses lecteurs expérimentaux restent FP32. Des assertions de finitude précèdent l'export.

L'archive `archives/v1_1_before_fp64_measurement/` conserve intégralement le développement 1.1 et ses scripts. Les pertes, argmax, facteurs de normalisation, sources, donneurs, positions, têtes, seuils et critères scientifiques sont inchangés. La différence de mesure sur développement est calculée depuis les états conservés et les poids officiels, sans nouveau forward, par `scripts/recompute_de_fp64.py`. Le self-repair, dérivé de cet effet direct, est actualisé avec une trace explicite des changements. Ce recalcul n'est pas une réplication de réseau.

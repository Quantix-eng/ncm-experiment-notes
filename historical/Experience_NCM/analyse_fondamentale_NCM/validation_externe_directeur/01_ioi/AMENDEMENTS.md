# Amendements logiciels du pilote IOI

## A1 — Export non fini après développement, 25 octobre 2026

Le premier runner a exécuté les 128 familles de développement et s'est arrêté à l'export JSON strict : `ValueError: Out of range float values are not JSON compliant: nan`. Temps au dernier compteur : 332,4 s. Aucun fichier d'observations, aucune décision de dommage et aucune sortie test n'ont été publiés. Les assertions natives et les recompositions avaient passé ; cela ne suffit pas à attribuer la cause du NaN.

Les scripts, le protocole, le gel et le journal de cette version sont conservés sous `sources/dev_v1/`. Leurs empreintes y sont recalculables. La version suivante ajoute une vérification de finitude localisée après chaque lecture et effet direct, et conserve les états NPZ avant l'export JSON. Il est interdit de remplacer un NaN par zéro, d'augmenter les marges scientifiques ou de sélectionner un cas favorable. Une sortie non finie doit arrêter la reprise et donner son contexte. Le nouveau gel précède cette reprise. Les règles de dommage, donneurs, têtes, métriques et partitions restent identiques.

Une cause trouvée et toute correction supplémentaire devront être ajoutées ici, sans transformer cet incident logiciel en résultat de compensation.

## A2 — Projection de mesure FP32 non finie, 25 octobre 2026

La reprise A1 localise le problème dès la première source/premier donneur dans `delta_DE`, avant toute ouverture de test. Un diagnostic distinct montre : vecteur64 et matrice64×768 entièrement finis, mais leur produit vectoriel PyTorch FP32 contient181 NaN ; le même produit en FP64 est fini. Le résidu final, les poids d'unembedding, leur moyenne et le facteur LN natif sont finis. Il s'agit d'un défaut numérique observé sur cette opération locale ; aucune cause matérielle générale n'est inférée.

Correction : calculer en FP64 la différence des deux activations puis sa projection pour la **mesure** d'effet direct. Le réseau, ses poids, les interventions effectives (projection officielle pleine), les suites et toutes les conditions restent CPU FP32. Aucune équation de métrique, marge, source, tête ou donneur n'est changée. Les versions A1 et initiale restent conservées. `diagnostic_projection.py` exporte les entrées minimales et compare la projection à NumPy FP64 et à une sommation indépendante. Le gel A2 précède un nouveau passage complet sur développement. Les valeurs non finies sont interdites, jamais remplacées.

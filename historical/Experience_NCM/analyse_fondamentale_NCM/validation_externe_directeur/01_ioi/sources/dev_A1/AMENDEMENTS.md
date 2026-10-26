# Amendements logiciels du pilote IOI

## A1 — Export non fini après développement, 25 octobre 2026

Le premier runner a exécuté les 128 familles de développement et s'est arrêté à l'export JSON strict : `ValueError: Out of range float values are not JSON compliant: nan`. Temps au dernier compteur : 332,4 s. Aucun fichier d'observations, aucune décision de dommage et aucune sortie test n'ont été publiés. Les assertions natives et les recompositions avaient passé ; cela ne suffit pas à attribuer la cause du NaN.

Les scripts, le protocole, le gel et le journal de cette version sont conservés sous `sources/dev_v1/`. Leurs empreintes y sont recalculables. La version suivante ajoute une vérification de finitude localisée après chaque lecture et effet direct, et conserve les états NPZ avant l'export JSON. Il est interdit de remplacer un NaN par zéro, d'augmenter les marges scientifiques ou de sélectionner un cas favorable. Une sortie non finie doit arrêter la reprise et donner son contexte. Le nouveau gel précède cette reprise. Les règles de dommage, donneurs, têtes, métriques et partitions restent identiques.

Une cause trouvée et toute correction supplémentaire devront être ajoutées ici, sans transformer cet incident logiciel en résultat de compensation.

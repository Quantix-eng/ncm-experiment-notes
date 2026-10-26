# Alphabet terminal et lecture certifiée

Commencer par [RAPPORT.md](RAPPORT.md) pour les observations, la méthode et les limites de leur interprétation. [PAPER.pdf](PAPER.pdf) est la version anglaise mise en page. La [note de révision](REVISION_EDITORIALE.md) explique les corrections de portée ; les versions précédentes, le gel et tous les résultats sont conservés.

Le protocole local et les tables étaient fixés avant les nouveaux tests, après un pilote favorable. Ce gel n'était pas une prérégistration indépendante et n'incluait pas les scripts. La nouveauté, un mécanisme général et un gain global ne sont pas établis. Une réplication indépendante avec code scellé, contrôles de codes et codec témoin reste à faire. Les assertions de vérification sont des contrôles logiciels, pas des milliers de réplications scientifiques.

`./run.sh` régénère les documents et vérifie les résultats conservés, sans réécrire les observations historiques. Pour recalculer toutes les expériences dans un nouveau dossier frère : `PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" ../.venv/bin/python scripts/reproduce.py --destination ../reproduction_alphabet`. Ce recalcul des mêmes entrées utilise les sources historiques ; ce n'est pas une réplication indépendante. Les rapports régénérés dans ce dossier frère ont la rédaction antérieure, accompagnée d'une notice ; l'interprétation révisée de référence reste ici. Les anciens dossiers sont lus sans écriture.

Lancer `scripts/verify.py` avec le Python du parent pour un audit sans régénération des documents (deux solveurs, statistiques, rejeu natif et causal sur 16 puzzles, archive et publication). Les commandes et variables de cache sont dans le rapport et `run.sh`.

La politique utilisable est `scripts/policy.py` : lookup, vérification, sinon décodeur natif. Elle n'utilise ni référence ni solveur. L'unicité est une précondition de la garantie de fidélité à la référence, pas une vérification gratuite effectuée par la politique.

Les données nouvelles ont une solution unique et 30 indices. Les réplications easy/medium partagent les premières familles ; les temps ne démontrent pas une accélération globale. Ne pas présenter les substitutions volontaires de chiffres comme une amélioration de résolution.

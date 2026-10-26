# Consolidation éditoriale et analyses auxiliaires des traces conservées

25 octobre 2026. Cette opération est post-hoc : les résultats principaux et les données de l'étape 3a sont déjà connus. Elle ne modifie aucun résultat, seuil, code ou manifeste historique et ne lance aucune nouvelle inférence NCM. Aucun résultat de cette opération ne recevra un statut confirmatoire.

## Mesure descriptive de distinction des entrées

Sur chacune des 16 familles F, définir un choix B uniforme entre les deux entrées uniques A et B déjà générées, et les codes déterministes R_k associés. Calculer exactement, sur les deux atomes de cette distribution construite, I(B;R_k|F=f), puis moyenner les familles. La variable B désigne ici l'indice d'entrée et non un échantillonnage de solutions d'un puzzle fixe.

Pour un code vectoriel complet et deux atomes équiprobables, cette quantité vaut un bit si les vecteurs diffèrent et zéro sinon. Calculer aussi la moyenne des informations par position, le nombre d'identifiants distincts, I(B;R_1:9|F), I(B;R_10:12|F), I(R_1:9;R_10:12|F). Recalculer les valeurs par des entropies de compteurs pour contrôler la formule binaire.

Ces quantités finies décrivent une discrimination dans les paires existantes, pas une estimation populationnelle de l'information commune de Wyner. Les deux entrées ont chacune une solution unique. Conditionnellement à l'entrée complète, la génération argmax est déterministe ; toute information mutuelle conditionnelle entre fonctions de cette entrée est nulle. Les informations calculées en conditionnant seulement sur F ne doivent pas être confondues avec ce dernier objet.

## Sensibilité descriptive du diagnostic d'entrée

Recompter les diagnostics sur les rapports continus u/g existants aux quatre valeurs fixées ici : 0,70, 0,80, 0,90 et 0,95. Conserver les autres conditions d'éligibilité du protocole 3a, dont le ratio dans [0,1], et le cas non monotone. Aucun seuil ne sera choisi d'après le résultat ni substitué au 0,80 historique. Rapporter tous les comptes et rappeler leur caractère post-hoc.

## Contrôles analytiques exacts de coordination

Énumérer les distributions uniformes de parité paire pour n=2..10 : entropie jointe n−1 bits, corrélation totale 1 bit et informations mutuelles deux à deux nulles pour n≥3. Vérifier par factorisation des supports que tout produit exact porté sur la parité fixe toutes les coordonnées. Cet audit fini accompagne une preuve générale ; ce n'est pas une optimisation numérique de l'information commune.

Pour les 16 bases Sudoku à deux solutions, le mélange uniforme des références a une entropie de branche d'un bit ; chaque case différente détermine cette branche. Vérifier qu'au moins deux cases diffèrent et que le produit de marginales non dégénérées permet une combinaison hors des deux solutions. La preuve de C_W=1 bit portera sur cette distribution **cible hypothétique**, pas sur une génération NCM déjà échantillonnée.

## Intégrité et publication

Sceller le script auxiliaire et cette note avec les entrées avant ce calcul. Il s'agit d'un gel local après examen des résultats historiques, pas d'une prérégistration indépendante. Produire PAPER.md en anglais, une note théorique en français, un registre des preuves et une note de consolidation. Rapports et manuscrit en Markdown uniquement ; aucune image, Canvas, présentation ou PDF.

Conserver les études antérieures et leur manuscrit sans modification. Les vérifications éditoriales contrôlent les liens, les chiffres et les empreintes ; elles ne constituent pas une revue indépendante et n'établissent ni priorité ni acceptabilité pour une revue.

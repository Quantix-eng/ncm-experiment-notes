# Journal de consolidation — 25 octobre 2026

La demande porte sur la rédaction du papier et le lien théorique avec Wyner, en conservant la validité scientifique. Le résultat 15/16 est traité comme diagnostic fonctionnel, pas comme preuve d'un mécanisme de compensation du pooling. Livrables exclusivement Markdown, accompagnés des scripts et observations nécessaires à leur audit.

## Ordre des opérations

1. Lecture des protocoles, sources, gels et résultats des étapes précédentes ; examen du chemin de génération, des axes des tenseurs et de la portée du résultat de pooling. Aucun fichier de ces campagnes n'est modifié.
2. Consultation ciblée des sources primaires citées dans le papier. Le texte éditeur original de Wyner n'a pas été directement accessible ; les définitions sont recoupées dans les extensions primaires multivariée et conditionnelle. Cette recherche n'est pas une revue systématique.
3. Rédaction d'un plan auxiliaire explicitement post-hoc sur traces déjà connues : distinction des entrées dans les codes, sensibilité à quatre seuils tous rapportés, énumérations illustrant les preuves. Une première commande de gel échoue parce que le protocole avait été créé dans un dossier racine incorrect ; déplacement du document au bon emplacement avant tout gel et tout résultat auxiliaire. Aucune observation n'est remplacée ni aucun critère modifié.
4. Gel local du plan, du script et de leurs entrées dans `results/auxiliary_frozen.json`, puis exécution unique d'`auxiliary.py`. Le gel ne constitue pas une prérégistration indépendante ; le résultat principal NCM était déjà connu.
5. Rédaction de `PAPER.md`, `THEORIE_WYNER.md` et du registre des preuves. Aucune nouvelle inférence de réseau, dose, point de coupure ou partition n'est exploré.
6. Relecture théorique par un second agent : preuves de parité et de deux solutions vérifiées ; précision apportée sur le coût **conditionnel au-delà de l'entrée complète**, et formulation d'un sampler stochastique explicitement futur. Ajout des précisions sur un message de longueur fixe et sur la possible non-injectivité codes/reconstruction. Cette intervention est une relecture interne, sans validation externe ni nouvelle réplication empirique.
7. Audit de consolidation par `verify.py` : empreintes historiques et auxiliaires, recomptages par comparaison directe des vecteurs, sensibilité, entropies et comptes centraux du papier, liens Markdown. Le résultat effectif figure dans `results/verification.json` ; le manifeste final scelle ensuite la livraison.

## Conservation et statut

Le manuscrit historique `manuscrit_unifie/MANUSCRIPT.md`, les protocoles gelés, les seuils NCM et les sorties originales sont conservés. Le nouvel article est une synthèse autonome, avec un apport auxiliaire identifié comme tel. La nouveauté vis-à-vis des travaux antérieurs et la valeur de benchmark de référence restent à valider. Aucun dépôt public ni soumission n'est effectué.

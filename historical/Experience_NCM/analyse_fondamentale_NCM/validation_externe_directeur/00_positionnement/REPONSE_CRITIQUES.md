# Révision après critiques : choix numériques, portée, figures et provenance

25 octobre 2026. Révision éditoriale avec figures et comparaisons arithmétiques post-hoc sur les intervalles conservés. Aucune nouvelle inférence de réseau, aucun apprentissage, aucun nouveau bootstrap, aucun changement des protocoles ou décisions historiques.

## 2. Seuils : critique retenue, sans justification inventée

Le gel local réduit une sélection après résultats ; il ne rend pas un seuil scientifiquement nécessaire. La section 4 distingue maintenant justification opérationnelle, traduction d'échelle et validation d'utilité, qui manque.

- **0,01 nat** : convention héritée, sans utilité minimale indépendante ni analyse de puissance préalable. L'amélioration correspond à environ 1,01 % de hausse du rapport géométrique des probabilités de référence ; cette traduction ne justifie pas son utilité.
- **0,05 nat et porte à 25 %** : filtres pratiques d'opportunité de réparation, sans optimalité démontrée.
- **Dose 3** : plus grande valeur d'une grille développement ayant échoué. Son utilisation ne rend pas la corruption informative. Cette expérience est en annexe.
- **Donneurs 37 et 73** : donnent deux donneurs reproductibles, distincts et non identiques au receveur. Aucune propriété causale des nombres ni robustesse à d'autres banques n'est revendiquée.
- **InterpBench 1,00** : unanimité exacte des sondes incluses, pas probabilité de confiance calibrée. Les trois seuils restent visibles.

La figure 3 compare les mêmes intervalles à six marges, explicitement après résultats. L10H0 a un avantage estimé positif, 0,0082 nat, intervalle descriptif 98,75 % [0,0060 ; 0,0109]. Son échec à dépasser 0,01 n'est pas une absence d'effet. À 0,005, la borne inférieure dépasserait la marge ; cette comparaison ne remplace pas le test historique. Le nouvel axe scientifique porte sur les effets continus, pas sur des classes mécanistes.

## 3. Portée : IOI devient l'argument central

Nouveau titre : « Recovery and Perturbation Matching in GPT-2: A Controlled IOI Case Study ». Le texte principal porte sur un checkpoint, une tâche, trois interventions et la séparation récupération/appariement/blocage.

Pythia et InterpBench restent deux contrôles de portée courts. NCM, copie, petits Transformers, calibration analytique et Wyner sont déplacés en annexes, avec les résultats négatifs et les liens vers les méthodes. Les quinze checkpoints ne sont plus présentés comme quinze réplications de l'argument principal.

## 4. Lisibilité : quatre figures fondées sur les artefacts

1. Schéma des cinq combinaisons état/suite, décrites en mots.
2. Dommage, récupération et blocage des backups/témoins, avec les intervalles conservés.
3. Avantage d'appariement continu et dépendance des décisions à la marge.
4. Risque/couverture InterpBench à tous les seuils annoncés, avec agrandissement de la région de faible risque.

Le texte principal utilise des noms explicites et trois ou quatre décimales. L'algèbre et les chiffres historiques détaillés restent en annexe. Les figures ne sont pas générées comme des illustrations inventées : leurs points proviennent des résumés JSON épinglés dans SENSIBILITE_MARGES.json.

## 5. Provenance : transparence conservée et précisée

La section 7.2 nomme OpenAI Codex et décrit l'usage substantiel dans la planification, le code, les analyses, les audits, la synthèse bibliographique et la rédaction. Les revues par agents restent des contrôles internes. Aucun audit humain complet, aucune réplication humaine indépendante et aucun accord humain final non documenté ne sont ajoutés.

Vérification de la politique TMLR le 25 octobre 2026 : la [politique officielle](https://jmlr.org/tmlr/editorial-policies.html#llm-use-policy-for-authors) autorise l'assistance par LLM, exclut leur qualité d'auteur et place la responsabilité sur les auteurs. Cette page ne formule pas l'obligation générale de divulgation décrite dans la critique. Nous conservons néanmoins une divulgation détaillée. Les politiques d'autres conférences n'ont pas été généralisées à TMLR ni auditées pour cette révision.

La réécriture ne peut pas fabriquer la vérification humaine manquante. Elle distingue ce qui est attesté par les artefacts, les limites des audits logiciels et la responsabilité humaine qui reste à exercer avant une soumission.

## Livraison et intégrité

Le manuscrit antérieur et ses sources de mise en page sont dans archives/avant_revision_critiques_20261025/. Les empreintes des artefacts scientifiques présents sont contrôlées dans VERIFICATION_CRITIQUES.json. Les anciens manifestes et vérificateurs restent des documents de leurs versions historiques ; ils ne décrivent pas la présente restructuration.

Le nouveau PDF utilise le style officiel TMLR en mode prépublication, avec auteurs anonymes provisoires. Ni soumission ni acceptation ne sont revendiquées.

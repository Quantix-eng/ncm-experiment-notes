# Antériorité des idées de recherche sur NCM

Recherche ciblée du **22 octobre 2026**. **La plupart des méthodes proposées ont déjà des précédents scientifiques. L'occasion de contribution se situe dans une question propre au raffinement multiscalaire, ses contrôles et ses résultats, plutôt que dans la simple application d'une sonde ou de LoRA.**

Périmètre : propositions de l'utilisateur et de l'assistant consignées dans `RAPPORT_METACOGNITION.md`. Sources primaires privilégiées : articles, actes de conférences et prépublications. Notices/résumés vérifiés ; sections méthodologiques consultées pour les précédents les plus proches : ATPO, SchED, IFT, mécanismes d'introspection, contrôle causal de la confiance et mesure psychophysique. Les résultats ci-dessous sont ceux rapportés par les auteurs, sans reproduction locale. Les prépublications ne sont pas assimilées à une validation indépendante.

« Déjà fait » signifie précédent identifiable pour le principe, **pas expérience identique sur NCM**. « Non trouvé sur NCM » reste une conclusion de recherche limitée. Le journal des requêtes et limites figure dans `audit/metacognition-literature-search.md`.

## 1. Entraîner un petit modèle observateur sur les états internes

**Précédent direct.** [Azaria et Mitchell, Findings EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.68/) entraînent un classificateur sur les activations cachées pour prédire la véracité d'énoncés lus ou générés. [Kossen et al., 2024, Semantic Entropy Probes](https://arxiv.org/abs/2406.15927) apprennent à approximer l'entropie sémantique depuis les états d'une seule génération.

Notre sonde de réussite Sudoku serait une adaptation de cette famille. La propriété sondée diffère : véracité d'énoncé, incertitude sémantique et réussite d'une grille ne sont pas interchangeables. Une sonde performante ne démontre ni un mécanisme natif de surveillance ni une introspection. **À conserver comme instrument et référence ; nouveauté méthodologique faible en soi.**

**Vérification spécifique du 23 octobre 2026 :** aucun précédent de classificateur auxiliaire entraîné à prédire réussite/erreur depuis les états de ce NCM identifié par les requêtes `"Next-Concept Modeling" "probe"`, `"Next-Concept Modeling" "probing"` et `"HazyResearch" "NCM" "linear probe"`. L'article original contient bien une annexe H, *Probing the Hierarchy*, mais ses expériences sont qualitatives : fixer des niveaux grossiers, mélanger des représentations et remplacer des codes, puis observer les textes. Cette annexe ne décrit pas l'entraînement d'une sonde prédictive de réussite. Distinguer donc **probing qualitatif déjà effectué sur NCM** et **sonde auxiliaire entraînée pour notre cible, non identifiée ici**. [Annexe H](https://arxiv.org/html/2610.14162v1).

## 2. Utiliser la trajectoire des niveaux pour anticiper succès, échec et correction

**Précédent très proche dans une autre architecture.** [Chen et al., 2025, Dynamic Confusion Zones / ATPO](https://arxiv.org/html/2511.15208v1) étudient entropie, marge et variation d'entropie pendant le débruitage de modèles de diffusion. Ils rapportent des associations avec la réussite finale et adaptent l'allocation des gradients aux étapes sélectionnées.

Cela recouvre une part importante de notre proposition de suivi temporel. Une étape de diffusion n'est cependant pas un niveau VQ de longueur croissante. La question plus précise **« à conflits visibles comparables, quelles erreurs seront corrigées par les niveaux restants ? »** reste une adaptation candidate, non identifiée ici sur NCM. Il faudra montrer un gain sur probabilités et contraintes seules et ne pas utiliser d'étapes futures dans les prédicteurs précoces. **Intérêt réel, originalité à démontrer.**

## 3. Ajouter du bruit aux codes, aux activations ou à l'apprentissage

**Déjà partiellement fait dans NCM lui-même.** L'[article NCM, §3.4 et annexe F.3](https://arxiv.org/html/2610.14162v1) remplace des codes de conditionnement pendant l'entraînement, en conservant les bonnes cibles. Il examine l'effet du taux de corruption sur la qualité. Ce n'est ni du bruit d'étiquette ni une étude de métacognition.

Pour l'apprentissage bruité et la calibration, [Calibrating Language Models for Neural Ranking under Noisy Supervision, UncertaiNLP 2025](https://aclanthology.org/2025.uncertainlp-main.22/) compare plusieurs niveaux de bruit et analyse la calibration. **Créer une variante bruitée n'est pas nouveau.** Tester si un signal interne fixé détecte la perte de fiabilité ou prédit la correction après corruption d'un niveau NCM reste une question différente. Le bruit peut aussi régulariser et améliorer un modèle : ne pas présupposer une dégradation.

## 4. Créer un modèle « arrogant » ou surconfiant

**Précédents directs pour calibration et fine-tuning.** [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) étudient la calibration et la mise à l'échelle des logits par température. [Wang et al., ACL 2025, CogCalib](https://aclanthology.org/2025.acl-long.722/) examinent comment le fine-tuning sur des connaissances déjà acquises peut induire une surconfiance, et proposent une correction.

Une transformation positive globale des logits conserve l'argmax et change les probabilités : bon témoin pour dissocier compétence et calibration, sans créer une personnalité. Un fine-tuning peut modifier les deux ; les mesurer séparément. **Ni « arrogance », ni LoRA, ni changement de température ne constituent une nouveauté scientifique à eux seuls.** Une mesure prétendue de métacognition ne doit pas être interprétée comme une capacité nouvelle simplement parce que la calibration change.

## 5. Comparer des modèles sous-entraînés et plusieurs tailles

**Le dispositif général existe.** [Biderman et al., 2023, Pythia](https://arxiv.org/abs/2304.01373) publient une série contrôlée de modèles et de checkpoints pour étudier l'entraînement et la taille. Ce papier fournit un précédent de protocole, pas une démonstration spécifique de métacognition sous-entraînée. [Kadavath et al., 2022](https://arxiv.org/abs/2207.05221) étudient déjà l'auto-évaluation et son évolution avec la taille.

Le NCM original compare aussi des tailles de générateurs LM1B (§4.5), sans que cela établisse leur métacognition. **Pas de précédent exact identifié pour notre comparaison métacognitive de checkpoints NCM précoces**, lesquels ne sont pas disponibles dans notre inventaire. Une adaptation brève d'un modèle mature n'est pas du sous-entraînement. La taille et la durée d'entraînement doivent constituer deux facteurs distincts.

## 6. Fine-tuner pour rendre une introspection mesurable

**Précédent direct, y compris avec LoRA.** [Hahami, Sinha et Jain, 2026, Introspection Fine-Tuning](https://arxiv.org/html/2607.14111v1) entraînent des modèles à localiser des perturbations internes. L'annexe D décrit LoRA. Ils testent aussi le transfert vers une comparaison de force des injections. Ce résultat concerne une compétence entraînée de détection/localisation, pas une preuve générale de connaissance de soi.

Ce travail impose un contrôle utile : une injection peut augmenter les réponses affirmatives même à des questions sans rapport avec l'introspection. **Détection apparente et détection spécifique doivent être séparées.** Leur tâche verbale ne se transpose pas directement à notre checkpoint Sudoku. Ajouter une tête d'évaluation ou entraîner une réponse métacognitive produirait un système adapté, à distinguer du NCM original. La nouveauté éventuelle serait l'objet multiscalaire et la validité du test, pas LoRA.

## 7. Perturber un état interne pour chercher une preuve causale

**Précédents directs pour la stratégie d'intervention.** [Lindsey, 2025/2026, Emergent Introspective Awareness](https://arxiv.org/abs/2601.01828) injecte des représentations et examine les comptes rendus des modèles. [Macar et al., 2026, Mechanisms of Introspective Awareness](https://arxiv.org/html/2603.21396) étudient les mécanismes et les interventions sur la détection.

Ces travaux portent sur des modèles conversationnels capables de produire un compte rendu, pas sur la hiérarchie NCM. Chez nous, modifier un code et observer un échec démontre la contribution du code à la résolution, **pas la surveillance de cet échec**. Une attribution métacognitive plus forte exigerait de distinguer un signal d'évaluation du contenu qui résout la tâche, puis d'étudier son effet sur une décision de contrôle. Témoins de direction, amplitude et difficulté nécessaires.

## 8. Vérifier que le signal est propre au modèle, et non à la difficulté

**Précédent particulièrement pertinent pour nos variantes.** [Binder et al., Looking Inward, ICLR 2025](https://openreview.net/pdf?id=eb5pkwIB5i) comparent l'auto-prédiction d'un modèle à la prédiction de son comportement par un autre modèle, entraîné sur ces comportements. Ils étudient aussi des changements de comportement.

Notre comparaison original/variantes peut reprendre cette logique de spécificité : prédire les échecs d'un modèle particulier, pas seulement ceux des puzzles difficiles pour tous. Une sonde externe, même spécifique, reste une mesure d'observateur. **Contrôle à ajouter, sans le présenter comme une idée nouvelle.**

## 9. Échantillonner plusieurs générations et mesurer leur accord

**Déjà établi.** [Wang et al., prépublication 2022 / ICLR 2023, Self-Consistency](https://arxiv.org/abs/2203.11171) agrègent plusieurs raisonnements ; [Manakul et al., EMNLP 2023, SelfCheckGPT](https://aclanthology.org/2023.emnlp-main.557/) exploitent la cohérence entre générations pour détecter des hallucinations. Le NCM original utilise déjà plusieurs candidats et une sélection MBR (§4.5).

Ces usages ne sont pas une seule et même méthode, mais rendent faible la nouveauté de « plusieurs réponses + accord ». Pour Sudoku déterministe, les répétitions identiques ne sont pas informatives. Un échantillonnage de codes changerait la politique évaluée et devrait être documenté. **Référence possible, pas axe de nouveauté principal.** « Self random » reste non défini dans notre échange ; aucune identité avec ces méthodes n'est présumée.

## 10. Comparer les réponses après symétries ou permutations des chiffres

**Précédent général direct, adaptation Sudoku non établie ici.** [Bahat et Shakhnarovich, 2020](https://arxiv.org/abs/2006.16705) estiment la confiance d'un classificateur à partir de transformations d'entrée préservant la sémantique. Notre idée suit la même logique avec les symétries du Sudoku et le retour au repère initial.

Cela constitue une mesure de robustesse/équivariance candidate ; elle ne prouve pas une représentation interne de l'incertitude. Un modèle peut se tromper de manière stable. **Bon contrôle, originalité générale faible ; précédent exact de cette évaluation sur NCM non trouvé.** Ne pas prétendre que l'absence trouvée sur NCM implique l'absence de littérature sur les symétries Sudoku en général.

## 11. Utiliser la confiance pour s'arrêter, s'abstenir ou explorer davantage

**Précédents directs et proches.** [Schuster et al., 2022, CALM](https://arxiv.org/abs/2207.07061) allouent dynamiquement le calcul ; [Mohamed et al., 2025, SchED](https://arxiv.org/html/2512.02892v1) proposent un arrêt anticipé de modèles de diffusion fondé sur les marges et la progression. [Kamath et al., ACL 2020](https://aclanthology.org/2020.acl-main.503/) étudient l'abstention sous changement de domaine.

Le précédent causal le plus important est [Kumaran et al., Nature Machine Intelligence, 7 septembre 2026](https://www.nature.com/articles/s42256-026-01293-x), également disponible en [texte intégral arXiv v3](https://arxiv.org/html/2603.22161v3). Les auteurs interviennent sur des signaux internes de confiance et observent des changements d'abstention. Ils traitent donc déjà la distinction corrélation/utilisation comportementale que nous recherchons.

**Construire un contrôleur externe n'est pas nouveau et ne démontrerait pas un contrôle natif NCM.** L'apport possible concerne un contrôle adapté aux niveaux résiduels, comparé à des règles simples et à budget égal, en incluant le coût des décodages intermédiaires. CALM sort de couches Transformer, SchED du débruitage : leurs axes de calcul ne sont pas nos niveaux de codes.

## 12. Proposer une évaluation de métacognition adaptée au NCM

**La question de mesure est déjà travaillée.** [Servajean et Servajean, 2026, v4](https://arxiv.org/html/2603.29693v4) distinguent sensibilité et calibration et proposent un cadre de mesure de la régulation des décisions. [Trinh et al., atelier CogInterp NeurIPS 2025](https://arxiv.org/abs/2512.10451) utilisent une sensibilité métacognitive pour guider la sélection de modèles.

Il reste possible de proposer une tâche nouvelle, **pas de déclarer nouvelle une métrique standard renommée**. Notre candidate : mesurer, à chaque niveau, l'anticipation de la correction future, le transfert après perturbation et le bénéfice de contrôle. Il faudra établir validité, fiabilité et valeur ajoutée sur les références existantes. La comparaison de sensibilité doit tenir compte des différences de réussite de tâche ; AUROC, calibration et meta-d′ ne sont pas interchangeables. Le cadre meta-d′ demanderait une adaptation justifiée à une réponse structurée.

## Décision de recherche

**À reformuler dans notre discours :** la sonde, le bruit, la surconfiance, LoRA, l'étude de taille et l'arrêt anticipé sont des outils ou questions connus. Notre proposition antérieure d'anticipation par trajectoire a notamment un précédent plus proche que ne le laissait entendre le premier cadrage : ATPO dans la diffusion.

**Question à privilégier :** à niveau fixé et à qualité apparente comparable, l'état multiscalaire prédit-il si les niveaux restants vont corriger une erreur, puis cette information permet-elle un contrôle utile ? Tester d'abord l'information décodable ; réserver toute conclusion sur la métacognition native à une preuve d'utilisation interne.

**Contribution plausible, non garantie :** étude contrôlée de la récupérabilité des erreurs dans une hiérarchie résiduelle à codes discrets. Références minimales : sonde de type Azaria, probabilités seules, dynamique de type ATPO, contraintes visibles, spécificité au modèle inspirée de Binder. Toute extension au contrôle devrait aussi se positionner face à SchED et Kumaran.

**Anomalies et limites bibliographiques :**

- Une partie de notre idée « bruit » était déjà présente dans NCM. Ses auteurs étudient la robustesse de génération, pas une surveillance métacognitive.
- Les preuves par détection d'injection peuvent être confondues par un changement du style de réponse ; IFT en donne un exemple et propose des contrôles.
- Le titre de la notice de recherche de Servajean différait du titre courant de la v4 ; le lien versionné et le titre courant sont utilisés ici.
- L'accès direct à Nature a échoué ; les métadonnées de l'éditeur et la version auteur arXiv ont permis la vérification. Certains résultats de recherche affichent des dates incohérentes avec leurs identifiants : ne pas dater une priorité à partir du seul index. IFT est cité à l'année 2026, sans conclusion sur son mois de première diffusion.
- Aucun travail primaire dédié à la métacognition de **ce NCM multiscalaire** n'a été identifié par cette recherche. Ce constat ne constitue ni une preuve d'absence, ni une revue systématique exhaustive, ni une garantie de nouveauté publiable. Les architectures LCM et NCP ne sont pas automatiquement assimilées au NCM de HazyResearch.

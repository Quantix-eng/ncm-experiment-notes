# Pilote InterpBench sur développement — version 0.1

25 octobre 2026. Ce protocole est gelé localement avant toute inférence. Il prépare une extension du benchmark ; il ne constitue pas l'évaluation externe réservée recommandée.

## Périmètre

Les tâches 8 et 33, déterminées par le directeur, sont les seules exécutables. Pour chacune, générer 128 séquences sources uniques par un PRNG fixé, avec 3 à 9 tokens hors BOS. Ajouter BOS et PAD jusqu'à une longueur de dix. L'identité 8 utilise des entiers 0..10. La tâche 33 prédit la parité de longueur de chaque mot ; elle ne prédit pas celle de la séquence. Réponses et encodeurs proviennent du code officiel, jamais des logits observés. Toutes les positions utiles de la séquence constituent une famille ; BOS et PAD sont exclus des pertes.

Les cas 13, 18, 19 et 21 sont réservés : leurs sources, configurations, graphes et poids peuvent être inventoriés, mais aucun forward ne doit être exécuté ici. La préparation des scripts et métadonnées ne constitue pas l'évaluation de leurs sorties.

## Contrat précis et oracle

L'unité d'échange est la sortie projetée d'une tête entière, à toutes les positions, avec donneur différent. Deux sites sont annoncés : L0H0 et L1H0. Les huit têtes des deux tâches de développement sont hors des circuits publiés ; le programme externe prédit donc une invariance de sa sortie sous ces échanges. L'échange n'intervient jamais sur Q/K/V séparément.

Pour la source i, les donneurs sont i+1 et i+2 modulo 128. Chacun conserve son padding propre : même longueur tensorielle, mais pas forcément même longueur de contenu. Ce facteur est documenté et ne sera pas éliminé après lecture des effets. La frontière est située après l'ajout de l'attention du bloc ciblé, avant son MLP ; les autres têtes du bloc et son biais restent natifs. La suite contient le MLP du bloc, puis attention et MLP de chaque bloc suivant, ajoutés dans l'ordre.

Conserver Y00, Y10, Y11, Y01 et Y1wrong. Le dernier rejoue une suite adaptée au deuxième donneur au même site. Vérifier la parité exacte du suffixe natif et du sham. L'implémentation autonome reproduit les équations TransformerLens 1.19.0 correspondant aux checkpoints ; un second calcul NumPy FP64 contrôle les logits natifs, avec tolérance absolue 2e-4 et relative 2e-5, et les argmax. Ce contrôle ne remplace pas encore une parité contre le chargeur officiel TransformerLens/Tracr.

## Mesures et décision

NLL moyenne sur tokens utiles, séquence entièrement exacte, accord de l'échange régénéré avec le programme externe, D=L10-L00, R=L10-L11, I=L10+L01-L11-L00, S=L1wrong-L11. Rapporter séparément par tâche et site. Pas d'IC populationnel, pas de test de significativité, pas de seuil appris, pas d'attribution à six classes. Un dommage absent ne valide aucune réparation. Les comptes décrivent les sources de développement et ne sont pas une preuve de généralisation.

Arrêter en cas de divergence d'encodage, d'échec de parité, d'anomalie du sham ou de modification des fichiers gelés. Un désaccord avec le programme doit être conservé et interprété d'abord comme une divergence réseau/programme dans le domaine testé. Toute correction préserve les fichiers précédents et reçoit un amendement. Aucune recherche d'un site favorable n'est autorisée.

## Suite requise

Avant les quatre tâches réservées, obtenir la correspondance complète Tracr–réseau, vérifier la parité officielle sur développement, définir l'extension produisant des prédictions d'interventions sans consulter l'oracle, et geler ses règles. Comparer ensuite aux métriques IIA et SIIA à granularité identique. Le pilote ci-dessus ne revendique aucune amélioration face à IIA.

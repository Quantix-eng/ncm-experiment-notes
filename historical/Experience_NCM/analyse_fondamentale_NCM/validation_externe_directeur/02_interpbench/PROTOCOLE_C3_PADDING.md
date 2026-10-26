# C3 — contrôle post-hoc du prétraitement natif

25 octobre 2026. Écrit après C2, avant les calculs ci-dessous. Ce contrôle ne remplace pas les résultats réservés C2 et n'ajuste aucun prédicteur.

C2 utilise des séquences non paddées ; le générateur officiel `gen_random_input_output` ajoute BOS/PAD jusqu'à la longueur maximale, tout en calculant sa référence RASP sur les seuls tokens utiles. Le faible accord natif du cas18 impose de distinguer un décalage de prétraitement et un désaccord réseau/programme avant toute interprétation des erreurs de diagnostic.

Sur les mêmes1024sources, sans sélection, conserver les références et masques RASP natifs C2 ; réutiliser les logits non paddés sauvegardés. Calculer une seule lecture native du réseau avec PAD à `n_ctx` et une lecture du compilateur Craft avec ces mêmes PAD. Comparer, sur les positions utiles définies : réseau sans PAD/programme ; réseau avec PAD/programme ; Craft avec PAD/programme ; réseau avec PAD/Craft avec PAD. Le programme n'est pas exécuté sur une entrée hors vocabulaire contenant PAD : son masque reste celui de l'entrée utile. Les sorties Craft intermédiaires et leurs éventuelles divergences sont conservées.

Les quatre tâches et tous les cas défavorables restent inclus. Aucun changement de poids, aucune corruption future, aucune nouvelle recherche de programme, aucun seuil de succès ajouté. Les comptes par longueur seront descriptifs. Le contrôle est explicitement post-hoc ; une amélioration avec PAD ne deviendra pas rétroactivement la performance C2. Une divergence persistante interdit de présenter ce checkpoint comme oracle exact de toutes les conditions testées.

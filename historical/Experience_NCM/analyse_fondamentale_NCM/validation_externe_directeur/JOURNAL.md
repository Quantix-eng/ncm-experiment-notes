# Journal de l'application de l'avis du directeur

25 octobre 2026, heures UTC dans les fichiers de gel.

- Lecture intégrale de l'avis ; source conservée sans altération dans `sources/AVIS_DIRECTEUR.md`. Anciennes campagnes et papier final V3 maintenus en lecture seule.
- Division interne : antécédents/portée/réplication humaine, InterpBench, Pythia ; intégration IOI et synthèse par l'agent principal. Une équipe d'agents ne constitue pas une validation humaine extérieure.
- Vérification des sources directes : contributions aval figées et facteur LN final figé existaient déjà dans l'article self-repair. Les têtes backups simultanées à L9H9 sont exclues de son ensemble aval. La publication IOI étudiée ne garantit pas une compensation pour notre intervention unitaire.
- Téléchargement MIB–IOI épinglé à une révision publique ; métadonnées et tokenisation seules consultées avant sélection. 128 familles dev et 512 test fixées, sans exclusion sur les sorties. Le gel de développement conserve les empreintes des scripts, du protocole, des données et des poids.
- Les assertions échouées sont conservées dans les branches correspondantes. Une correction motivée doit laisser un amendement et un nouveau gel ; elle ne doit pas ouvrir les sorties réservées pendant le diagnostic.

Les faits de calcul et décisions de passage sont consignés dans les rapports de branche et le rapport global à la clôture.

## Clôture demandée par l'utilisateur

- IOI L9H9 : test512 familles terminé ;432,42s pour le runner valide, puis audit primaire passé. Les réplications L9H6/L10H0 et l'audit supplémentaire préparé n'ont pas été lancés.
- Pythia : test512 séquences effectivement terminé ; audit supplémentaire du lecteur et contrôles scalaires passés. Les résultats moyens faibles/négatifs sont conservés.
- InterpBench : audit officiel et oracle Craft sur développement terminés ; correspondances statiques non résolues pour trois tâches réservées, dont les sorties restent fermées.
- À la demande de limiter les dépenses et de livrer un bilan court, les deux sous-agents encore actifs ont été interrompus. Aucune nouvelle campagne n'est lancée. Le [rapport global](RAPPORT.md) constitue le point de reprise.

## Reprise autorisée : secondaires IOI et InterpBench

- Demande ultérieure explicite : terminer InterpBench et commencer L9H6/L10H0. Travail effectué par l'agent principal seul, sans nouveau sous-agent.
- Les deux runners IOI secondaires gelés sont exécutés jusqu'au bout et audités. L10H0 ne franchit pas la marge de spécificité ; ce résultat reste négatif. Le primaire et ses observations restent inchangés.
- Avant sorties réservées InterpBench : récupération du changement historique du programme13, comparaison des identifiants HF des poids/arêtes. Les snapshots et le patch sont conservés. Les auxiliaires sans masque ne deviennent pas des contrôles inertes.
- C2 : domaine, donneurs de même longueur et règles fixés avant les forwards réservés. Les prédictions sont gelées avant les futures interventions. Quatre tâches,1024sources,5632requêtes ; aucune discordance RASP/Craft sur les masques non vides. Scripts de score aussi scellés avant sorties neuronales.
- Cas18 : défaut de conformité conservé ; C3 est un contrôle de padding explicitement post-hoc, gelé avant calcul, sans remplacement de C2. L'audit supplémentaire des masques ne change ni les prédictions ni les scores primaires.
- Le bilan du premier arrêt est archivé dans `sources/RAPPORT_avant_reprise_secondaires_C2.md`. La valeur ajoutée prédictive générale reste non établie. Aucun apprentissage ni nouvelle inférence NCM n'est lancé.

## Intégration éditoriale des résultats au papier

- Demande explicite d'intégrer tous les nouveaux résultats, avec une rédaction naturelle et scientifiquement crédible. Aucun nouveau sous-agent, réseau ou calcul d'inférence statistique n'est utilisé.
- L'ancienne copie de `00_positionnement/PAPER_REVISE.md` et le bilan général précédent sont archivés dans `00_positionnement/archives/avant_integration_resultats_20261025/`.
- Ajout des méthodes et résultats IOI primaire/secondaires, Pythia et InterpBench C2/C3 ; remplacement du programme prospectif obsolète. Résumé, introduction, discussion, limites et conclusion sont réécrits autour des expériences effectivement exécutées.
- Les échecs L10H0/cas18, les limites de couverture et l'absence de supériorité prédictive restent visibles. Le texte ne revendique pas une validation humaine ou un statut de référence.
- Vérification éditoriale des nombres, des liens et des empreintes scientifiques dans `00_positionnement/VERIFICATION_INTEGRATION.json`. Le PDF V3 reste une version historique, sans ces nouvelles expériences ; le livrable actualisé est le manuscrit Markdown.

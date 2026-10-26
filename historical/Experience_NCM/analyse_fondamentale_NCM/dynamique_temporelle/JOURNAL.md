# Journal concis

- Piste pooling mise de côté sans modification. Protocole temporel rédigé avant calculs de cette extension ; traces antérieures réutilisées comme telles, pas présentées comme de nouvelles observations indépendantes.
- L'hypothèse d'irréversibilité est testée par des contre-exemples natifs avant toute interprétation causale. Les interventions distinguent arrêt, omission, lecture fractionnaire et continuation régénérée.
- Observations terminées : 3 088/12 333 corrections test à 9 sont transitoires ; contre-exemples natifs conservés. Ajout d'un contrôle descriptif des 81 cases et des séparateurs pour exclure une explication purement syntaxique des échecs précoces. Aucun seuil ni sous-échantillon modifié.
- Incident sans effet sur les résultats : un import `math.log` masquait le logger final dans le premier calcul observationnel, après sauvegarde du JSON. Import inutilisé retiré, calcul complet rejoué avec les contrôles de cases ajoutés.
- Interventions terminées : parité du runner, 128 trajectoires natives, omissions 8..12 et continuations fractionnaires ; bruit avant/après sur 64 puzzles × trois graines × deux doses. Omission 9 : 103/128 contre 105/128 ; omission 12 : 0/128. L'hypothèse de verrouillage n'est pas retenue.
- Publication générée à partir des JSON et traces NPZ ; contrôles arithmétiques, rejeu local et audit des quatre campagnes antérieures dans scripts/verify.py. La vérification ne rejoue qu'un lot de 16, avec portée annoncée.

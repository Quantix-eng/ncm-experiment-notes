# Contrôle post-hoc : contribution du niveau et compensation fine

Motif : les reconstructions hard restent exactes alors que les codes du niveau 5 changent. Une reconstruction finale préservée peut venir d'un effet limité de ces codes au décodeur ou d'une compensation par les niveaux fins réencodés.

Sur les 128 premières solutions test hard déjà fixées, sans nouvelle sélection de variante :

- Décoder les codes natifs, les codes entièrement réencodés avec le pooling sélectionné, puis un hybride qui ne remplace que les neuf codes du niveau 5, en gardant tous les autres codes natifs.
- Rapporter exactitude finale, cases changées et erreurs de reconstruction. Ne pas assimiler l'hybride hors distribution à une variante déployable.
- Décoder aussi les accumulations après niveaux 4 et 5 des solutions encodées. Comparer les cases exactes et leurs différences appariées par solution ; il ne s'agit pas de résolution à partir des indices.

Ce complément est décidé après observation des reconstructions propres, ne participe pas à la sélection, et reste explicitement post-hoc.

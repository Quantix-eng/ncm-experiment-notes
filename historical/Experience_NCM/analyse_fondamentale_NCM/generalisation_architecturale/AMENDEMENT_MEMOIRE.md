# Amendement d'exécution : LM1B et mémoire

Le premier audit LM1B a été interrompu avant sauvegarde de résultats par une erreur système ENOSPC lors de l'écriture du log, après pression mémoire et croissance du swap macOS. Les résultats TinyStories avaient déjà été sauvegardés. Aucun résultat partiel LM1B ne sera utilisé.

Reprise LM1B après les générations Sudoku : mêmes 256 exemples aléatoires, 128 longs, indices, graine, métriques et poids. Initialiser la structure sur le périphérique meta puis charger les tenseurs CPU mappés avec load_state_dict(assign=True), afin d'éviter une copie mémoire des poids. Exécuter les lots de LM1B un exemple à la fois ; le protocole initial indiquait quatre. TinyStories reste inchangé. Vérifier le chargement strict, l'égalité de l'accumulation explicite et de encode, les poids avant/après, et les codes sur huit exemples entre lots 1 et 4.

Cette modification porte sur les ressources d'exécution, sans changement de la sélection des données ni des critères d'analyse. Les logs de la tentative interrompue restent conservés.

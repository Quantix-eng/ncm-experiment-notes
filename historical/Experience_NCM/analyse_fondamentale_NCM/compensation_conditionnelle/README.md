# Plan croisé et génération LM1B

Lire RAPPORT.md pour les conclusions, PAPER.md/PAPER.pdf pour l'article anglais et OBJECTIFS_ET_PROTOCOLE.md pour le protocole gelé. Les six campagnes précédentes sont conservées.

`./run.sh` reproduit les expériences et les documents. Les poids LM1B small sont téléchargés sous cette extension, les poids/tokenizer/corpus antérieurs sont lus dans le parent. Les gros poids ne sont pas inclus dans l'archive de sources. Python 3.13, PyTorch 2.8.0, NumPy 1.26.4 ; verrou dans `../sources/requirements-lock.txt`. La publication requiert Pandoc et pdfLaTeX aux chemins déclarés dans publish.py ; PyMuPDF est lu dans `../generalisation_architecturale/.deps` pour la vérification.

La génération LM1B est native eager, à une trajectoire par graine. Ses checkpoints individuels sont repris si présents. Les paramètres chargés par mmap sont identiques ; les buffers RoPE non persistants sont reconstruits exactement selon le constructeur pour éviter une allocation temporaire inutile. Les deux premières graines déjà calculées sont rejouées après cette optimisation. Le codec oracle utilise des lots de quatre, avec parité explicite contre des lots unitaires sur les quatre premières phrases.

La vérification recalcule les métriques et rejoue 16 nouveaux puzzles sur les quatre conditions, puis la première génération LM1B et ses douze lectures, le post-traitement final officiel et une phrase oracle. Elle ne rejoue pas toutes les trajectoires ; run.sh reconstruit la campagne entière. Les écritures et caches sont locaux. Le téléchargement utilise IPv4 dans ce processus car la route IPv6 locale échoue ; il vérifie l'empreinte officielle du fichier.

# D06 — Corroboration par blocage de composants publiés

1. Utiliser les Backup Name Movers de Wang, annoncées indépendamment de nos scores. Pour L9H9/L9H6, retenir seulement les six têtes strictement aval : L10H1,H10,H6,H2 ; L11H9,H2. L9H0/H7 sont simultanées et ne peuvent s'adapter à la sortie d'une autre tête de leur couche.
2. Figer leurs activations natives au dernier token pendant la reprise perturbée. Les autres composants restent libres. Rejouer les contributions natives n'est pas ablater les têtes à zéro.
3. Comparer au blocage de six autres têtes, quatre en couche10 et deux en couche11, fixé avant le pilote. Pour L10H0, ne garder que les deux têtes de couche11 et leur témoin2. La cardinalité seule ne rend pas le témoin inerte.
4. Rapporter perte blocage−perte adaptée, effet du témoin et leur différence B. Un B positif ne suffit pas si le blocage des backups améliore lui aussi la perte : la nécessité exige un effet absolu défavorable et une récupération utile.
5. Si ces prédictions échouent, conserver l'échec de l'attribution restreinte, sans chercher après test une autre liste de têtes. Les MLP, les têtes négatives et la LN peuvent encore contribuer ; un échec ne réfute pas tout self-repair.

Les statistiques sont conditionnées à une porte logicielle et de dommage passée. [Rapport](RAPPORT.md).

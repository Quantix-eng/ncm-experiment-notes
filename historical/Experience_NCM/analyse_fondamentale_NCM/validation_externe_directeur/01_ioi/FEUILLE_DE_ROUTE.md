# D03–D04 — GPT-2 small sur MIB–IOI

## Proposition et question

Appliquer le protocole à une compensation étudiée indépendamment, en donnant priorité à GPT-2 small sur IOI. La question principale porte sur la nécessité d'adaptation aval, sa spécificité et une corroboration par blocage ; elle ne porte pas sur l'attribution de l'une des six classes analytiques à tout GPT-2.

## Séquence méthodologique

1. Épingler poids, jeu public, tokenizer et implémentation. Vérifier les trois variantes et les réponses à un token. Sélectionner 128 familles dev et 512 test sans sorties de réseau, ni filtre de justesse ou d'effet.
2. Définir L9H9 primaire, deux donneurs distincts au même site, cinq croisements, mesures NLL/logits/justesse et contrôle de blocage. Déclarer ce qui reste disponible depuis le receveur.
3. Geler scripts et protocole, exécuter seulement développement. Reproduire le sham bit à bit, les contributions dans l'ordre, puis comparer à une passe officielle munie d'un hook sur la tête.
4. Contrôler la porte pratique : dommage moyen ≥0,05 nat et ≥25 % des familles avec dommage moyen >0,01 nat. Si échec, fermer la branche avant test, sans choisir une autre tête plus favorable.
5. Si réussite, sceller le protocole test avant ses sorties. Exécuter 512 familles, puis L9H6 et L10H0 comme résultats secondaires séparés avec leur propre frontière. Garder tous les cas, y compris échecs natifs et récupérations négatives.
6. Calculer contrastes appariés par famille, décrire les templates, comparer les estimands sans confondre désaccord et erreur. Juger si l'intervention de confirmation soutient une conclusion supplémentaire.

## Livrables et coûts

[Protocole précis](PROTOCOLE_DEV.md), `data/provenance.json`, `data/data_audit.json`, `data/selected.json`, `results/freeze_dev.json`, observations et états résiduels, analyses et audit. Le [rapport](RAPPORT.md) donne l'état final. Les temps écoulés des runners seront mesurés ; ils ne seront pas appelés benchmark matériel isolé si d'autres branches calculent simultanément.

## Limites attendues

Les donneurs natifs et les données sont extérieurs, mais l'implémentation est interne. Les représentations échangées peuvent être hors distribution. Les deux donneurs d'une famille ne sont pas deux unités indépendantes ; le réemploi de la banque de donneurs conditionne les intervalles. La compensation publiée sous knockout conjoint des trois Name Movers ne garantit pas un succès pour une seule tête rééchantillonnée. Le test public n'est ni secret ni indépendant de tout historique de préentraînement.

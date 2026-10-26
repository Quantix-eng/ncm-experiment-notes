# Manuscrit unifié : plan et règles de rédaction

Synthèse du 24 octobre, plan révisé le 25 octobre 2026. La phase exploratoire est close. Aucune nouvelle inférence, perturbation, sonde, simulation ou estimation statistique ne sera exécutée pour cette synthèse. Les opérations autorisées ici sont l'inventaire, la copie, l'extraction de valeurs déjà enregistrées, la mise en page et les vérifications documentaires et d'intégrité.

## Thèse centrale

**Que permettent réellement d'attribuer les révisions apparentes du NCM ?** C'est la question qui relie le manuscrit. Nous partons du comportement observé, identifions les explications que les contrôles permettent de séparer, puis terminons par une propriété locale exploitable et vérifiable. Nous ne construisons pas une chaîne causale pooling → oscillations → alphabet : cette chaîne n'a pas été testée.

Phrase directrice : « Le codec suffit à produire des reconstructions non monotones ; une symétrie de l'agrégation explique une perte d'information bien définie, sans être identifiée comme la cause de ces oscillations ; le test d'omission ne confirme pas la compensation spécifique annoncée ; enfin, les codes terminaux permettent des éditions locales mesurées et une lecture partielle certifiée. »

L'apport visé est une attribution mécaniste bornée et auditable sur les checkpoints étudiés. Une anatomie complète, une contribution majeure et une nouveauté fondamentale ne sont pas des acquis à déclarer. Leur appréciation appartient à l'évaluation scientifique externe.

## Architecture du texte

1. **Introduction : de la révision visible à la question d'attribution.** Figure 1 et quatre réponses circonscrites ; aucune affirmation sur la conscience.
2. **Cadre expérimental : que lisons-nous et qu'intervenons-nous ?** Récurrence, distinction codes/valeurs, données et statut rétrospectif.
3. **Une grille qui change implique-t-elle une décision du générateur ?** Réfutation du verrouillage, contrôle codec oracle et extension texte. Le codec est suffisant, son rôle exclusif n'est pas établi.
4. **Quelle information l'encodeur peut-il conserver ?** Invariance de la moyenne additive, niveau constant, correction pondérée et limite du générateur gelé. Cette section établit un mécanisme d'agrégation distinct ; elle ne prétend pas expliquer les oscillations.
5. **L'adaptation à une omission répare-t-elle spécifiquement l'état ?** Plan croisé et contraste annoncé, avec les quatre conditions visibles dans le corps du texte.
6. **Qu'est-ce qui devient directement lisible et manipulable au dernier niveau ?** Édition terminale, contrôle limité, certificat externe et coût ; pas de récit de stabilisation démontrée.
7. **Discussion et limites de l'attribution.** Retour à la question initiale, antécédents, explications encore ouvertes et portée des résultats négatifs.
8. Conclusion brève ; références ; annexes méthodologiques et résultats secondaires ; supplément de preuves et code.

Chaque section suit la même progression : question → contrôle discriminant → observation → conclusion autorisée → limite. Les transitions expliquent pourquoi la question suivante reste ouverte, au lieu de juxtaposer quatre études.

## Figure 1 et budget de lecture

La Figure 1 représente les chemins exécutés : solution connue → encodeur/quantificateur, ou puzzle → générateur, puis accumulation résiduelle et décodeur. Quatre annotations numérotées localisent les observations. Les flèches représentent le calcul ; les numéros repèrent un contrôle ou une lecture d'observation, sans ajouter de flèches entre les découvertes. Aucun lien ne représente une causalité non testée entre pooling et oscillations. Le niveau 5 possède neuf positions ; il est distinct du niveau de génération 9, qui en possède 37.

Objectif éditorial : environ 8–12 pages de corps principal, sans altérer la feuille de style officielle pour imposer une longueur. Garder dans le corps les preuves courtes, définitions des contrastes, chiffres principaux avec IC et limites qui changent l'interprétation. Déplacer après les références les détails des fenêtres, tableaux exhaustifs, contrôles de bruit, incidents de chargement et résultats secondaires, y compris les résultats défavorables. L'archive supplémentaire regroupe les JSON et scripts des **huit campagnes**, pas seulement huit fichiers JSON. Un résultat négatif nécessaire à l'interprétation ne doit pas être dissimulé en annexe.

Le brouillon composé tient finalement en **7 pages avec les références, puis 7 pages d'annexes**. Cette longueur naturelle ne justifie pas d'allonger artificiellement le texte pour atteindre le budget indicatif.

## Formulations à conserver

- Le théorème porte sur une moyenne avant normalisation et histoire résiduelle ; il ne force pas l'effondrement du tokenizer complet.
- Le codec suffit à produire des revirements sans génération ; leur origine exclusivement codec n'est pas démontrée. Aucune expérience ne mesure la conscience.
- Aucune intervention ne relie la symétrie du pooling aux oscillations. L'alphabet terminal n'établit pas une stabilisation dynamique ni l'absence de conscience.
- Une compensation utile et spécifique à l'omission 9 n'est pas établie. L'expérience ne prouve pas que toute adaptation serait aveugle ou inutile.
- L'échec d'un test de compensation et les sondes externes ne constituent pas un test exhaustif de métacognition : ne pas écrire « NCM ne fait pas de métacognition selon nos métriques » comme conclusion générale.
- L'édition terminale est observée sous un contrôle gaussien limité. Sa nouveauté, sa généralité et une sémantique indépendante du contexte ne sont pas établies.
- La garantie de non-dégradation vient de l'unicité, du vérificateur externe et du repli exact, pas d'un mécanisme appris du NCM.
- La synthèse est rétrospective. Elle ne crée ni prérégistration globale, ni nouvelle confirmation indépendante. Les comparaisons locales conservent leurs critères, leurs IC et leurs limites d'origine.

## Organisation des preuves

Chaque affirmation centrale reçoit un identifiant, des pointeurs vers les JSON historiques, les scripts correspondants, les traces et une limite explicite. Les chiffres imprimés sont extraits de ces pointeurs ; aucune nouvelle moyenne ou incertitude n'est calculée. Les petits tableaux NPY peuvent être copiés, octet pour octet, hors des archives NPZ existantes pour produire un supplément léger. Les poids et grands états restent dans les dossiers historiques, avec leurs empreintes.

Les huit campagnes restent intactes. Le dossier unifié constitue une nouvelle vue documentaire et une copie organisée du code, pas une réécriture de leur histoire. Le format visé est le style TMLR officiel en mode brouillon/preprint, sans prétendre qu'une soumission a eu lieu.

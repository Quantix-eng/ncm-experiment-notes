# Rapport — antécédents directs et corrections expérimentales

25 octobre 2026. Consultation ciblée des sources primaires ; il ne s'agit pas d'une revue systématique de littérature.

## Self-Repair : priorité vérifiée

Rushing et Nanda publient **Explorations of Self-Repair in Language Models**, ICML 2024, PMLR 235, pages 42836–42855. Le §2.1 définit la mesure `SR = Δlogit − ΔDE_head`. Le §3 compare des contributions aval figées avec une normalisation finale libre ou figée. Une récupération du logit peut ainsi provenir de la lecture finale. L'effet direct est calculé avec le facteur de normalisation original ; les logits sont centrés sur le vocabulaire. Ces choix sont des conditions de comparaison, pas des détails facultatifs. [Notice ICML](https://proceedings.mlr.press/v235/rushing24a.html), [texte §2–3](https://arxiv.org/html/2402.15390v2#S2.SS1)

Conséquence pour notre étude : attribuer à notre protocole l'invention de la continuation figée ou de la séparation LayerNorm/aval serait erroné. La question encore ouverte est l'utilité empirique de leur combinaison avec les contrôles de dose, de spécificité et d'accès à l'information sur des interventions réservées.

## IOI : composants et régime d'intervention

Wang et al. identifient les Name Movers `(9,6),(9,9),(10,0)`. Leur annexe F donne huit backups : `(9,0),(9,7),(10,1),(10,2),(10,6),(10,10),(11,2),(11,9)`. Le résultat de backup cité porte sur le knockout conjoint des trois Name Movers ; il ne garantit pas une compensation après resampling d'une seule tête. Les auteurs indiquent aussi des limites de complétude de leur circuit. [Article IOI, §3.1 et annexe F](https://arxiv.org/html/2211.00593v1#A6)

L'application à nos frontières impose les choix suivants, par la structure causale du forward GPT-2 :

- intervention `(9,9)` ou `(9,6)` : seules les quatre têtes backup de couche 10 et les deux de couche 11 sont strictement aval ;
- intervention `(10,0)` : seules `(11,2),(11,9)` sont strictement aval ;
- les sorties d'autres têtes de la couche attaquée sont calculées à partir du même état préalable. Elles ne peuvent répondre à la substitution de sortie d'une tête sœur.

Le blocage de l'ensemble aval documenté ne couvre pas toute compensation possible. Les MLP, les têtes de suppression et la normalisation finale restent d'autres canaux. Une absence d'effet invaliderait l'attribution à cet ensemble dans cette expérience, sans réfuter toute adaptation du réseau.

Un groupe témoin doit être de même cardinalité et distribué sur les mêmes couches. Il doit être choisi avant les effets. Les têtes négatives `(10,7),(11,10)` ne sont pas des témoins supposés inertes : une réduction de suppression peut elle-même contribuer à la compensation. L'analyse de copy suppression fournit un antécédent distinct à prendre en compte dans l'interprétation. [McDougall et al., BlackboxNLP 2024](https://aclanthology.org/2024.blackboxnlp-1.22/)

## Antécédent complémentaire

**The Hydra Effect** étudie déjà des changements de contributions aval après ablation dans le rappel factuel. Cette priorité supplémentaire interdit de présenter la compensation aval comme une découverte du présent programme. [McGrath et al., 2023](https://arxiv.org/abs/2307.15771)

## Incohérence bibliographique Pythia

Dans Self-Repair §3.2, le texte désigne L11H10, alors que la légende de la figure 5 désigne L11H0. Le même passage donne L11H2 pour l'analyse du facteur de normalisation. Pour notre campagne, H2 et H10 suivent la recommandation et le texte ; l'incohérence doit rester visible. Une future inspection du code des auteurs pourra résoudre la légende, sans changer silencieusement les cibles après calcul.

## MIB et classification : objets distincts

MIB comporte une piste de localisation de circuits et une piste de localisation de variables causales. Un score d'arêtes EAP-IG n'est donc pas une attribution de compensation. Une comparaison exige soit un même objet de localisation, soit une règle annoncée transformant les propositions en prédictions interventionnelles réservées. [MIB, ICML 2025](https://proceedings.mlr.press/v267/mueller25a.html)

## Livrables et portée

Les corrections sont transmises aux volets expérimentaux. Les [paragraphes de manuscrit](../CORRECTIONS_MANUSCRIT.md) insèrent ces antécédents sans changer les nombres historiques. Les contrôles numériques des nouvelles campagnes et une évaluation extérieure restent nécessaires ; ce rapport ne les remplace pas.

"""Render a French report from the retained measurements; never rerun inference."""
import json
from pathlib import Path

D = Path(__file__).resolve().parent


def main():
    q = json.loads((D / "results/summary.json").read_text())
    v = json.loads((D / "results/verification.json").read_text())
    f = json.loads((D / "results/protocol_frozen.json").read_text())
    assert v["status"] == "passed"

    def pct(x):
        return f"{100*x:.2f}".replace(".", ",")

    def result(key):
        r = q[key]
        count = f"{r['count']:,}".replace(",", " ")
        total = f"{r['total']:,}".replace(",", " ")
        return f"**{count}/{total} = {pct(r['rate'])} %**, IC descriptif 95 % [{pct(r['ci95'][0])} ; {pct(r['ci95'][1])}]"

    favorable = q["decision"] == "no_composition_veto_at_tested_scope"
    interpretation = ("Le pilote ne détecte pas de forte dégradation propre à la composition selon les seuils locaux fixés : les doubles éditions restent donc une possibilité empirique, limitée au domaine testé."
                      if favorable else "Le pilote franchit au moins un seuil local de dégradation de la composition : la règle testée ne justifie pas d'investissement architectural supplémentaire.")
    strata = "\n".join(
        f"- `{s['stratum']}` : deux cibles {s['both']['count']}/{s['both']['total']} ({pct(s['both']['rate'])} %) ; contrat complet {s['clean']['count']}/{s['clean']['total']} ; rétention {s['retention']['count']}/{s['retention']['total']} ({pct(s['retention']['rate'])} %)."
        for s in q["by_stratum"])
    text = f"""# Composition de deux écritures terminales — rapport du 25 octobre 2026

{interpretation} **Ce résultat ne démontre ni une interface symbolique fiable pour des demandes arbitraires, ni un avantage architectural.** Les échecs absolus et ceux des éditions isolées sont conservés ci-dessous.

## Expérience et statut

32 puzzles à 30 indices, indices 512 à 543 du corpus historique `recherche_positive/data/fresh.json` ; ces sources ne sont pas celles des 128 puzzles de l'ancien test causal. Ce corpus a déjà servi à analyser les sorties natives : il ne s'agit pas d'un nouvel échantillon réservé. Un seul checkpoint hard, CPU FP32, un thread ; aucun apprentissage, solveur, recherche de code par exemple ou changement de table.

Le gel local date de **{f['utc']}**. Il inclut le protocole, les scripts d'expérience et de vérification, les données, les tables, les poids et les sources importées. Le journal ne constitue pas une prérégistration indépendante. Les empreintes sont inchangées avant et après les tests. Les états natifs régénérés correspondent bit à bit aux codes et latents des 32 sources conservées ; les 32 shams reproduisent leurs sorties.

Une paire de cases initialement vides est tirée, à partir des seuls indices, dans chacune de cinq configurations par puzzle. Les huit chiffres différents de la sortie native de chaque case donnent 64 cibles doubles par paire. Les cibles identiques entre cases sont incluses. Total : **160 paires, 10 240 doubles éditions, 2 560 éditions isolées**. Les positions peuvent se répéter entre configurations ; les essais ne sont pas indépendants.

L'intervention est `z_final + (R12(c12 édité) - R12(c12 natif))`. Toutes les autres entrées discrètes restent fixes. Le résidu commun est recalculé pour les deux codes simultanément. On mesure les effets après le décodeur ; la commutation triviale des affectations de codes n'est pas le résultat testé. Aucune continuation du générateur n'est exécutée.

## Résultats

- Deux chiffres demandés écrits : {result('both')}.
- Un seul chiffre demandé écrit : {result('one')}.
- Aucun des deux chiffres demandés écrit : {result('neither')}.
- Contrat complet — les deux cibles, les 79 autres cases et les huit séparateurs préservés : {result('clean')}.
- Contrats d'écriture isolée réussis : {result('single_clean')}.
- Conjonction des deux contrats isolés correspondants : {result('matched')}.
- Rétention du contrat double lorsque les deux contrats isolés passent : {result('retention')}.
- Contrats isolés tous deux réussis mais contrat double raté : **{q['loss']['count']}/{q['matched']['count']}**, répartis sur **{q['sources_with_loss']}/32 sources**.
- Contrat double réussi malgré l'échec d'au moins un contrat isolé : **{q['gain']['count']}**.
- Double édition changeant au moins une des 79 autres cases : **{q['collateral']['count']}/10 240**, sur **{q['sources_with_collateral']}/32 sources**.
- Double édition changeant un séparateur : **{q['format_change']['count']}/10 240** ; séparateur invalide : **{q['format_invalid']['count']}/10 240**.
- Deux cases cibles toutes deux inchangées : **{q['both_unchanged']['count']}/10 240**. « Aucun chiffre demandé » n'implique donc pas automatiquement l'absence de tout changement.

La perte nette par rapport à la conjonction isolée vaut **{pct(q['net_composition_loss']['rate'])} point(s)**, IC descriptif 95 % [{pct(q['net_composition_loss']['ci95'][0])} ; {pct(q['net_composition_loss']['ci95'][1])}]. Cette comparaison est appariée aux mêmes cases et cibles ; ce n'est pas le carré d'un taux moyen de réussite.

{strata}

Les demandes avec chiffres identiques et différents sont séparées dans `summary.json`. Les comparaisons entre configurations restent descriptives : les relations spatiales et Sudoku ne sont pas orthogonales.

## Décision et portée scientifique

La règle exploratoire, fixée avant les sorties, signalait une forte dégradation si la rétention était inférieure à 95 %, ou si la perte nette dépassait 5 points. Décision du runner : **`{q['decision']}`**. Ces seuils sont des choix pratiques du pilote, sans justification confirmatoire indépendante. {interpretation}

La rétention globale de 94,59 % est proche du seuil heuristique et son IC recouvre 95 % : il ne faut pas transformer le franchissement ponctuel en rejet statistique confirmé. Le constat le plus marqué est local : pour les cases voisines, 210 des 1 182 conjonctions isolées réussies échouent après composition, soit **17,77 % de pertes** ; la rétention vaut **82,23 %**, IC descriptif 95 % [77,79 ; 85,80]. Les cas sans unité commune ont une rétention de **99,44 %**. Ces différences motivent une dépendance au contexte spatial dans le domaine testé, sans en localiser la cause dans la convolution, l'attention ou une autre partie du lecteur.

Les 370 gains compensent numériquement les 298 pertes : le taux global du contrat double est même supérieur de 0,70 point à la conjonction des contrats isolés. Il n'y a donc pas d'effondrement global établi de la précision moyenne. Le résultat négatif concerne la **fiabilité des contrats individuels sous composition**, particulièrement pour des cases voisines, et la fiabilité absolue de l'interface proposée. Je recommande de ne pas engager la Piste 3 comme projet architectural en l'état ; cela ne prouve pas l'impossibilité d'une autre règle d'écriture.

Un audit **post-hoc**, distinct des critères gelés, sépare les cibles identiques des cibles différentes. Entre voisines, la rétention est de 9/130 = 6,92 % pour deux chiffres identiques, contre 963/1 052 = 91,54 % pour deux chiffres différents ; **89/1 052 = 8,46 %** des contrats isolés réussis échouent donc encore avec des cibles différentes. La perte ne se réduit pas aux demandes de même chiffre dans une ligne. Des cibles différentes ne garantissent toutefois pas une grille Sudoku valide. Cet audit ne localise aucun mécanisme et ne change pas la décision prospective. Voir `audit_target_equality.py` et `results/posthoc_target_equality_audit.json`.

Un échec de la demande double ne démontre pas à lui seul une interaction destructrice : il faut regarder les deux écritures isolées correspondantes. Réciproquement, une bonne rétention conditionnelle ne transforme pas la règle d'écriture en interface totale : elle conditionne l'analyse sur les opérations qui fonctionnent déjà isolément, sans fournir une méthode gratuite pour les reconnaître avant décodage.

Une modification directe des deux tokens de sortie satisfait exactement le même contrat terminal par construction, sans intervention latente. Le présent pilote n'établit aucune capacité qui lui échappe. Il ne démontre ni raisonnement révisé, ni adaptation des autres cases, ni accélération, ni amélioration de résolution. Les chiffres demandés peuvent rendre le Sudoku invalide ; la validité de la grille n'est pas l'objectif de cette intervention. Aucun solveur n'a servi à choisir une cible favorable.

Le résultat est conditionnel au checkpoint, à la table, au corpus non uniforme et aux paires testées. Il ne couvre pas les 16² codes, toutes les positions, plusieurs checkpoints, trois éditions ou plus, ni des interventions suivies de continuation. Il ne constitue pas un certificat universel.

## Incertitude et contrôles

Les IC sont des percentiles de 10 000 rééchantillonnages des **32 puzzles sources entiers**, configurations et cibles conservées ensemble. Ils sont descriptifs, sans tests de significativité ni correction de multiplicité. Les bornes bootstrap dégénérées en l'absence d'événement ne prouvent pas un taux populationnel nul. Pour un événement absent de toutes les sources, la borne supérieure de l'intervalle Wilson bilatéral à 95 % sur la proportion de sources affectées vaut **{pct(q['zero_event_source_wilson_upper95'])} %**, sous une approximation binomiale ; elle ne borne pas directement le taux par intervention.

Le script distinct `verify.py` recalcule les métriques par boucles scalaires, les intervalles globaux et la rétention. Il rejoue les **400 interventions de la première source** avec des appels singleton, sur les 89 tokens, et retrouve toutes les sorties stockées. C'est un audit logiciel par le même agent de recherche, pas une réplication indépendante ou une revue externe. Temps écoulé du runner : {q['elapsed_seconds']:.1f} s, chargement et contrôles compris ; ce temps n'est pas un benchmark comparatif.

## Artefacts et arrêt demandé

- [Protocole](protocol.json), [gel et empreintes](results/protocol_frozen.json), [journal](JOURNAL.md).
- [Résultats agrégés](results/summary.json), [toutes les observations](results/interventions.npz), [audit](results/verification.json).
- [Commandes de reproduction et vérification](README.md), [runner](experiment.py), [vérificateur](verify.py).

**Arrêt après l'étape 1.** Le benchmark causal de l'étape 2 et le travail sur Wyner de l'étape 3 n'ont pas été lancés. Le manuscrit et les campagnes antérieures sont inchangés.
"""
    (D / "RAPPORT.md").write_text(text)
    print(D / "RAPPORT.md")


if __name__ == "__main__":
    main()

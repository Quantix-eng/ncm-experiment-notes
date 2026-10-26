"""Markdown report from retained observations; no charts or presentation."""
import json
from pathlib import Path

D = Path(__file__).resolve().parent


def main():
    q = json.loads((D/'results/summary.json').read_text())
    audit = json.loads((D/'results/verification.json').read_text())
    sources = json.loads((D/'data/families.json').read_text())
    truth = json.loads((D/'data/evaluator_truth.json').read_text())
    observations = json.loads((D/'results/observations.json').read_text())
    assert audit['status'] == 'passed'
    def percent(x): return f'{100*x:.2f}'.replace('.',',')+' %'
    scores = []
    for name, groups in q['scores'].items():
        s = groups['all']
        precision = 'non définie, aucune attribution unique' if s['singleton_accuracy'] is None else percent(s['singleton_accuracy'])
        scores.append(f"- `{name}` : vraie classe conservée dans {s['covered']}/{s['n']} cas ; "
                      f"{s['singleton']} attributions uniques, dont {s['singleton_correct']} correctes ; "
                      f"précision des attributions uniques {precision} ; taille moyenne d'ensemble {s['mean_set_size']:.3f}.")
    score_text = '\n'.join(scores)
    test_text = '\n'.join(f"- `{name}`, familles réservées : {g['split:evaluation']['singleton_correct']}/{g['split:evaluation']['n']} identifications exactes ; "
                          f"couverture {percent(g['split:evaluation']['coverage'])}." for name,g in q['scores'].items())
    natural = [row for row in sources['rows'] if row['family']=='sudoku']
    clues = [row['clues']['a'] for row in natural]
    changes = [len(row['changed_answer_cells_excluding_clue']) for row in natural]
    first_fixed = next(row for row in observations if truth[row['case_id']]['label']=='fixed_response')
    trap = next(t for t in first_fixed['crossed'] if t['direction']==0 and t['sign']==1)
    text = f"""# Attribution des révisions : protocole et calibration des témoins

Rapport de l'étape 2 — 25 octobre 2026. **Un protocole exécutable est maintenant formalisé et calibré sur des mécanismes connus par construction.** Les témoins présentent les mêmes trajectoires et la même réussite finale ; les interventions permettent d'en distinguer certains, et imposent une abstention pour d'autres. Ce résultat valide le fonctionnement interne du dispositif sur cette bibliothèque, pas une performance externe d'interprétabilité ni un mécanisme NCM.

## 1. Ce qui a été construit

Le [protocole complet](PROTOCOLE.md) définit les accès O (trajectoires seules), C (continuations croisées à entrée fixe) et E (changements d'entrée, échanges d'état et ablation de suite). Il fixe les sorties accessibles, les perturbations, les pertes, les partitions, les méthodes de diagnostic, les contrôles et les conditions d'arrêt.

Les témoins sont : relecture oracle d'une réponse déjà disponible ; tolérance à un canal perturbé avec suite fixe utile ; compensation depuis une ancre d'état ; recalcul oracle depuis l'entrée ; réponse fixe à toute perturbation détectée ; et mélange des sources état/entrée. Le recalcul peut lui-même compenser : ces noms ne représentent pas des catégories causales mutuellement exclusives dans tous les systèmes.

Toutes les réponses correctes sont fournies à la construction par un oracle de tâche. Aucun témoin n'est un solveur neuronal entraîné. Le recalcul oracle consulte la réponse associée à l'entrée courante ; il ne démontre pas qu'un NCM exécute un algorithme Sudoku.

Les quatre lectures natives partagent un lecteur oscillant fixé à l'avance. Toutes les trajectoires discrètes d'une même source sont identiques et toutes les sorties finales sont exactes. Une méthode ne peut donc pas gagner en classant les systèmes par précision finale. Ce contrôle est analytique et construit pour exposer une limite de mesure ; sa réussite ne constitue pas une découverte empirique sur des réseaux.

## 2. Corpus et partitions effectivement exécutés

**{q['n_families']} familles sources**, soit 12 égalités, 12 parités, 12 permutations et 16 familles Sudoku nouvelles. Égalité et parité sont réservées au développement ; permutations et Sudoku à l'évaluation structurelle. Aucun seuil n'a été ajusté sur leurs sorties. Toutes les variantes et interventions d'une source restent dans sa partition.

Cinq programmes purs sont exécutés sur chaque source. Sur les 28 sources réservées, trois mélanges supplémentaires (`alpha=0.25, 0.5, 0.75`) sont exécutés, sans exposition au développement. Total : **{q['n_cases']} instances de programme**, 4 perturbations par instance et **12 conditions naturelles entrée/état/suite par instance**. Les trois variantes de mélange augmentent leur poids dans un taux micro ; les résultats par classe et la macro-moyenne sont conservés séparément dans `summary.json`.

Ces partitions n'établissent pas une généralisation apprise : les règles sont dérivées des équations connues, et aucune méthode n'est entraînée ici. Les clés analytiques exposent des réponses contrôlées et ne sont pas des problèmes de difficulté comparable au Sudoku. Les nombreuses interventions ne forment pas des réplications indépendantes.

## 3. Contre-factuels Sudoku vérifiés

Les **16 bases ont exactement deux solutions**, contrôlées par MRV et Algorithm X avec une limite de trois. Ajouter l'un de deux chiffres alternatifs à une case où les solutions diffèrent produit **32 puzzles à solution unique**, appariés par famille, de même nombre d'indices et différents en une seule case.

Un troisième puzzle par famille ajoute un indice vrai à la branche A : **16 contrôles à réponse inchangée**, également uniques selon les deux solveurs. Au total, chaque famille fournit une base ambiguë et trois variantes uniques. Les branches A/B ont entre **{min(clues)} et {max(clues)} indices** ; leur réponse diffère sur **{min(changes)} à {max(changes)} cases hors indice modifié**. La difficulté humaine n'a pas été mesurée.

Le contrôle invariant change le nombre d'indices ; il ne neutralise donc pas ce facteur. Les paires naturelles améliorent le contrôle des réponses mais ne rendent pas les échanges d'état automatiquement sur distribution. Les clés D4 et normalisation de chiffres sont exclues contre les corpus locaux officiels et externes et contre les familles déjà retenues ; ce contrôle ne couvre pas toutes les isomorphies ni les données d'entraînement inconnues. Les essais de génération, y compris les rejets structurels, sont conservés.

## 4. Résultats des méthodes de calibration

{score_text}

{test_text}

L'observation prudente conserve toutes les classes : elle évite une attribution fausse mais ne fournit aucune localisation. La méthode croisée réduit l'ambiguïté tout en conservant la vraie classe ; sa couverture ne doit pas être confondue avec une identification exacte. Le diagnostic complet distingue les programmes annoncés dans cette bibliothèque finie. Une réponse intermédiaire est nommée mélange uniquement sous cette hypothèse de bibliothèque ; dans un réseau ou une famille inconnue, il faut répondre « mélange ou mécanisme non modélisé ».

**Le résultat parfait du diagnostic complet est attendu par construction.** Les mêmes équations ont servi à définir les contrôles et à concevoir les règles. Il s'agit d'un test de calibration du protocole et du code, pas d'un résultat compétitif sur des méthodes indépendantes, d'un test secret ou d'une validation de généralisation à un réseau appris.

## 5. Deux limites d'identifiabilité mises à l'épreuve

À trajectoires seules, les programmes sont indiscernables. Le témoin oracle, dont la réponse est déjà disponible avant les lectures, peut présenter correction puis dégradation puis correction sans adapter sa continuation. Une règle forçant une compensation depuis l'état se trompe donc sur tous les autres programmes alors que leur précision finale est identique.

À entrée fixe, les programmes de compensation, de recalcul et de mélange ont les mêmes réponses aux continuations croisées testées. Les contrastes positifs montrent ici une récupération fonctionnelle spécifique, mais ne disent pas si la cible vient de l'entrée ou de l'ancre. Les changements d'entrée et d'état réciproques sont nécessaires pour distinguer ces programmes dans notre bibliothèque. Les exemples d'indiscernabilité sont conservés dans `results/nonidentifiability_examples.json`.

Le programme de réponse fixe fournit aussi un piège : sur la première perturbation positive, **R={trap['R']:.6g} et I={trap['I']:.6g}**, mais **S={trap['S']:.6g}**. Sa récupération peut être parfaite avec R et I positifs, bien qu'il applique la même correction à une autre perturbation. Les suites mal appariées et les quatre doses détectent ce défaut. Une seule récupération favorable ne suffit pas à établir une adaptation à l'identité de la perturbation.

La MSE des scores continus est le critère de calibration ; les argmax sont secondaires. Une perturbation de signe négatif peut augmenter la perte sans changer le chiffre lu. Il ne faut ni écarter ces cas après observation, ni les assimiler à une absence d'effet latent. Aucune NLL n'est attribuée à ces scores analytiques.

## 6. Gel et vérification

Gel local avant génération : **{q['freeze_utc']}**. Il épingle protocole, scripts scientifiques, solveurs et sources importées, corpus historiques lus, rapports antérieurs et manifestes. Il ne s'agit pas d'une prérégistration indépendante. Aucun amendement des équations, seuils ou scripts gelés n'a été nécessaire après les nouvelles observations.

L'audit vérifie les solutions par deux solveurs, les partitions, les équations de sortie en forme fermée, les shams, les trajectoires communes, les limites d'identifiabilité, les projections d'accès et les comptes de score. Il passe sur **{audit['cases']} instances**, avec **{audit['checks']['closed_form_outputs']} comparaisons de sorties aux formes fermées**. Ce sont des contrôles logiciels par le même agent de recherche ; ils ne constituent pas autant de réplications scientifiques indépendantes. Les empreintes gelées sont inchangées.

Les nombres rapportés décrivent une collection finie construite ; aucun IC populationnel ou test de significativité ne transforme les requêtes répétées en observations indépendantes. Temps d'exécution du runner : **{q['elapsed_seconds']:.1f} s**, génération incluse ; ce n'est pas une mesure comparative d'efficacité.

## 7. Ce qui reste à établir

Cette étape fournit un protocole d'évaluation et une calibration initiale, **pas encore un benchmark de référence**. Les méthodes sont des diagnostics analytiques définis par le même auteur que les témoins. Les familles de tâches réservées et les mélanges réservés exercent le dispositif sans remplacer une validation extérieure.

Il reste à implémenter et geler l'adaptateur NCM sur les contre-factuels naturels, puis à tester au moins une famille apprise distincte et des méthodes d'attribution proposées indépendamment. Les systèmes appris devront être analysés par leurs effets fonctionnels, en conservant les cas non identifiables ; leurs classes ne seront pas obtenues en leur attribuant nos étiquettes de construction. Aucune de ces campagnes apprises n'est déclarée exécutée ici.

Les résultats négatifs antérieurs restent intacts : la non-confirmation de compensation pour une omission précise ne devient pas une étiquette « système incapable de compenser ». Le pilote de composition terminale fournit une limite d'une règle d'écriture ; il n'établit pas un mécanisme de révision.

## 8. Positionnement et livrables

[MIB, ICML 2025](https://mib-bench.github.io/) propose des évaluations de localisation causale et un modèle à circuit connu. Notre objectif plus étroit est d'évaluer l'attribution d'un changement temporel face à un lecteur qui peut produire une apparence de correction. Cette différence d'objectif ne démontre aucune nouveauté ou priorité.

[Tyen et al., ACL 2024](https://aclanthology.org/2024.findings-acl.826/) séparent localisation d'erreur et correction avec une indication fournie. Notre protocole rend également explicite la provenance de l'information utilisée par les témoins ; il ne reproduit pas leurs expériences. Ces sources primaires ont été consultées le 25 octobre 2026 ; il ne s'agit pas d'une revue systématique.

- [Protocole détaillé](PROTOCOLE.md), [paramètres](protocol.json), [gel](results/protocol_frozen.json).
- [Familles et contre-factuels](data/families.json), [vérité réservée à l'évaluateur](data/evaluator_truth.json).
- [Observations et attributions](results/observations.json), [scores détaillés](results/summary.json), [audit](results/verification.json).
- [Reproduction](README.md), [équations des témoins](controls.py), [règles d'accès et diagnostic](methods.py).

**Arrêt après l'étape 2.** Aucun travail sur Wyner, aucune nouvelle inférence NCM et aucune modification du manuscrit ne font partie de cette exécution. La priorité suivante est une décision sur la campagne apprise, avant toute revendication de benchmark général.
"""
    (D/'RAPPORT.md').write_text(text)
    print(D/'RAPPORT.md')


if __name__ == '__main__':
    main()

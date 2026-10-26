"""Post-inference counts of predeclared exactness, clues and format endpoints.

This descriptive exporter was added after the main results; it changes no
frozen equation, threshold, primary contrast or decision. No model is called.
"""
from support import D,np,CELL,SEP,data,write,check_freeze
from common import UNITS


def quality(ids,reference,prefix):
    digits = ids[CELL]
    givens = prefix[1:-1][CELL]
    return dict(exact=bool(np.array_equal(ids,reference)),
                bad_digits=bool(np.any((digits<1)|(digits>9))),
                bad_separators=bool(np.any(ids[SEP]!=10)),
                violated_clues=bool(np.any((givens!=0)&(digits!=givens))),
                invalid_units=bool(any(sorted(digits[u])!=list(range(1,10)) for u in UNITS)))


def aggregate(rows):
    return dict(n=len(rows),**{key:sum(row[key] for row in rows) for key in rows[0]})


def main():
    check_freeze()
    _,prefix,ref,_,_ = data()
    with np.load(D/'results/native.npz') as z:
        ni = z['logits'].argmax(-1)
    with np.load(D/'results/crossed.npz') as z:
        ci = z['logits'].argmax(-1)
    with np.load(D/'results/exchanges.npz') as z:
        ei = z['logits'].argmax(-1)
    result = dict(scope='Descriptive post-processing of predeclared output-quality endpoints; no new inference or decision change',
                  O={},C={},E={})
    for a,name in enumerate(('A','B','Aplus')):
        result['O'][name] = aggregate([quality(ni[f,a,-1],ref[f,a],prefix[f,a]) for f in range(16)])
    for k,name in enumerate(('Y00','Y10','Y11','Y01','Y1wrong')):
        result['C'][name] = aggregate([quality(ci[f,a,j,k],ref[f,a],prefix[f,a]) for f in range(16) for a in range(2) for j in range(4)])
    for pair,right in enumerate((1,2)):
        variants = (0,right)
        result['E'][('answer_change','answer_preserve')[pair]] = {}
        for name in ('fixed_left','fixed_right','regenerated','frozen_recipient_native','frozen_donor_native'):
            recipient,donor = [],[]
            for f in range(16):
                for i in range(2):
                    j = 1-i
                    kind = {'fixed_left':0,'fixed_right':1,'regenerated':2,
                            'frozen_recipient_native':i,'frozen_donor_native':j}[name]
                    ids = ei[f,pair,i,j,kind]
                    recipient.append(quality(ids,ref[f,variants[i]],prefix[f,variants[i]]))
                    donor.append(quality(ids,ref[f,variants[j]],prefix[f,variants[j]]))
            result['E'][('answer_change','answer_preserve')[pair]][name] = dict(recipient=aggregate(recipient),donor=aggregate(donor))
    write(D/'results/output_quality.json',result)
    crows = '\n'.join(f"- {name} : {item['exact']}/{item['n']} grilles exactes, "
                      f"{item['violated_clues']} violations d'indices, {item['bad_separators']} formats de séparateurs incorrects."
                      for name,item in result['C'].items())
    erows = '\n'.join(f"- `{name}` : solution du receveur {item['recipient']['exact']}/32 ; "
                      f"solution du donneur {item['donor']['exact']}/32 ; "
                      f"violation d'au moins un indice du receveur {item['recipient']['violated_clues']}/32."
                      for name,item in result['E']['answer_change'].items())
    q = __import__('json').loads((D/'results/summary.json').read_text())
    cprecise = '\n'.join(f"- {key} : {q['C'][key]['mean']:.8g} nat, IC 98,75 % "
                         f"[{q['C'][key]['ci_adjusted'][0]:.8g} ; {q['C'][key]['ci_adjusted'][1]:.8g}]."
                         for key in ('D','R','I','S'))
    text = f"""# Lecture des résultats réels : justesse, contrôles et limites

Complément du 25 octobre 2026 au [rapport principal](RAPPORT.md). Aucun nouveau calcul de réseau : uniquement les argmax des logits conservés. Les mesures de justesse, d'indices et de format étaient prévues au protocole. Leur export séparé par `audit_sorties.py` a été ajouté après lecture des résultats principaux ; il n'appartenait pas au gel initial. Les seuils, analyses principales et diagnostics E restent inchangés. Les comptes ci-dessous sont descriptifs.

## Conclusion de portée

Les interventions ont pu être exécutées sur les vrais tenseurs FP32 avec parité native exacte. Elles apportent une observation fonctionnelle, mais ne départagent pas un calcul adaptatif nécessaire d'une restitution par une suite déjà suffisante dans les cas testés.

Sur les 32 échanges A↔B, **la suite recalculée et la suite native du receveur figée donnent toutes deux 32/32 solutions du receveur**. La suite native du donneur figée donne 32/32 solutions du donneur, donc zéro solution du receveur. Les états avant reprise sont pourtant les mêmes dans chaque comparaison de suites. L'adaptation à l'état échangé n'apporte donc aucun gain d'exactitude finale par rapport au contrôle natif du receveur sur ces échanges.

Le contrôle utilise une suite obtenue auparavant par une génération native complète du receveur : ce n'est pas une réponse gratuite, ni un gain de calcul démontré. Ce résultat ne prouve pas non plus que n'importe quelle suite imposerait sa solution, que l'état serait ignoré ou qu'une régénération serait inutile sur d'autres perturbations. Les différences de confiance existent, comme le montrent les effets NLL du rapport principal.

Les 15/16 diagnostics de préférence dominée par l'entrée restent descriptifs. Dans `sudoku-012`, u/g=1,0069949, hors [0,1] : ce cas reste explicitement non identifiable selon le seuil fixé, sans arrondi ou reclassement opportuniste. **Aucun des 16 cas n'obtient une vérité de mécanisme interne.**

## C : aucun dommage de justesse à réparer à cette dose

{crows}

Les 128 essais proviennent de 16 familles et deux ancrages chacun ; ce ne sont pas 128 unités indépendantes. Toutes les conditions conservent les 89 tokens de leur solution propre, y compris les séparateurs. Des logits et des codes peuvent changer sans modifier l'argmax final. Le test C n'établit donc pas une réparation après dégradation importante ; il établit une robustesse de justesse pour cette dose et ces supports, et une équivalence du dommage moyen de NLL dans la marge préfixée. Il ne faut pas augmenter la dose dans ce même pilote pour chercher une récupération favorable.

Les valeurs suivantes évitent les arrondis à zéro des petits nombres dans le rapport principal :

{cprecise}

Toutes sont très inférieures à la marge de 0,01 nat fixée avant inférence. L'équivalence porte sur la moyenne et cette marge, pas sur une invariance exacte des logits ou de chaque entrée.

## E : toutes les suites figées et recalculées

Les comptes portent sur les deux échanges hors diagonale de chacune des 16 familles. `fixed_left` fige la suite A, `fixed_right` la suite B ; leur résultat agrégé ne doit pas être confondu avec un appariement systématique au receveur. Les deux lignes `frozen_recipient_native` et `frozen_donor_native` réalignent ces mêmes conditions déjà prévues dans le plan factoriel, sans sélectionner de nouvelles sorties.

{erows}

Tous ces résultats restent des Sudoku complets valides au sens des 27 unités et du format. Une sortie de donneur peut néanmoins violer l'indice discriminant du receveur : la validité d'une grille seule n'est pas la résolution de son puzzle d'entrée.

Dans les échanges A↔Aplus, la référence reste identique : les 32 sorties sont exactes pour chacune des trois politiques de suite (native du receveur, native du donneur, recalculée), sans violation des indices, chiffres ou séparateurs. Les variations de probabilités et NLL du rapport principal ne sont donc pas des changements de solution.

## Ce qui reste non identifié

Les données restent compatibles avec plusieurs réalisations internes utilisant les indices et les niveaux tardifs. Elles ne prouvent ni un recalcul depuis zéro, ni une compensation spécifique nécessaire, ni un mélange causal à coefficients constants. L'effet descriptif moyen de l'état sur la préférence A/B n'est pas nul ; il ne doit pas être effacé par la réussite des contrôles figés.

Un prochain test devrait être une **nouvelle campagne** préalablement gelée : autre point de coupure ou corruption produisant un dommage utile sur développement indépendant, puis évaluation sans sélection sur les cas effectivement abîmés. Aucun de ces essais supplémentaires n'a été exécuté ici. L'étape 3a s'arrête avec les résultats présents.
"""
    (D/'COMPLEMENT_SORTIES.md').write_text(text)
    report = D/'RAPPORT.md'
    if report.exists():
        marker = '<!-- lecture-sorties -->'
        raw = report.read_text()
        if marker not in raw:
            heading,body = raw.split('\n',1)
            note = f"\n\n{marker}\n**Contrôle essentiel :** sur les échanges A↔B, la suite recalculée et la suite native du receveur figée donnent toutes deux **32/32 solutions du receveur**. L'adaptation de la suite n'est donc pas nécessaire à la justesse finale dans ces cas. En C, les 128 perturbations conservent déjà les solutions avec suite figée. Voir le [complément de lecture et de format](COMPLEMENT_SORTIES.md), qui précise le statut post-traitement de ces comptes sans modifier les décisions gelées.\n"
            report.write_text(heading+note+body)
    check_freeze()
    print(result)


if __name__ == '__main__':
    main()

"""Counterfactual source families, generated only after the local freeze."""
import json
import random
from support import D, ROOT, Q, CELL, np, write, historical_data, symkey, backtracking, exact_cover_count, valid_grid


def exclusions():
    excluded = set()
    for path in historical_data():
        if path.suffix == '.json':
            rows = json.loads(path.read_text())['rows']
            solutions = [row['solution'] for row in rows]
        else:
            # Existing official plain-data files; use their restricted loader.
            from ncm_mac.sudoku import load_boards
            solutions = np.asarray(load_boards(path))[:, CELL]
        excluded.update(symkey(s) for s in solutions)
    return excluded


def sudoku(rng):
    excluded = exclusions()
    rows = []
    attempts = []
    for attempt in range(Q['max_sudoku_attempts']):
        solution = backtracking([0]*81, 1, rng)[0]
        if symkey(solution) in excluded:
            attempts.append(dict(attempt=attempt, accepted=False, reason='historical_solution_key'))
            continue
        base = solution.copy()
        order = list(range(81))
        rng.shuffle(order)
        found = None
        for j in order:
            old = base[j]
            base[j] = 0
            solutions = backtracking(base, 3)
            if len(solutions) > 2:
                base[j] = old
            elif len(solutions) == 2:
                found = solutions
                break
        if found is None:
            attempts.append(dict(attempt=attempt, accepted=False, reason='no_two_solution_base'))
            continue
        sa, sb = found
        if any(symkey(s) in excluded for s in found):
            attempts.append(dict(attempt=attempt, accepted=False, reason='historical_branch_key'))
            continue
        assert all(valid_grid(s, base) for s in found)
        assert exact_cover_count(base, 3) == 2
        differing = [j for j in range(81) if sa[j] != sb[j]]
        j = rng.choice(differing)
        pa, pb = base.copy(), base.copy()
        pa[j], pb[j] = sa[j], sb[j]
        eligible = [k for k in range(81) if pa[k] == pb[k] == 0]
        keep_j = rng.choice(eligible)
        plus = pa.copy()
        plus[keep_j] = sa[keep_j]
        for puzzle, target in ((pa, sa), (pb, sb), (plus, sa)):
            assert backtracking(puzzle, 3) == [target]
            assert exact_cover_count(puzzle, 3) == 1
        rows.append(dict(source_id=f'sudoku-{len(rows):03d}', family='sudoku', split='evaluation',
                         base=base, input_a=pa, input_b=pb, input_invariant=plus,
                         target_a=[x-1 for x in sa], target_b=[x-1 for x in sb], classes=9,
                         eligible=eligible, distinguishing_cell=j, redundant_cell=keep_j,
                         oracle_solutions=[sa, sb], solver_counts=dict(base=2, a=1, b=1, invariant=1),
                         clues=dict(base=sum(x != 0 for x in base), a=sum(x != 0 for x in pa),
                                    b=sum(x != 0 for x in pb), invariant=sum(x != 0 for x in plus)),
                         changed_answer_cells_excluding_clue=[k for k in differing if k != j]))
        for s in found:
            excluded.add(symkey(s))
        attempts.append(dict(attempt=attempt, accepted=True, source_id=rows[-1]['source_id']))
        print(json.dumps(dict(stage='sudoku_two_solutions', done=len(rows), total=Q['sudoku_families'], attempt=attempt)), flush=True)
        if len(rows) == Q['sudoku_families']:
            return rows, attempts
    raise RuntimeError('Fixed Sudoku attempt budget exhausted; do not silently reduce sample')


def analytic(rng):
    rows = []
    for family in ('equality', 'parity', 'permutation'):
        seen = set()
        for i in range(Q['analytic_per_family']):
            while True:
                if family == 'equality':
                    bits = [rng.randrange(2) for _ in range(6)]
                    sa = [v for bit in bits for v in (bit, bit)]
                    sb = sa.copy()
                    j = rng.randrange(6)*2
                    sb[j] = sb[j+1] = 1-sa[j]
                    classes = 2
                elif family == 'parity':
                    bits = [rng.randrange(2) for _ in range(7)]
                    sa = bits+[sum(bits) % 2]
                    sb = sa.copy()
                    for j in rng.sample(range(8), 2):
                        sb[j] = 1-sb[j]
                    classes = 2
                else:
                    sa = list(range(6))
                    rng.shuffle(sa)
                    sb = sa.copy()
                    j, k = rng.sample(range(6), 2)
                    sb[j], sb[k] = sb[k], sb[j]
                    classes = 6
                if tuple(sa) not in seen:
                    seen.add(tuple(sa))
                    break
            source_id = f'{family}-{i:03d}'
            rows.append(dict(source_id=source_id, family=family,
                             split='development' if family in Q['development_families'] else 'evaluation',
                             target_a=sa, target_b=sb, classes=classes, eligible=list(range(len(sa))),
                             input_a=dict(task=family, key=sa, redundant_tag=0),
                             input_b=dict(task=family, key=sb, redundant_tag=0),
                             input_invariant=dict(task=family, key=sa, redundant_tag=1)))
    return rows


def generate():
    rng = random.Random(Q['seed'])
    rows = analytic(rng)
    natural, attempts = sudoku(rng)
    rows.extend(natural)
    write(D/'data/families.json', dict(seed=Q['seed'], rows=rows, sudoku_attempts=attempts,
          exclusion='D4 plus digit normalization against local official and historical external solution corpora; incomplete Sudoku isomorphism audit'))
    return rows

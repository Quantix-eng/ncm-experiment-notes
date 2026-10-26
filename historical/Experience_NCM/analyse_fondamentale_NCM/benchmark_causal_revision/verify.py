"""Independent closed-form checks and two-solver data audit; same research author."""
import json
from support import D, Q, np, write, check_freeze, sha, backtracking, exact_cover_count, valid_grid, symkey
from methods import METHODS, project


def main():
    check_freeze()
    dataset = json.loads((D/'data/families.json').read_text())
    families = {row['source_id']: row for row in dataset['rows']}
    truth = json.loads((D/'data/evaluator_truth.json').read_text())
    records = json.loads((D/'results/observations.json').read_text())
    summary = json.loads((D/'results/summary.json').read_text())
    raw = np.load(D/'results/raw_vectors.npz')
    checks = dict(sudoku_families=0, analytic_families=0, closed_form_outputs=0, method_access_projections=0,
                  native_trajectory_equivalence_groups=0, crossed_nonidentifiability_groups=0)
    branch_keys = set()
    for source in families.values():
        if source['family'] != 'sudoku':
            sa, sb = source['target_a'], source['target_b']
            if source['family'] == 'equality':
                assert all(s[j] == s[j+1] for s in (sa,sb) for j in range(0,12,2))
            elif source['family'] == 'parity':
                assert all(sum(s)%2 == 0 for s in (sa,sb))
            else:
                assert sorted(sa) == sorted(sb) == list(range(6))
            assert sa != sb
            assert source['input_invariant']['key'] == sa
            checks['analytic_families'] += 1
            continue
        base = source['base']
        solutions = backtracking(base, 3)
        assert len(solutions) == 2 and exact_cover_count(base, 3) == 2
        assert solutions == source['oracle_solutions']
        assert all(valid_grid(s,base) for s in solutions)
        for puzzle, target in [(source['input_a'],solutions[0]), (source['input_b'],solutions[1]),
                               (source['input_invariant'],solutions[0])]:
            assert backtracking(puzzle, 3) == [target]
            assert exact_cover_count(puzzle, 3) == 1
        assert sum(a!=b for a,b in zip(source['input_a'],source['input_b'])) == 1
        j = source['distinguishing_cell']
        assert source['input_a'][j] != source['input_b'][j]
        assert sum(a!=b for a,b in zip(source['input_a'],source['input_invariant'])) == 1
        assert source['clues']['a'] == source['clues']['b']
        assert source['clues']['invariant'] == source['clues']['a']+1
        assert all(source['input_a'][c] == source['input_b'][c] == 0 for c in source['eligible'])
        assert len(source['changed_answer_cells_excluding_clue']) >= 1
        # Distinct source families cannot share an audited branch key.
        keys = {symkey(s) for s in solutions}
        assert not keys & branch_keys
        branch_keys.update(keys)
        checks['sudoku_families'] += 1
    grouped = {}
    finite_labels = set(Q['pure_controls']+['mixed'])
    predictions = {method:[] for method in METHODS}
    for record in records:
        cid = record['case_id']
        gold = truth[cid]
        source = families[gold['source_id']]
        grouped.setdefault(gold['source_id'],[]).append((record,gold))
        assert source['family'] == gold['family'] and source['split'] == gold['split']
        assert gold['label'] != 'mixed' or gold['split'] == 'evaluation'
        a = {key.split('__',1)[1]:raw[key] for key in raw.files if key.startswith(cid+'__')}
        sa = np.eye(source['classes'])[source['target_a']]
        sb = np.eye(source['classes'])[source['target_b']]
        assert np.array_equal(a['answer_a'],sa) and np.array_equal(a['answer_b'],sb)
        eps = Q['zero_tolerance']
        def check(value, expected):
            assert np.max(np.abs(value-expected)) < eps, (cid, gold['label'])
            checks['closed_form_outputs'] += 1
        v,d0,d1 = a['v'],a['direction0'],a['direction1']
        assert np.isclose(np.linalg.norm(d0),np.linalg.norm(d1)) and np.sum(d0*d1) == 0
        check(v,.1*d0)
        check(a['trajectory'],np.stack([sa+t*d0 for t in Q['reader_schedule']]))
        assert record['trajectory'] == a['trajectory'].argmax(-1).tolist()
        assert record['trajectory'][0] != record['trajectory'][1]
        assert record['trajectory'][1] == record['trajectory'][3]
        assert record['trajectory'][2] == record['trajectory'][0]
        passive = gold['label'] in ('oracle_replay','tolerant')
        for trial in record['crossed']:
            direction, sign = trial['direction'],trial['sign']
            delta = sign*(d0 if direction == 0 else d1)
            wrong = sign*(d1 if direction == 0 else d0)
            if passive:
                expected = {k:sa for k in ('Y00','Y10','Y11','Y01','Y1wrong')}
            elif gold['label'] == 'fixed_response':
                expected = dict(Y00=sa,Y10=sa+delta,Y11=sa+delta-d0,Y01=sa-d0,Y1wrong=sa+delta-d0)
            else:
                expected = dict(Y00=sa,Y10=sa+delta,Y11=sa,Y01=sa-delta,Y1wrong=sa+delta-wrong)
            for key, value in expected.items():
                observed = a[f'cross_{direction}_{sign}_{key}']
                check(observed,value)
                assert abs(float(np.square(observed-sa).mean())-trial['losses'][key]) < eps
            l = trial['losses']
            assert abs(trial['R']-(l['Y10']-l['Y11'])) < eps
            assert abs(trial['I']-(l['Y10']+l['Y01']-l['Y11']-l['Y00'])) < eps
            assert abs(trial['S']-(l['Y1wrong']-l['Y11'])) < eps
        for input_index, si in enumerate((sa,sb)):
            for state_index, sj in enumerate((sa,sb)):
                for tail in ('A','B','regenerated'):
                    expected = sj
                    if tail == 'regenerated' and gold['label'] == 'input_recompute': expected = si
                    if tail == 'regenerated' and gold['label'] == 'mixed': expected = gold['alpha']*sj+(1-gold['alpha'])*si
                    check(a[f'natural_{input_index}_{state_index}_{tail}'], expected)
        expected_input = sb if gold['label'] == 'input_recompute' else sa
        expected_state = sa if gold['label'] == 'input_recompute' else sb
        if gold['label'] == 'mixed':
            expected_input = gold['alpha']*sa+(1-gold['alpha'])*sb
            expected_state = gold['alpha']*sb+(1-gold['alpha'])*sa
        check(a['input_switch'],expected_input)
        check(a['state_switch'],expected_state)
        check(a['invariant'],sa)
        check(a['no_tail'],sa if gold['label']=='oracle_replay' else sa-v)
        for method,(regime,fn) in METHODS.items():
            public = project(record,regime)
            assert not {'case_id','label','alpha','source_id','continuation_calls'} & public.keys()
            prediction = fn(public)
            assert prediction == record['predictions'][method]
            assert set(prediction) <= finite_labels and len(set(prediction)) == len(prediction)
            predictions[method].append((prediction,gold))
            checks['method_access_projections'] += 1
    examples = []
    for source_id, group in grouped.items():
        trajectories = [json.dumps(row['trajectory']) for row,_ in group]
        assert len(set(trajectories)) == 1
        checks['native_trajectory_equivalence_groups'] += 1
        adaptive = [(r,g) for r,g in group if g['label'] in ('state_compensation','input_recompute','mixed')]
        first = adaptive[0][0]['crossed']
        for row,_ in adaptive[1:]:
            for t,u in zip(first,row['crossed']):
                assert np.allclose([t[k] for k in ('R','I','S')],[u[k] for k in ('R','I','S')],atol=1e-14,rtol=0)
        checks['crossed_nonidentifiability_groups'] += 1
        if not examples:
            examples.append(dict(source_id=source_id, case_ids=[r['case_id'] for r,_ in adaptive],
                                 conclusion='Crossed same-input observations cannot distinguish state compensation, input recompute and the announced mixtures.'))
    # Recompute score counts without the runner's scorer.
    for method, items in predictions.items():
        for key, s in summary['scores'][method].items():
            if key == 'macro_exact_by_control': continue
            chosen = items
            if ':' in key:
                field,value = key.split(':',1)
                chosen = [(p,g) for p,g in items if g[field] == value]
            assert s['n'] == len(chosen)
            assert s['covered'] == sum(g['label'] in p for p,g in chosen)
            assert s['singleton'] == sum(len(p)==1 for p,g in chosen)
            assert s['singleton_correct'] == sum(p == [g['label']] for p,g in chosen)
            assert np.isclose(s['mean_set_size'],np.mean([len(p) for p,g in chosen]))
    assert len(families) == 3*Q['analytic_per_family']+Q['sudoku_families']
    assert checks['sudoku_families'] == Q['sudoku_families']
    assert summary['scores']['complete_intervention']['all']['exact_identification'] == 1
    assert summary['scores']['crossed_only']['all']['coverage'] == 1
    check_freeze()
    write(D/'results/nonidentifiability_examples.json',examples)
    write(D/'results/verification.json',dict(status='passed',checks=checks,
          sources=len(families),cases=len(records),natural_conditions_per_case=12,
          sudoku_solution_counts='Two independent solvers with cap=3: base 2, both branches and redundant-clue control 1',
          summary_sha256=sha(D/'results/summary.json'),raw_vectors_sha256=sha(D/'results/raw_vectors.npz'),
          limitation='Same research author; equations and software audited, not an independent external evaluation.'))
    print(json.dumps(dict(status='passed',checks=checks)),flush=True)


if __name__ == '__main__':
    main()

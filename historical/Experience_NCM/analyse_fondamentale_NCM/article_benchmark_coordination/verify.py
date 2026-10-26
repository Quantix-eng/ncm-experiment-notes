"""Publication audit: retained inputs, numerical claims and local Markdown links.

No model calls, fitting or threshold selection. Recomputations do not call the
auxiliary entropy or mutual-information functions. Not an external replication.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import unquote

import numpy as np

D = Path(__file__).resolve().parent
ROOT = D.parent
HASHES = {}


def digest(path):
    path = path.resolve()
    if path not in HASHES:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1048576), b''):
                h.update(block)
        HASHES[path] = h.hexdigest()
    return HASHES[path]


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def main():
    historical = {}
    for name in ('composition_terminale', 'benchmark_causal_revision', 'benchmark_ncm_reel'):
        base = ROOT/name
        final = base/'results/final_manifest.json'
        manifest = read(final)
        digest(final)
        for relative, expected in manifest['files'].items():
            assert digest(base/relative) == expected, (name, relative)
        frozen = read(base/'results/protocol_frozen.json')
        for relative, expected in frozen['files'].items():
            assert digest(ROOT/relative) == expected, ('historical freeze', name, relative)
        historical[name] = dict(final_manifest_files=len(manifest['files']),
                                original_frozen_files=len(frozen['files']), unchanged=True)
    frozen = read(D/'results/auxiliary_frozen.json')
    digest(D/'results/auxiliary_frozen.json')
    for relative, expected in frozen['files'].items():
        assert digest(ROOT/relative) == expected, ('auxiliary freeze', relative)
    aux = read(D/'results/auxiliary.json')
    digest(D/'results/auxiliary.json')
    with np.load(ROOT/'benchmark_ncm_reel/results/native.npz') as archive:
        codes = archive['codes'][:, :2]
        offsets = archive['offsets']
        levels = archive['levels']
    level_counts = []
    for k in range(12):
        block = codes[:, :, int(offsets[k]):int(offsets[k+1])]
        changes = np.not_equal(block[:, 0], block[:, 1])
        counts = np.count_nonzero(changes, axis=1)
        n = int(np.count_nonzero(counts))
        level_counts.append(n)
        expected = aux['levels'][k]
        assert expected['positions'] == len(block[0, 0]) == int(levels[k])
        assert n == expected['distinguishable_families']
        assert expected['average_branch_MI_bits'] == n/16
        assert expected['observed_code_ids'] == np.unique(block).tolist()
        assert math.isclose(expected['mean_position_MI_bits'], float(changes.mean()), abs_tol=1e-15)
        for f in range(16):
            row = aux['per_family'][f]['levels'][k]
            assert row['whole_vector_MI_bits'] == int(counts[f] > 0)
            assert math.isclose(row['mean_position_MI_bits'], counts[f]/int(levels[k]), abs_tol=1e-15)
    assert level_counts == [6, 1, 5, 6, 0, 11, 11, 16, 16, 16, 16, 16]
    split = int(offsets[9])
    coarse = np.any(codes[:, 0, :split] != codes[:, 1, :split], axis=1)
    fine = np.any(codes[:, 0, split:] != codes[:, 1, split:], axis=1)
    assert coarse.all() and fine.all()
    for f in range(16):
        row = aux['per_family'][f]
        assert row['branch_coarse_MI_bits'] == row['branch_fine_MI_bits'] == row['coarse_fine_MI_bits'] == 1
    assert aux['mean_coarse_branch_MI_bits'] == aux['mean_fine_branch_MI_bits'] == aux['mean_coarse_fine_MI_given_family_bits'] == 1

    neural = read(ROOT/'benchmark_ncm_reel/results/summary.json')
    calibration = read(ROOT/'benchmark_causal_revision/results/summary.json')
    quality = read(ROOT/'benchmark_ncm_reel/results/output_quality.json')
    assert calibration['n_families'] == 52 and calibration['n_cases'] == 344
    scores = calibration['scores']
    assert scores['trajectory_naive']['all']['singleton_correct'] == 52
    assert scores['trajectory_cautious']['all']['covered'] == 344
    assert scores['trajectory_cautious']['all']['singleton'] == 0
    assert scores['crossed_only']['all']['covered'] == 344
    assert scores['crossed_only']['all']['singleton_correct'] == 52
    assert math.isclose(scores['crossed_only']['all']['mean_set_size'], 824/344)
    assert scores['complete_intervention']['all']['singleton_correct'] == 344
    assert scores['complete_intervention']['split:evaluation']['singleton_correct'] == 224
    assert sum(row['exact'] for row in quality['O'].values()) == 48
    for row in quality['C'].values():
        assert row['n'] == row['exact'] == 128
        assert row['violated_clues'] == row['bad_separators'] == 0
    assert neural['C']['margin_nats'] == .01
    assert neural['C']['decision'] == 'mean_frozen_damage_equivalent_within_prespecified_margin'
    for key in ('D', 'R', 'I', 'S'):
        assert all(abs(bound) < .01 for bound in neural['C'][key]['ci_adjusted'])
    assert abs(neural['C']['D']['mean']-1.7471847246302982e-6) < 1e-18
    assert abs(neural['E']['input_effect']['mean']-29.467573208642055) < 1e-12
    assert abs(neural['E']['state_effect']['mean']-3.179794898332947) < 1e-12
    sensitivity = []
    for threshold, expected in zip((.7, .8, .9, .95), (15, 15, 8, 4)):
        selected = []
        eligible_count = 0
        for row in neural['E']['per_family']:
            ratio = row['input_ratio']
            eligible = (row['native_exact_A'] and row['native_exact_B'] and
                        row['native_span'] > .1 and abs(row['interaction_ratio']) <= .2 and
                        ratio is not None and 0 <= ratio <= 1)
            eligible_count += int(eligible)
            if eligible and ratio >= threshold:
                selected.append(row['source_id'])
            assert math.isclose(row['input_effect']+row['state_effect'], row['native_span'], abs_tol=1e-12)
        assert len(selected) == expected and eligible_count == 15
        sensitivity.append(dict(threshold=threshold, eligible=eligible_count,
                                input_dominant=len(selected), total_families=16))
    assert sensitivity == aux['threshold_sensitivity']
    labels = neural['E']['label_counts']
    assert labels == dict(input_dominant_response_preference_in_this_test=15, non_monotone_non_identifiable=1)
    exception = [r for r in neural['E']['per_family'] if r['source_id'] == 'sudoku-012'][0]
    assert exception['input_ratio'] > 1
    assert exception['functional_label'] == 'non_monotone_non_identifiable'
    exchange = quality['E']['answer_change']
    for policy in ('regenerated', 'frozen_recipient_native'):
        assert exchange[policy]['recipient']['exact'] == 32
        assert exchange[policy]['donor']['exact'] == 0
    assert exchange['frozen_donor_native']['recipient']['exact'] == 0
    assert exchange['frozen_donor_native']['donor']['exact'] == 32
    assert exchange['frozen_donor_native']['recipient']['violated_clues'] == 32
    for policy in ('regenerated', 'frozen_recipient_native', 'frozen_donor_native'):
        assert quality['E']['answer_preserve'][policy]['recipient']['exact'] == 32

    for n in range(2, 11):
        # Generate from independent seeds and a global decoder, rather than
        # filtering all n-bit vectors as in auxiliary.py.
        seeds = list(itertools.product((0, 1), repeat=n-1))
        outputs = [seed+(sum(seed) % 2,) for seed in seeds]
        assert len(set(outputs)) == 2**(n-1)
        assert all(sum(row) % 2 == 0 for row in outputs)
        result = aux['parity_checks'][n-2]
        assert result['n'] == n and result['joint_entropy_bits'] == math.log2(len(outputs))
        assert result['total_correlation_bits'] == 1
        assert result['safe_product_supports'] == len(outputs)
        if n >= 3:
            for i in range(n):
                for j in range(i):
                    frequencies = [sum(row[i] == a and row[j] == b for row in outputs)
                                   for a, b in itertools.product((0, 1), repeat=2)]
                    assert len(set(frequencies)) == 1
            assert result['max_pairwise_MI_bits'] == 0
    sources = read(ROOT/'benchmark_causal_revision/data/families.json')['rows']
    sudoku = [row for row in sources if row['family'] == 'sudoku']
    assert len(sudoku) == 16
    for source, target in zip(sudoku, aux['two_solution_target_checks']):
        left, right = source['oracle_solutions']
        different = [j for j, (a, b) in enumerate(zip(left, right)) if a != b]
        assert len(different) >= 2 and len(different) == target['differing_cells']
        assert source['source_id'] == target['source_id']
        # Each of two changed coordinates recovers the balanced branch exactly.
        for j in different[:2]:
            assert len({left[j], right[j]}) == 2
        assert target['hypothetical_uniform_two_solution_Wyner_bits'] == 1
        assert target['unique_branch_conditional_Wyner_bits'] == 0

    checked_links = 0
    deferred_links = []
    for document in sorted(D.glob('*.md')):
        for target in re.findall(r'\]\(([^)]+)\)', document.read_text()):
            if target.startswith(('https://', 'http://', '#', 'mailto:')):
                continue
            target = unquote(target.split('#', 1)[0].strip('<>'))
            destination = (document.parent/target).resolve()
            if destination == (D/'results/verification.json').resolve():
                deferred_links.append(destination)
            else:
                assert destination.exists(), (document.name, target)
            checked_links += 1
    # Include theoretical context and executed adapter in the retained-input ledger.
    for relative in ('correction_pooling/RAPPORT.md', 'manuscrit_unifie/MANUSCRIPT.md',
                     'vendor/ncm_mac/cached.py', 'benchmark_ncm_reel/runtime.py'):
        digest(ROOT/relative)
    inputs = {str(path.relative_to(ROOT)): value for path, value in sorted(HASHES.items())}
    save(D/'results/input_manifest.json', dict(scope='Retained scientific inputs; post-hoc publication consolidation',
                                             files=inputs))
    result = dict(status='passed', utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  scope='Internal software audit; no model inference or independent empirical replication',
                  historical_integrity=historical, auxiliary_frozen_files=len(frozen['files']),
                  input_files_hashed=len(inputs), local_markdown_links_checked=checked_links,
                  level_distinguishability_counts=level_counts, threshold_sensitivity=sensitivity,
                  parity_global_decoder_checks=9, two_solution_target_checks=16,
                  principal_calibration_and_neural_counts_checked=True,
                  mathematical_scope='Finite checks support the written proofs; no Wyner rate estimated for NCM')
    destination = D/'results/verification.json'
    if destination.exists():
        previous = read(destination)
        if {k: v for k, v in previous.items() if k != 'utc'} == {k: v for k, v in result.items() if k != 'utc'}:
            result['utc'] = previous['utc']
    save(destination, result)
    assert all(path.exists() for path in deferred_links)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

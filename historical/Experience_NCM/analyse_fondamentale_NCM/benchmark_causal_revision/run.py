"""Frozen calibration runner. Failures are retained; existing outputs not replaced."""
import hashlib
import json
import platform
import sys
import time
from support import D, Q, np, write, freeze, check_freeze, sha
from dataset import generate
from controls import observe, LABELS
from methods import METHODS, project


def score(records, truth):
    result = {}
    for method in METHODS:
        slices = [('all', records)]
        for key in ('split', 'family', 'label'):
            for value in sorted({row[key] for row in truth.values()}):
                selected = [row for row in records if truth[row['case_id']][key] == value]
                slices.append((f'{key}:{value}', selected))
        details = {}
        for name, rows in slices:
            n = len(rows)
            if not n:
                continue
            predictions = [row['predictions'][method] for row in rows]
            labels = [truth[row['case_id']]['label'] for row in rows]
            singleton = [len(p) == 1 for p in predictions]
            correct = [len(p) == 1 and p[0] == label for p, label in zip(predictions, labels)]
            covered = [label in p for p, label in zip(predictions, labels)]
            details[name] = dict(n=n, covered=sum(covered), coverage=sum(covered)/n,
                                 singleton=sum(singleton), singleton_correct=sum(correct),
                                 exact_identification=sum(correct)/n,
                                 singleton_accuracy=sum(correct)/sum(singleton) if any(singleton) else None,
                                 mean_set_size=sum(map(len, predictions))/n,
                                 abstention_fraction=1-sum(singleton)/n)
        details['macro_exact_by_control'] = float(np.mean([details[f'label:{label}']['exact_identification']
                                                        for label in LABELS if f'label:{label}' in details]))
        result[method] = details
    return result


def main():
    if sys.argv[1:] == ['--freeze']:
        freeze()
        print('Local protocol/code/input freeze written.')
        return
    assert not sys.argv[1:], 'Usage: run.py [--freeze]'
    frozen = check_freeze()
    assert not (D/'data/families.json').exists(), 'Use a new sibling directory for reproduction'
    start = time.monotonic()
    families = generate()
    records, truth, arrays = [], {}, {}
    for source_index, source in enumerate(families):
        models = [(name, None) for name in Q['pure_controls']]
        if source['split'] == 'evaluation':
            models += [('mixed', alpha) for alpha in Q['mixture_alphas']]
        for name, alpha in models:
            case_id = hashlib.sha256(f"{Q['seed']}:{source['source_id']}:{name}:{alpha}".encode()).hexdigest()[:20]
            rng = np.random.default_rng(Q['seed']+1009*source_index)
            record, vectors = observe(source, name, alpha, rng)
            truth[case_id] = dict(source_id=source['source_id'], family=source['family'], split=source['split'], label=name, alpha=alpha)
            record['case_id'] = case_id
            record['predictions'] = {method: function(project(record, regime)) for method, (regime, function) in METHODS.items()}
            records.append(record)
            for key, vector in vectors.items():
                arrays[f'{case_id}__{key}'] = vector
        print(json.dumps(dict(stage='controls', done=source_index+1, total=len(families))), flush=True)
    # Class order must not leak through record order.
    np.random.default_rng(Q['seed']).shuffle(records)
    write(D/'data/evaluator_truth.json', truth)
    write(D/'results/observations.json', records)
    np.savez_compressed(D/'results/raw_vectors.npz', **arrays)
    summary = dict(status='calibration_executed_not_independently_validated', n_families=len(families),
                   n_cases=len(records), scores=score(records, truth),
                   partitions={name:sum(row['split'] == name for row in families) for name in ('development', 'evaluation')},
                   counts_by_family={name:sum(row['family'] == name for row in families) for name in ('equality', 'parity', 'permutation', 'sudoku')},
                   utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   elapsed_seconds=time.monotonic()-start,
                   environment=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform(), dtype='float64', inference='constructed controls; no learned model'),
                   freeze_sha256=sha(D/'results/protocol_frozen.json'), freeze_utc=frozen['utc'])
    write(D/'results/summary.json', summary)
    check_freeze()
    print(json.dumps({k:v for k,v in summary.items() if k!='scores'}), flush=True)


if __name__ == '__main__':
    main()

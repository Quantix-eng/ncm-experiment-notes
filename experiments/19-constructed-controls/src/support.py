"""Paths, local freeze and immutable inputs; no writes to previous campaigns."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
ROOT = D.parent
os.environ['MPLCONFIGDIR'] = str(D / '.cache/matplotlib')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(ROOT / 'scripts'))
from common import np, UNITS, CELL, write
from solvers import backtracking, exact_cover_count

Q = json.loads((D / 'protocol.json').read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def historical_data():
    return ([ROOT / 'data/external.json', ROOT / 'compensation_conditionnelle/data/fresh.json',
             ROOT / 'recherche_positive/data/fresh.json'] +
            [ROOT / f'data/sudoku/{task}/solutions.pickle' for task in ('easy', 'medium', 'hard')])


def freeze():
    path = D / 'results/protocol_frozen.json'
    assert not path.exists(), 'Refusing to replace freeze'
    files = (list(D.glob('*.py')) + [D/'PROTOCOLE.md', D/'protocol.json'] + historical_data()
             + [ROOT/'scripts/common.py', ROOT/'scripts/solvers.py', ROOT/'protocol.json']
             + list((ROOT/'vendor/ncm_mac').rglob('*.py'))
             + list(ROOT.glob('**/results/final_manifest.json'))
             + list(ROOT.glob('**/RAPPORT.md'))
             + [ROOT/'manuscrit_unifie/MANUSCRIPT.md'])
    files = sorted(set(p for p in files if p.is_file() and D not in p.parents)) + sorted(D.glob('*.py')) + [D/'PROTOCOLE.md', D/'protocol.json']
    write(path, dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                     files={str(p.relative_to(ROOT)): sha(p) for p in files},
                     status='local freeze before corpus generation; not independent preregistration'))


def check_freeze():
    frozen = json.loads((D/'results/protocol_frozen.json').read_text())
    for name, digest in frozen['files'].items():
        assert sha(ROOT/name) == digest, name
    return frozen


def symkey(solution):
    a = np.asarray(solution).reshape(9, 9)
    variants = []
    for flip in (False, True):
        base = a[:, ::-1] if flip else a
        for k in range(4):
            mapping = {}
            variants.append(tuple(mapping.setdefault(int(x), len(mapping)+1) for x in np.rot90(base, k).ravel()))
    return min(variants)


def valid_grid(solution, puzzle):
    return (len(solution) == 81 and all(sorted(solution[i] for i in u) == list(range(1, 10)) for u in UNITS)
            and all(p == 0 or p == s for p, s in zip(puzzle, solution)))

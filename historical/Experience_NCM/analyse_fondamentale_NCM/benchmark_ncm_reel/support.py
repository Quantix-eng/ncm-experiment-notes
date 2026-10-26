"""Study-local utilities; older campaigns are read only."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
ROOT = D.parent
os.environ['MPLCONFIGDIR'] = str(D/'.cache/matplotlib')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0,str(ROOT/'scripts'))
from common import np, torch, CELL, SEP, SudokuModel, tokens, write, evaluate
Q = json.loads((D/'protocol.json').read_text())
DATA = ROOT/'benchmark_causal_revision/data/families.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def freeze():
    path = D/'results/protocol_frozen.json'
    assert not path.exists(), 'Refusing to replace freeze'
    files = (list(D.glob('*.py'))+[D/'PROTOCOLE.md',D/'protocol.json',DATA,ROOT/'protocol.json']
             +list((ROOT/'vendor/ncm_mac').rglob('*.py'))
             +list((ROOT/'artifacts/sudoku/hard').glob('*'))
             +[ROOT/'scripts/common.py',ROOT/'scripts/solvers.py']
             +list(ROOT.glob('**/results/final_manifest.json'))+list(ROOT.glob('**/RAPPORT.md'))
             +[ROOT/'manuscrit_unifie/MANUSCRIPT.md'])
    files = sorted(set(p for p in files if p.is_file() and (D not in p.parents or p.parent == D)))
    write(path,dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                   files={str(p.relative_to(ROOT)):sha(p) for p in files},
                   status='local freeze before any new NCM inference; not independent preregistration'))


def check_freeze():
    f = json.loads((D/'results/protocol_frozen.json').read_text())
    for name,digest in f['files'].items():
        assert sha(ROOT/name) == digest, name
    return f


def data():
    rows = [r for r in json.loads(DATA.read_text())['rows'] if r['family']=='sudoku']
    assert len(rows) == Q['families']
    puzzles = np.array([[r[k] for k in ('input_a','input_b','input_invariant')] for r in rows])
    refs = np.array([[[d+1 for d in r[k]] for k in ('target_a','target_b','target_a')] for r in rows])
    masks = np.all(puzzles == 0,axis=1)
    different = masks & (refs[:,0]!=refs[:,1])
    assert np.all(masks.sum(1)>1) and np.all(different.sum(1)>0)
    prefix = tokens(puzzles.reshape(-1,81),True).reshape(len(rows),3,91)
    reference = tokens(refs.reshape(-1,81)).reshape(len(rows),3,89)
    return rows,prefix,reference,masks,different


def emit(stage,**values):
    print(json.dumps(dict(stage=stage,**values)),flush=True)


def logprob(logits):
    v = np.asarray(logits,dtype=np.float64)
    v = v-v.max(-1,keepdims=True)
    return v-np.log(np.exp(v).sum(-1,keepdims=True))


def nll(logits,reference,mask):
    lp = logprob(logits)[...,CELL,:]
    target = np.asarray(reference)[CELL]
    selected = np.take_along_axis(lp,target.reshape((1,)*(lp.ndim-2)+(81,1)),axis=-1)[...,0]
    return -selected[...,mask].mean(-1)


def preference(logits,reference_a,reference_b,mask):
    return nll(logits,reference_a,mask)-nll(logits,reference_b,mask)

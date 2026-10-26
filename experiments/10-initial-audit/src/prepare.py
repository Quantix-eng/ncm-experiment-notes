from common import *
from collections import Counter

def symkey(c):
    vals=[]; a=np.array(c).reshape(9,9)
    for flip in [False,True]:
        for k in range(4):
            v=np.rot90(a[:,::-1] if flip else a,k).ravel(); mapping={0:0}; norm=[]
            for d in v:
                d=int(d)
                if d not in mapping: mapping[d]=len(mapping)
                norm.append(mapping[d])
            vals.append(bytes(norm))
    return hashlib.sha256(min(vals)).hexdigest()

if __name__=='__main__':
    rng=np.random.default_rng(P['seed']); audit={}; groupkeys={}
    for d in ['easy','medium','hard']:
        p,s=boards(d); keys=[symkey(c) for c in p[:,1:-1][:,CELL]]; groupkeys[d]=keys
        audit[d]=dict(n=len(p),unique_exact_puzzles=len(set(map(bytes,p))),unique_known_symmetry_puzzles=len(set(keys)),
                      unique_exact_solutions=len(set(map(bytes,s))),unique_known_symmetry_solutions=len(set(symkey(c) for c in s[:,CELL])))
    keys=groupkeys['hard']; groups={k:np.flatnonzero(np.array(keys)==k).tolist() for k in sorted(set(keys))}
    shuffled=list(groups); rng.shuffle(shuffled); splits=[[],[],[]]
    for key in shuffled:
        dest=0 if len(splits[0])<1000 else (1 if len(splits[1])<400 else 2)
        splits[dest].extend(groups[key])
    names=['train','validation','test']; split=dict(zip(names,splits))
    assert sum(map(len,splits))==2000
    for i in range(3):
        for j in range(i): assert not (set(keys[k] for k in splits[i]) & set(keys[k] for k in splits[j]))
    cross={d:np.random.default_rng(P['seed']).permutation(2000)[:P['cross_n']].tolist() for d in groupkeys}
    write(ROOT/'results/splits.json',dict(**split,cross=cross,grouping='puzzle under D4 and digit first-occurrence normalization; not exhaustive Sudoku isomorphism',audit=audit))
    manifest={str(f.relative_to(ROOT)):sha(f) for sub in ['artifacts','data','sources','vendor/ncm_mac'] for f in (ROOT/sub).rglob('*') if f.is_file() and '__pycache__' not in str(f)}
    write(ROOT/'results/input_manifest.json',manifest)
    write(ROOT/'results/protocol_frozen.json',dict(protocol=P,todo_sha256=sha(ROOT/'sources/TODO_protocol_v1.md'),protocol_sha256=sha(ROOT/'protocol.json'),
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),splits_sha256=sha(ROOT/'results/splits.json')))
    write(ROOT/'results/environment.json',dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,
        threads=torch.get_num_threads(),deterministic=torch.are_deterministic_algorithms_enabled(),device='CPU',dtype='float32'))
    progress('prepared',audit=audit,sizes=list(map(len,splits)))

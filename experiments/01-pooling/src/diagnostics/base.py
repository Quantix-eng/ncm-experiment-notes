"""Study-local paths, immutable previous inputs and deterministic utilities."""
import os,sys,json,time,hashlib
from pathlib import Path
D=Path(__file__).resolve().parents[1]; ROOT=D.parent
os.environ['MPLCONFIGDIR']=str(D/'.cache/matplotlib')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(ROOT/'scripts'))
from common import CELL,SEP,UNITS,PEERS,boards,tokens,candidate_counts,SudokuModel,torch,np,write,sha,ci_rate,evaluate
Q=json.loads((D/'protocol.json').read_text());torch.manual_seed(Q['seed'])
SPLIT=json.loads((ROOT/'results/splits.json').read_text())

def log(stage,**kw):
    obj=dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    write(D/'results/progress.json',obj);print(json.dumps(obj,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v)),flush=True)

def freeze():
    write(D/'results/protocol_frozen.json',dict(protocol=Q,protocol_sha256=sha(D/'protocol.json'),objectives_sha256=sha(D/'OBJECTIFS_ET_PROTOCOLE.md'),
       previous_manifest_sha256=sha(ROOT/'results/final_manifest.json'),split_sha256=sha(ROOT/'results/splits.json'),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))

def external_boards():
    ext=json.loads((ROOT/'data/external.json').read_text())['rows']
    return tokens(np.array([r['puzzle'] for r in ext]),True),tokens(np.array([r['solution'] for r in ext]))

def softmax(s,temp=1):
    s=np.asarray(s,dtype=float)/temp;s-=s.max(-1,keepdims=True);v=np.exp(s);return v/v.sum(-1,keepdims=True)

def paired_mean_ci(values):
    v=np.asarray(values,dtype=float);rng=np.random.default_rng(Q['seed']);b=v[rng.integers(0,len(v),(Q['bootstrap'],len(v)))].mean(1)
    return dict(mean=float(v.mean()),ci95=np.quantile(b,[.025,.975]),n_puzzles=len(v),method='paired percentile bootstrap by puzzle, fixed models')

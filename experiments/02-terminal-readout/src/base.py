"""Local study paths and utilities; previous campaigns are read-only."""
import os,sys,json,time,hashlib
from pathlib import Path
D=Path(__file__).resolve().parents[1];ROOT=D.parent
os.environ['MPLCONFIGDIR']=str(D/'.cache/matplotlib')
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
sys.path.insert(0,str(ROOT/'scripts'))
from common import CELL,SEP,UNITS,PEERS,boards,tokens,SudokuModel,torch,np,write,evaluate,ci_rate
sys.path.insert(0,str(D/'scripts'))
Q=json.loads((D/'protocol.json').read_text()) if (D/'protocol.json').exists() else dict(seed=202610246,bootstrap=4000)
SPLIT=json.loads((ROOT/'results/splits.json').read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def log(stage,**kw):
    v=dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw);write(D/'results/progress.json',v);print(json.dumps(v,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()),flush=True)
U=np.array(UNITS)
def lookup_board(codes,mapping):
    out=np.full((len(codes),89),10,dtype=np.int64);out[:,CELL]=np.asarray(mapping)[codes[:,-89:][:,CELL]];return out
def certificate(ids,prefix):
    v=np.asarray(ids)[:,CELL];giv=np.asarray(prefix)[:,1:-1][:,CELL]
    return np.all(np.sort(v[:,U],axis=-1)==np.arange(1,10),(1,2))&np.all((giv==0)|(v==giv),1)&np.all(ids[:,SEP]==10,1)
def bootstrap(v):
    v=np.asarray(v,dtype=float);rng=np.random.default_rng(Q['seed']);b=v[rng.integers(0,len(v),(Q['bootstrap'],len(v)))].mean(1);alpha=.05/2
    return dict(mean=float(v.mean()),ci95=np.quantile(b,[.025,.975]),ci_familywise=np.quantile(b,[alpha/2,1-alpha/2]),n=len(v))
def ratio_boot(n,d):
    n=np.asarray(n,dtype=float);d=np.asarray(d,dtype=float);rng=np.random.default_rng(Q['seed']);ix=rng.integers(0,len(n),(Q['bootstrap'],len(n)));den=d[ix].sum(1);b=n[ix].sum(1)[den>0]/den[den>0];alpha=.05/2
    return dict(mean=float(n.sum()/d.sum()),ci95=np.quantile(b,[.025,.975]),ci_familywise=np.quantile(b,[alpha/2,1-alpha/2]),n=len(n),numerator=int(n.sum()),denominator=int(d.sum()))

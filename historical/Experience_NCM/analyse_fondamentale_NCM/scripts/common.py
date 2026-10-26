"""Shared invariants: fixed token positions, gold-free features, CPU-only execution."""
import os, sys, json, hashlib, platform, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('PYTHONPYCACHEPREFIX', str(ROOT/'.cache/pycache'))
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0, str(ROOT/'vendor'))
import numpy as np
import torch
from ncm_mac.sudoku import SudokuModel, load_boards
P = json.loads((ROOT/'protocol.json').read_text())
torch.set_num_threads(P['threads'])
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.manual_seed(P['seed'])
CELL = np.array([i for i in range(89) if i%10 != 9])
SEP = np.array([i for i in range(89) if i%10 == 9])
UNITS = ([list(range(r*9,r*9+9)) for r in range(9)] + [list(range(c,81,9)) for c in range(9)] +
 [[(br+r)*9+bc+c for r in range(3) for c in range(3)] for br in (0,3,6) for bc in (0,3,6)])
PEERS = [set().union(*(set(u) for u in UNITS if i in u))-{i} for i in range(81)]

def write(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    def convert(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        if isinstance(x,Path): return str(x)
        raise TypeError(type(x).__name__)
    raw=json.dumps(obj,ensure_ascii=False,indent=2,default=convert,allow_nan=False)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp'); tmp.write_text(raw); tmp.replace(path)

def progress(stage, **kwargs):
    obj=dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kwargs)
    write(ROOT/'results/progress.json',obj); print(json.dumps(obj,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v)),flush=True)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def boards(d):
    p=np.array(load_boards(ROOT/f'data/sudoku/{d}/puzzles.pickle'),dtype=np.int64)
    s=np.array(load_boards(ROOT/f'data/sudoku/{d}/solutions.pickle'),dtype=np.int64)
    assert p.shape==(2000,91) and s.shape==(2000,89)
    assert np.all(p[:,[0,-1]]==11) and np.all(p[:,1:-1][:,SEP]==10)
    assert np.all(s[:,SEP]==10) and np.all((s[:,CELL]>=1)&(s[:,CELL]<=9))
    for u in UNITS: assert np.all(np.sort(s[:,CELL][:,u],axis=1)==np.arange(1,10))
    given=p[:,1:-1][:,CELL]
    assert np.all((given==0)|(given==s[:,CELL]))
    return p,s

def tokens(cells,prompt=False):
    x=np.full((len(cells),89),10,dtype=np.int64); x[:,CELL]=cells
    if prompt: x=np.pad(x,((0,0),(1,1)),constant_values=11)
    return x

def candidate_counts(cells):
    return np.array([0 if v else len(set(range(1,10))-{cells[j] for j in PEERS[i]}) for i,v in enumerate(cells)])

def difficulty(prefix):
    out=[]
    for c in np.asarray(prefix)[:,1:-1][:,CELL]:
        cnt=candidate_counts(c); blank=c==0; work=c.copy(); rounds=0
        while True:
            cc=candidate_counts(work); singles=np.flatnonzero((work==0)&(cc==1))
            if not len(singles): break
            for i in singles:
                choices=set(range(1,10))-{work[j] for j in PEERS[i]}
                if len(choices)==1: work[i]=next(iter(choices))
            rounds+=1
            if rounds>81: raise AssertionError('Propagation loop')
        out.append([blank.sum()/81,cnt[blank].mean(),cnt[blank].std(),(cnt==1).sum()/max(1,blank.sum()),
                    ((c==0)&(work!=0)).sum()/max(1,blank.sum()),rounds/81])
    return np.array(out,dtype=np.float32)

def observed(ids,given):
    c=ids[:,CELL]; invalid=((c<1)|(c>9)).sum(1); badsep=(ids[:,SEP]!=10).sum(1)
    clue=((given!=0)&(c!=given)).sum(1)
    conflicts=np.zeros(len(c))
    for u in UNITS:
        for d in range(1,10):
            n=(c[:,u]==d).sum(1); conflicts+=n*(n-1)/2
    valid=(invalid+badsep+clue+conflicts)==0
    return np.column_stack([invalid/81,badsep/8,clue/81,conflicts/972,valid])

def evaluate(ids,ref,prefix):
    o=observed(ids,prefix[:,1:-1][:,CELL]); y=np.all(ids==ref,axis=1)
    return dict(n=len(ids),solved=int(y.sum()),solved_rate=float(y.mean()),
      cell_accuracy=float((ids[:,CELL]==ref[:,CELL]).mean()),valid_and_clues_rate=float(o[:,-1].mean()),
      malformed=int(np.any((ids[:,CELL]<1)|(ids[:,CELL]>9),axis=1).sum()))

@torch.inference_mode()
def trace(model,prefix,local=False):
    """Features extracted without reference solutions. Hook output is pre-ln_f."""
    hidden=[]; heads=[]; handles=[]
    def hh(mod,args,out): hidden.append(out.detach().clone())
    def lh(mod,args,out): heads.append(out.detach().clone())
    try:
        handles.append(model.net.transformer.blocks[-1].register_forward_hook(hh))
        for head in model.net.lm_heads: handles.append(head.register_forward_hook(lh))
        pred=model.predict(torch.tensor(prefix,dtype=torch.long))
    finally:
        for h in handles: h.remove()
    assert len(hidden)==13 and len(heads)==12
    assert not any(b.attn.caching or b.attn.cached_k is not None for b in model.net.transformer.blocks)
    hs=hidden[1:]; snapshots,=model.vqvae.vq.idx_to_zhats(pred.codes,return_zlevels=False)
    assert len({z.data_ptr() for z in snapshots})==12
    assert torch.equal(snapshots[-1],pred.reconstruction)
    running=torch.zeros_like(snapshots[-1]); offset=0
    for k,length in enumerate(model.net.levels):
        emb=model.vqvae.vq.embeddings[k](pred.codes[:,offset:offset+length])
        if k<11: emb=model.vqvae.vq._up_to_full(emb,k,89)
        running.add_(model.vqvae.vq._apply_quant_resi(emb,k))
        assert torch.equal(running,snapshots[k])
        assert torch.equal(heads[k].argmax(-1),pred.codes[:,offset:offset+length])
        offset+=length
    decoded=[]; summaries=[]; bases=[]; cps=[]; last=None
    given=prefix[:,1:-1][:,CELL]; diff=difficulty(prefix)
    for k,z in enumerate(snapshots):
        dl=model.vqvae.decoder(z); dp=dl.softmax(-1).numpy(); ids=dl.argmax(-1).numpy(); decoded.append(ids)
        if k==11: assert torch.equal(dl,pred.logits)
        cp=heads[k].softmax(-1).numpy(); cps.append(cp)
        ce=-(cp*np.log(np.maximum(cp,1e-30))).sum(-1); cs=np.sort(cp,axis=-1)
        free=np.array([dp[i,CELL][given[i]==0] for i in range(len(given))])
        de=-(free*np.log(np.maximum(free,1e-30))).sum(-1); ds=np.sort(free,axis=-1)
        prob=np.column_stack([ce.mean(1),ce.std(1),(cs[:,:,-1]-cs[:,:,-2]).mean(1),cs[:,:,-1].mean(1),
          de.mean(1),de.std(1),(ds[:,:,-1]-ds[:,:,-2]).mean(1),ds[:,:,-1].mean(1)])
        changes=np.zeros(len(prefix)) if last is None else (ids[:,CELL]!=last[:,CELL]).mean(1)
        bases.append(np.column_stack([diff,prob,observed(ids,given),changes]))
        h=hs[k]; summaries.append(torch.cat([h.mean(1),h.std(1,correction=0)],-1).numpy())
        last=ids
    out=dict(codes=pred.codes.numpy(),decoded=np.stack(decoded,1),hidden=np.stack(summaries,1),
             baseline=np.stack(bases,1),code_probs=np.concatenate(cps,1))
    if local: out['local_hidden']=hs[-1][:,CELL].numpy()
    return out

def ci_rate(success,n):
    z=1.959963984540054; p=success/n; den=1+z*z/n
    mid=(p+z*z/(2*n))/den; half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(mid-half),float(mid+half)]

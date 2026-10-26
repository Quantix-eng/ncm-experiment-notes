"""Four crossed residual states/tails; no gradient, patching, or model update."""
from base import *
from runtime import Engine,parity
from assets import filehash
import random
from solvers import backtracking,exact_cover_count
# Copy the symmetry key, rather than import another campaign's write paths.
def symkey(c):
    a=np.asarray(c).reshape(9,9);keys=[]
    for flip in [False,True]:
        b=a[:,::-1] if flip else a
        for k in range(4):
            v=np.rot90(b,k).ravel();mapping={};keys.append(tuple(mapping.setdefault(int(x),len(mapping)+1) for x in v))
    return min(keys)
def fresh():
    dest=D/'data/fresh.json'
    if dest.exists():return json.loads(dest.read_text())
    excluded=set()
    for task in ['hard','easy','medium']:
        _,s=boards(task);excluded.update(symkey(r[CELL]) for r in s)
    old=json.loads((ROOT/'data/external.json').read_text())['rows'];excluded.update(symkey(r['solution']) for r in old)
    rng=random.Random(Q['seed']);rows=[];attempts=0
    while len(rows)<Q['fresh_n']:
        attempts+=1;assert attempts<Q['fresh_n']*100
        s=backtracking([0]*81,limit=1,rng=rng)[0];key=symkey(s)
        if key in excluded:continue
        p=s.copy();order=list(range(81));rng.shuffle(order)
        for j in order:
            if sum(v!=0 for v in p)<=Q['fresh_clues']:break
            old=p[j];p[j]=0
            if len(backtracking(p,limit=2))!=1:p[j]=old
        if sum(v!=0 for v in p)!=Q['fresh_clues']:continue
        assert exact_cover_count(p,2)==1 and backtracking(p,1)[0]==s
        excluded.add(key);rows.append(dict(puzzle=p,solution=s,attempt=attempts))
        if len(rows)%32==0:log('fresh_puzzles',done=len(rows),attempts=attempts)
    out=dict(seed=Q['seed'],rows=rows,attempts=attempts,uniqueness='MRV and Algorithm X',exclusion='D4 plus digit-normalization solution keys against all official, previous external, and own earlier rows; incomplete isomorphism audit',protocol_sha256=filehash(D/'protocol.json'))
    write(dest,out);return out
def bootstrap(v):
    v=np.asarray(v,dtype=float);rng=np.random.default_rng(Q['seed']);b=v[rng.integers(0,len(v),(Q['bootstrap'],len(v)))].mean(1)
    return dict(mean=float(v.mean()),ci95=np.quantile(b,[.025,.975]),ci975=np.quantile(b,[.0125,.9875]),n=len(v))
def summarize(a):
    report={};ref=a['reference'];blank=a['prefix'][:,1:-1][:,CELL]==0
    for c in Q['conditions']:
        logit=torch.tensor(a[c+'_logits']);gold=torch.tensor(ref);nll=-logit.log_softmax(-1).gather(-1,gold[:,:,None]).squeeze(-1).numpy()[:,CELL];loss=np.array([v[m].mean() for v,m in zip(nll,blank)]);ids=a[c+'_ids'];a[c+'_loss']=loss
        report[c]=dict(metrics=evaluate(ids,ref,a['prefix']),mean_nll_blank=float(loss.mean()),changed_grids_vs_native=int(np.any(ids!=a['Y00_ids'],axis=1).sum()),mean_nll_all=float(nll.mean()))
    l={c:a[c+'_loss'] for c in Q['conditions']};contrasts=dict(R=bootstrap(l['Y10']-l['Y11']),I=bootstrap(l['Y10']+l['Y01']-l['Y11']-l['Y00']),random_minus_adapted=bootstrap(l['random_tail_delta']-l['Y11']),delta_success_Y11_Y10=bootstrap(np.all(a['Y11_ids']==ref,1).astype(float)-np.all(a['Y10_ids']==ref,1)))
    report['contrasts']=contrasts;report['co_primary_positive']=all(contrasts[k]['ci975'][0]>0 for k in ['R','I']);report['geometry']={k:dict(mean=float(a[k].mean()),median=float(np.median(a[k])),q10_q90=np.quantile(a[k],[.1,.9])) for k in ['cosine_to_negative_delta','remaining_delta_ratio','tail_delta_norm','delta_norm']}
    return report
@torch.inference_mode()
def run_set(model,name,p,s,indices):
    accum={};N=len(p)
    def save(k,v):accum.setdefault(k,[]).append(v.numpy().copy() if torch.is_tensor(v) else np.asarray(v).copy())
    for lo in range(0,N,16):
        sl=slice(lo,min(lo+16,N));n=len(p[sl])
        with Engine(model,p[sl]) as e:
            z8,c8=e.first8();z9,c9,_,_=e.step(8,z8);r9=e.contribution(8,c9);delta=-r9
            native=z9.clone();adapt=z8.clone();t0=torch.zeros_like(z8);td=torch.zeros_like(z8);frozen=z8.clone();cross=z9.clone();cn=c8+[c9];ca=c8+[c9]
            for k in range(9,12):
                native,c,_,_=e.step(k,native);rc=e.contribution(k,c);t0=t0+rc;frozen=frozen+rc;cn.append(c)
                adapt,d,_,_=e.step(k,adapt);rd=e.contribution(k,d);td=td+rd;cross=cross+rd;ca.append(d)
            dt=td-t0;unit=np.stack([np.random.default_rng(Q['seed']+1000003*int(i)).normal(size=(89,256)) for i in indices[sl]]);unit/=np.linalg.norm(unit.reshape(n,-1),axis=1)[:,None,None];rand=torch.tensor(unit,dtype=torch.float32)*dt.flatten(1).norm(dim=1)[:,None,None]
            states=dict(Y00=native,Y10=frozen,Y11=adapt,Y01=cross,random_tail_delta=frozen+rand)
            target=model.vqvae._encode_pre_quant(torch.tensor(s[sl]));save('native_codes',torch.cat(cn,1));save('adapted_codes',torch.cat(ca,1))
            for c,z in states.items():
                logits=model.vqvae.decoder(z);save(c+'_logits',logits);save(c+'_ids',logits.argmax(-1));save(c+'_latent_mse',(z-target).square().mean((1,2)))
            norm=delta.flatten(1).norm(dim=1);dn=dt.flatten(1).norm(dim=1);save('delta_norm',norm);save('tail_delta_norm',dn);save('cosine_to_negative_delta',(-delta*dt).sum((1,2))/(norm*dn).clamp_min(1e-20));save('remaining_delta_ratio',(delta+dt).flatten(1).norm(dim=1)/norm)
            # Arithmetic re-association is measured, never silently substituted.
            save('reassociation_max_abs',torch.stack([(native-(z9+t0)).abs().flatten(1).max(1).values,(adapt-(z8+td)).abs().flatten(1).max(1).values],1))
            if name=='hard_test':
                old=np.load(ROOT/'results/hard.npz');assert np.array_equal(torch.cat(cn,1).numpy(),old['codes'][indices[sl]]);assert np.array_equal(states['Y00'].numpy(),model.vqvae.vq.idx_to_zhats(torch.cat(cn,1),return_zlevels=False)[0][-1].numpy());assert np.array_equal(accum['Y00_ids'][-1],old['decoded'][indices[sl],-1])
        log('crossed_batch',dataset=name,done=lo+n,total=N)
    a={k:np.concatenate(v) for k,v in accum.items()};a.update(prefix=p,reference=s,indices=indices);report=summarize(a);np.savez_compressed(D/f'results/{name}_crossed.npz',**a);write(D/f'results/{name}_crossed.json',report);log('crossed_complete',dataset=name,R=report['contrasts']['R'],I=report['contrasts']['I'])
def main():
    freshdata=fresh();parity();model=SudokuModel(ROOT/'artifacts/sudoku/hard')
    a=np.load(ROOT/'results/hard.npz');ix=np.array(SPLIT['test']);run_set(model,'hard_test',a['prefix'][ix],a['reference'][ix],ix)
    a=np.load(ROOT/'results/external.npz');run_set(model,'external_previous',a['prefix'],a['reference'],np.arange(len(a['reference']))+100000)
    p=tokens(np.array([r['puzzle'] for r in freshdata['rows']]),True);s=tokens(np.array([r['solution'] for r in freshdata['rows']]));run_set(model,'fresh',p,s,np.arange(len(p))+200000)
if __name__=='__main__':main()

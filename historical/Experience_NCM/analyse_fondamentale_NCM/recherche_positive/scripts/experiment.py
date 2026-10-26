"""Fresh reserved corpus, train-only terminal readout, and code-writing interventions."""
from base import *
from runtime import Engine
from solvers import backtracking,exact_cover_count
import random,gc
def symkey(c):
    a=np.asarray(c).reshape(9,9);keys=[]
    for flip in [False,True]:
        b=a[:,::-1] if flip else a
        for k in range(4):
            mapping={};keys.append(tuple(mapping.setdefault(int(v),len(mapping)+1) for v in np.rot90(b,k).ravel()))
    return min(keys)
def new_data():
    path=D/'data/fresh.json'
    if path.exists():return json.loads(path.read_text())
    assert (D/'results/protocol_frozen.json').exists();excluded=set()
    for task in ['hard','easy','medium']:
        _,s=boards(task);excluded.update(symkey(row[CELL]) for row in s)
    for file in [ROOT/'data/external.json',ROOT/'compensation_conditionnelle/data/fresh.json']:
        excluded.update(symkey(r['solution']) for r in json.loads(file.read_text())['rows'])
    rng=random.Random(Q['seed']);rows=[];attempts=0
    while len(rows)<Q['fresh_n']:
        attempts+=1;assert attempts<Q['fresh_n']*100
        solution=backtracking([0]*81,1,rng)[0];key=symkey(solution)
        if key in excluded:continue
        p=solution.copy();order=list(range(81));rng.shuffle(order)
        for j in order:
            if sum(v!=0 for v in p)<=Q['clues']:break
            old=p[j];p[j]=0
            if len(backtracking(p,2))!=1:p[j]=old
        if sum(v!=0 for v in p)!=Q['clues']:continue
        assert exact_cover_count(p,2)==1 and backtracking(p,1)[0]==solution
        excluded.add(key);rows.append(dict(puzzle=p,solution=solution,attempt=attempts))
        if len(rows)%128==0:log('positive_fresh',done=len(rows),total=Q['fresh_n'])
    result=dict(seed=Q['seed'],rows=rows,attempts=attempts,protocol_sha256=sha(D/'protocol.json'),exclusion='D4/digit-normalized solution keys against official, both prior external corpora and own previous rows; incomplete isomorphism audit',uniqueness='MRV plus independent Algorithm X')
    write(path,result);return result

@torch.inference_mode()
def native_and_hybrid(model,task,p,ref,tables):
    arrays={};mapping=np.array(tables[task]['mapping'])
    def add(k,v):arrays.setdefault(k,[]).append(v.numpy().copy() if torch.is_tensor(v) else np.asarray(v).copy())
    for lo in range(0,len(p),16):
        pred=model.predict(torch.tensor(p[lo:lo+16]));z=pred.reconstruction;codes=pred.codes;candidate=lookup_board(codes.numpy(),mapping);accept=certificate(candidate,p[lo:lo+16]);native=[];hybrid=[]
        for j in range(len(z)):
            ids=model.vqvae.decoder(z[j:j+1]).argmax(-1).numpy()[0];native.append(ids);hybrid.append(candidate[j] if accept[j] else ids)
        add('codes',codes);add('zfinal',z);add('batched_native',pred.ids);add('native',native);add('candidate',candidate);add('accept',accept);add('hybrid',hybrid)
        if (lo+16)%128==0:log('terminal_readout',checkpoint=task,done=min(lo+16,len(p)),total=len(p))
    a={k:np.concatenate(v) for k,v in arrays.items()};a.update(prefix=p,reference=ref)
    before=np.all(a['native']==ref,1);after=np.all(a['hybrid']==ref,1);verified=a['accept'];assert np.all(np.all(a['candidate']==ref,1)[verified]);assert not np.any(before&~after)
    q=dict(n=len(p),coverage=bootstrap(verified),native=evaluate(a['native'],ref,p),hybrid=evaluate(a['hybrid'],ref,p),lookup_only=evaluate(a['candidate'],ref,p),cell_agreement_lookup=float((a['candidate'][:,CELL]==a['native'][:,CELL]).mean()),native_grids_reproduced_by_lookup=int(np.all(a['candidate']==a['native'],1).sum()),introduced=int((before&~after).sum()),rescued=int((~before&after).sum()),certified_correct=int(np.all(a['candidate']==ref,1)[verified].sum()),certified=int(verified.sum()),quality_delta=bootstrap(after.astype(float)-before),singleton_vs_batched_changed_grids=int(np.any(a['native']!=a['batched_native'],1).sum()))
    np.savez_compressed(D/f'results/{task}_readout.npz',**a);write(D/f'results/{task}_readout.json',q);log('readout_complete',checkpoint=task,coverage=q['coverage'],native=q['native']['solved'],hybrid=q['hybrid']['solved']);return a

@torch.inference_mode()
def interventions(model,a,tables):
    N=Q['causal_n'];a={k:v[:N] for k,v in a.items()};alphabet=np.array(tables['hard']['alphabet_by_digit']);saved={};checks=[]
    def add(k,v):saved.setdefault(k,[]).append(v.numpy().copy() if torch.is_tensor(v) else np.asarray(v).copy())
    for lo in range(0,N,16):
        sl=slice(lo,min(lo+16,N));n=len(a['prefix'][sl]);z=torch.tensor(a['zfinal'][sl]);codes=torch.tensor(a['codes'][sl,-89:]);prefix=a['prefix'][sl];native=torch.tensor(a['native'][sl]);positions=np.array([np.random.default_rng(Q['seed']+lo+j).choice(np.flatnonzero(prefix[j,1:-1][CELL]==0)) for j in range(n)]);tp=CELL[positions]
        with Engine(model,prefix) as e:
            r=e.contribution(11,codes);assert torch.equal(model.vqvae.decoder(z).argmax(-1),native);checks.append('sham_'+str(lo));allp=[];allr=[];norms=[];cnorms=[]
            for digit in range(1,10):
                c=codes.clone();c[torch.arange(n),torch.tensor(tp)]=int(alphabet[digit-1]);delta=e.contribution(11,c)-r;norm=delta.norm(dim=2);unit=np.stack([np.random.default_rng(Q['seed']+(lo+j)*100003+digit*7919).normal(size=(89,256)) for j in range(n)]);unit/=np.linalg.norm(unit,axis=2)[:,:,None];noise=torch.tensor(unit,dtype=torch.float32)*norm[:,:,None];actual=noise.norm(dim=2);assert torch.allclose(norm,actual,atol=1e-6,rtol=1e-6)
                allp.append(model.vqvae.decoder(z+delta).argmax(-1).numpy());allr.append(model.vqvae.decoder(z+noise).argmax(-1).numpy());norms.append(norm.numpy());cnorms.append(actual.numpy())
            add('patched',np.stack(allp,1));add('random',np.stack(allr,1));add('delta_norm_position',np.stack(norms,1));add('random_norm_position',np.stack(cnorms,1));add('cell',positions)
        log('terminal_writing',done=lo+n,total=N)
    b={k:np.concatenate(v) for k,v in saved.items()};b.update(native=a['native'],prefix=a['prefix'],reference=a['reference'],source_indices=np.arange(N),alphabet=alphabet)
    idx=np.arange(N);native_digit=b['native'][idx,CELL[b['cell']]];eligible=np.arange(1,10)[None]!=native_digit[:,None];hits={};collateral={}
    for kind in ['patched','random']:
        pred=b[kind];target=pred[idx[:,None],np.arange(9)[None],CELL[b['cell']][:,None]];h=target==np.arange(1,10)[None];hits[kind]=h;change=pred[:,:,CELL]!=b['native'][:,None,CELL];change[idx[:,None],np.arange(9)[None],b['cell'][:,None]]=False;collateral[kind]=change.sum(2);b[kind+'_hits']=h;b[kind+'_collateral']=collateral[kind]
    b['eligible']=eligible;b['native_digit']=native_digit
    numerator=(hits['patched']&eligible).sum(1);rn=(hits['random']&eligible).sum(1);den=eligible.sum(1);effect=(numerator-rn)/den
    report=dict(n_sources=N,target_writing=ratio_boot(numerator,den),random_target=ratio_boot(rn,den),paired_advantage=bootstrap(effect),collateral={k:dict(mean_other_cells=float(v[eligible].mean()),median_other_cells=float(np.median(v[eligible])),q90=float(np.quantile(v[eligible],.9))) for k,v in collateral.items()},sham_checks=checks,per_digit=[dict(digit=d+1,trials=int(eligible[:,d].sum()),hits=int((hits['patched'][:,d]&eligible[:,d]).sum()),random_hits=int((hits['random'][:,d]&eligible[:,d]).sum())) for d in range(9)])
    np.savez_compressed(D/'results/writing_interventions.npz',**b);write(D/'results/writing_interventions.json',report);log('writing_complete',effect=report['paired_advantage'],target=report['target_writing'])

def main():
    data=new_data();p=tokens(np.array([r['puzzle'] for r in data['rows']]),True);ref=tokens(np.array([r['solution'] for r in data['rows']]));tables=json.loads((D/'results/discovery_tables.json').read_text());model=SudokuModel(ROOT/'artifacts/sudoku/hard');a=native_and_hybrid(model,'hard',p,ref,tables);interventions(model,a,tables);del a,model;gc.collect()
    for task in ['easy','medium']:
        model=SudokuModel(ROOT/f'artifacts/sudoku/{task}');native_and_hybrid(model,task,p[:Q['replication_n']],ref[:Q['replication_n']],tables);del model;gc.collect()
    log('positive_experiments_complete')
if __name__=='__main__':main()

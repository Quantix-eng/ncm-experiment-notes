"""Temporal omissions, fractional readout and matched pre/post perturbations."""
from base import *
from runtime import Engine,parity
from observations import filehash

def put(acc,name,**values):
    v=acc.setdefault(name,{})
    for key,value in values.items():
        if torch.is_tensor(value):value=value.numpy()
        v.setdefault(key,[]).append(np.asarray(value))

def finish(e,z,prior,start):
    codes=list(prior)
    for k in range(start,12):z,c,_,_=e.step(k,z);codes.append(c)
    return dict(ids=e.model.vqvae.decoder(z).argmax(-1),codes=torch.cat(codes,1))

def effect(ids,reference,prefix,native):
    a=np.all(native==reference,axis=1);b=np.all(ids==reference,axis=1)
    return dict(metrics=evaluate(ids,reference,prefix),changed_grids=int(np.any(ids!=native,axis=1).sum()),introduced=int((a&~b).sum()),rescued=int((~a&b).sum()),
      delta_success=paired_mean_ci(b.astype(float)-a),changed_cells=float((ids[:,CELL]!=native[:,CELL]).mean()))

@torch.inference_mode()
def main():
    parity();model=SudokuModel(ROOT/'artifacts/sudoku/hard');cached=np.load(ROOT/'results/hard.npz');ix=np.array(SPLIT['test'])[:Q['intervention_n']];p=cached['prefix'][ix];ref=cached['reference'][ix]
    acc={};reader={};seedpairs={}
    for lo in range(0,len(ix),16):
        sl=np.arange(lo,min(lo+16,len(ix)));prefix=p[sl];n=len(sl)
        with Engine(model,prefix) as e:
            z8,c8=e.first8();z8copy=z8.clone();early=model.vqvae.decoder(z8).argmax(-1);assert np.array_equal(early.numpy(),cached['decoded'][ix[sl],7])
            native=e.branch(z8,c8);assert np.array_equal(native['codes'].numpy(),cached['codes'][ix[sl]]) and np.array_equal(native['final_ids'].numpy(),cached['decoded'][ix[sl],-1])
            c9=native['codes9'];r9=e.contribution(8,c9);z9=z8+r9;assert torch.equal(z9,native['z9'])
            put(acc,'native',ids=native['final_ids'],codes=native['codes'])
            # Same discrete level-9 contribution, only its amplitude is read out.
            for gamma in Q['reader_gamma']:
                logits=model.vqvae.decoder(z8+gamma*r9);put(reader,str(gamma),logits=logits,ids=logits.argmax(-1))
            for gamma in Q['continuation_gamma']:
                out=finish(e,z8+gamma*r9,c8+[c9],9);put(acc,'gamma_'+str(gamma),**out)
            for level in Q['omit_levels']:
                z=torch.zeros_like(z8);cc=[];removed=None;before=None
                for k in range(12):
                    previous=z;z,c,_,_=e.step(k,z,omit_contribution=k==level-1);cc.append(c)
                    if k==level-1:removed=e.contribution(k,c).flatten(1).norm(dim=1);before=previous.flatten(1).norm(dim=1)
                ids=model.vqvae.decoder(z).argmax(-1);put(acc,'omit_'+str(level),ids=ids,codes=torch.cat(cc,1),removed_norm=removed,previous_state_norm=before)
            # Identical delta direction and absolute norm for pre-9 and post-9.
            if lo<Q['susceptibility_n']:
                take=min(n,Q['susceptibility_n']-lo);assert take==n
                scale=z8.flatten(1).norm(dim=1)
                for strength in Q['susceptibility_strengths']:
                    for seed in Q['susceptibility_seeds']:
                        unit=np.stack([np.random.default_rng(seed+int(j)*100003).normal(size=(89,256)) for j in ix[sl]])
                        unit/=np.linalg.norm(unit.reshape(n,-1),axis=1)[:,None,None];delta=torch.tensor(unit,dtype=torch.float32)*(scale*strength)[:,None,None]
                        early_out=e.branch(z8,c8,delta);late_out=finish(e,z9+delta,c8+[c9],9)
                        put(acc,f'noise_{strength}_{seed}_before',ids=early_out['final_ids'],codes=early_out['codes'],delta_norm=delta.flatten(1).norm(dim=1),z8norm=scale)
                        put(acc,f'noise_{strength}_{seed}_after',**late_out,delta_norm=delta.flatten(1).norm(dim=1),z8norm=scale)
            assert torch.equal(z8,z8copy)
        log('temporal_batch',done=lo+n,total=len(ix))
    acc={name:{k:np.concatenate(v) for k,v in values.items()} for name,values in acc.items()};reader={name:{k:np.concatenate(v) for k,v in values.items()} for name,values in reader.items()}
    arrays=dict(indices=ix,prefix=p,reference=ref)
    for name,values in acc.items():
        for key,v in values.items():arrays[name+'_'+key]=v
    np.savez_compressed(D/'results/interventions.npz',**arrays)
    native=acc['native']['ids'];report={}
    for name,v in acc.items():
        count=len(v['ids']);r=effect(v['ids'],ref[:count],p[:count],native[:count]);r['changed_codes']=int((v['codes']!=acc['native']['codes'][:count]).sum())
        if 'removed_norm' in v:r['mean_removed_norm']=float(v['removed_norm'].mean());r['mean_relative_removed_norm']=float((v['removed_norm']/v['previous_state_norm']).mean())
        report[name]=r
    assert np.array_equal(acc['gamma_1.0']['ids'],native) and np.array_equal(acc['gamma_1.0']['codes'],acc['native']['codes'])
    assert np.array_equal(acc['gamma_0.0']['ids'],acc['omit_9']['ids']) and np.array_equal(acc['gamma_0.0']['codes'],acc['omit_9']['codes'])
    write(D/'results/interventions.json',report)
    rdr=dict(indices=ix,reference=ref,prefix=p);rr={}
    for gamma,v in reader.items():
        logits=torch.tensor(v['logits']);lp=logits.log_softmax(-1);gold=torch.tensor(ref)[:,:,None];p_gold=lp.gather(-1,gold).squeeze(-1);others=logits.clone();others.scatter_(-1,gold,-torch.inf);margin=logits.gather(-1,gold).squeeze(-1)-others.max(-1).values
        rr[gamma]=dict(cell_accuracy=float((v['ids'][:,CELL]==ref[:,CELL]).mean()),solved=int(np.all(v['ids']==ref,axis=1).sum()),nll_cells=float(-p_gold[:,CELL].mean()),gold_margin_mean=float(margin[:,CELL].mean()))
        for key,value in v.items():rdr[gamma+'_'+key]=value
    np.savez_compressed(D/'results/reader.npz',**rdr);write(D/'results/reader.json',rr)
    susceptibility={};n=Q['susceptibility_n'];base=native[:n];base_success=np.all(base==ref[:n],axis=1)
    for strength in Q['susceptibility_strengths']:
        v={}
        for timing in ['before','after']:
            predictions=np.stack([acc[f'noise_{strength}_{seed}_{timing}']['ids'] for seed in Q['susceptibility_seeds']]);changed=np.any(predictions!=base[None],axis=-1);success=np.all(predictions==ref[None,:n],axis=-1)
            v[timing]=dict(change_rate=float(changed.mean()),success_rate=float(success.mean()),per_puzzle_change=changed.mean(0),per_puzzle_success=success.mean(0))
        v['after_minus_before_change']=paired_mean_ci(v['after']['per_puzzle_change']-v['before']['per_puzzle_change']);v['after_minus_before_success']=paired_mean_ci(v['after']['per_puzzle_success']-v['before']['per_puzzle_success']);susceptibility[str(strength)]=v
    write(D/'results/susceptibility.json',susceptibility);log('temporal_interventions_complete',native_success=report['native']['metrics']['solved'],omit9_success=report['omit_9']['metrics']['solved'])

if __name__=='__main__':main()

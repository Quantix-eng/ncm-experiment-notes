"""Local steering before step 9 with direct-readout and code-mediated controls."""
from base import *
from runtime import Engine
from probes import feature_blocks,predict
import pickle

@torch.inference_mode()
def main(conditional=False):
    tag='steering_conditional' if conditional else 'steering'
    saved=pickle.loads((D/('results/conditional_models.pkl' if conditional else 'results/probe_models.pkl')).read_bytes())
    m=saved['models']['combined']['alternative'] if conditional else saved['models']['next_combined'];meta=saved['metadata'];bd=meta['baseline_dim']
    a=np.load(D/'results/revision_official.npz');oldcache=np.load(ROOT/'results/hard.npz');te=np.array(SPLIT['test'])[:Q['causal_n']]
    small={key:a[key][te] for key in a.files}
    if conditional:
        from conditional import blocks
        b,s,xy,blank=blocks(small,oldcache['baseline'][te,7],meta['global_projection'])
    else:b,s,xy,blank=feature_blocks(small,oldcache['baseline'][te,7],meta['global_projection'])
    x=np.concatenate([b,s],-1)
    old=small['decoded8'][:,CELL];pp=predict(m,x,forbidden=old if conditional else None);wanted=pp[:,:,1:10].copy()
    wanted[np.arange(len(te))[:,None],np.arange(81)[None,:],np.clip(old-1,0,8)]=-np.inf
    target=wanted.argmax(-1)+1;gain=np.take_along_axis(pp,target[...,None],-1)[...,0]-np.take_along_axis(pp,old[...,None],-1)[...,0]
    if conditional:gain*=predict(saved['models']['combined']['gate'],x)[:,:,1]
    eligible=blank&(old>=1)&(old<=9);gain[~eligible]=-np.inf;cell=gain.argmax(1);assert np.all(np.isfinite(gain[np.arange(len(te)),cell]))
    A=old[np.arange(len(te)),cell];B=target[np.arange(len(te)),cell];assert np.all(A!=B)
    w=m['weight'];raw=(w[:,B].T-w[:,A].T)[:,bd:bd+256]/m['scale'][bd:bd+256];norm=np.linalg.norm(raw,axis=1);assert np.all(norm>1e-12)
    directions=raw/norm[:,None];rng=np.random.default_rng(Q['seed']);random=rng.normal(size=(Q['random_directions'],len(te),256));random/=np.linalg.norm(random,axis=-1,keepdims=True)
    model=SudokuModel(ROOT/'artifacts/sudoku/hard');results={};arrays={};ref=small['reference'];prefix=small['prefix'];rows=np.arange(len(te));tokenpos=CELL[cell]
    def record(name,part,out,indirect,direct,native,delta):
        ids=out['decoded9'].numpy();prob=out['prob9'].numpy();final=out['final_ids'].numpy();j=part
        def values(p):
            p=p.numpy() if torch.is_tensor(p) else p
            pb=p[np.arange(len(j)),tokenpos[j],B[j]];pa=p[np.arange(len(j)),tokenpos[j],A[j]]
            return np.log(np.maximum(pb,1e-15))-np.log(np.maximum(pa,1e-15))
        v=results.setdefault(name,dict(full_logodds=[],indirect_logodds=[],direct_only_logodds=[],full_adopt=[],indirect_adopt=[],direct_adopt=[],
                    final_ids=[],decoded9=[],codes9=[],offtarget_changed=[],delta_norm=[],native_state_norm=[]))
        for key,value in [('full_logodds',values(prob)),('indirect_logodds',values(indirect.softmax(-1))),('direct_only_logodds',values(direct.softmax(-1))),
            ('full_adopt',ids[np.arange(len(j)),tokenpos[j]]==B[j]),('indirect_adopt',indirect.argmax(-1).numpy()[np.arange(len(j)),tokenpos[j]]==B[j]),
            ('direct_adopt',direct.argmax(-1).numpy()[np.arange(len(j)),tokenpos[j]]==B[j])]:v[key].append(np.asarray(value))
        v['final_ids'].append(final);v['decoded9'].append(ids);v['codes9'].append(out['codes9'].numpy())
        changes=ids[:,CELL]!=native['decoded9'].numpy()[:,CELL];changes[np.arange(len(j)),cell[j]]=False;v['offtarget_changed'].append(changes.sum(1))
        v['delta_norm'].append(delta.flatten(1).norm(dim=1).numpy());v['native_state_norm'].append(np.linalg.norm(small['z8'][j,cell[j]].astype(float),axis=1))
    for lo in range(0,len(te),16):
        part=np.arange(lo,min(lo+16,len(te)));p=prefix[part]
        with Engine(model,p) as engine:
            z8,codes8=engine.first8();assert np.array_equal(z8[:,CELL].numpy(),small['z8'][part])
            native=engine.branch(z8,codes8);assert np.array_equal(native['decoded9'].numpy(),small['decoded9'][part])
            zero=torch.zeros_like(z8);record('sham',part,native,model.vqvae.decoder(native['z9']),model.vqvae.decoder(native['z9']),native,zero)
            for strength in ([Q['primary_strength']] if conditional else Q['causal_strengths']):
                for kind,unit in [('positive',directions[part]),('negative',-directions[part])]+[(f'random{i}',random[i,part]) for i in range(Q['random_directions'])]:
                    delta=torch.zeros_like(z8);selected=z8[torch.arange(len(part)),torch.tensor(tokenpos[part])];size=selected.norm(dim=-1)*strength
                    delta[torch.arange(len(part)),torch.tensor(tokenpos[part])]=torch.tensor(unit,dtype=torch.float32)*size[:,None]
                    out=engine.branch(z8,codes8,delta);indirect=model.vqvae.decoder(z8+engine.contribution(8,out['codes9']));direct=model.vqvae.decoder((z8+delta)+engine.contribution(8,native['codes9']))
                    record(f'{strength}_{kind}',part,out,indirect,direct,native,delta)
        log('steering_batch',done=lo+len(part),total=len(te))
    results={name:{key:np.concatenate(value) for key,value in v.items()} for name,v in results.items()};report={}
    base=results['sham'];basefinal=np.all(base['final_ids']==ref,axis=1)
    for name,v in results.items():
        new=np.all(v['final_ids']==ref,axis=1)
        report[name]=dict(n=len(te),full_adoption=float(v['full_adopt'].mean()),indirect_adoption=float(v['indirect_adopt'].mean()),direct_only_adoption=float(v['direct_adopt'].mean()),
           full_logodds_gain=paired_mean_ci(v['full_logodds']-base['full_logodds']),indirect_logodds_gain=paired_mean_ci(v['indirect_logodds']-base['indirect_logodds']),
           direct_only_logodds_gain=paired_mean_ci(v['direct_only_logodds']-base['direct_only_logodds']),
           mean_offtarget_changed_cells=float(v['offtarget_changed'].mean()),changed_level9_codes=int((v['codes9']!=base['codes9']).sum()),
           final_metrics=evaluate(v['final_ids'],ref,prefix),delta_final_success=paired_mean_ci(new.astype(float)-basefinal),
           introduced_errors=int((basefinal&~new).sum()),rescued=int((~basefinal&new).sum()))
        for key,value in v.items():arrays[name+'_'+key]=value
    strength=str(Q['primary_strength']);randommean=np.mean([results[strength+f'_random{i}']['indirect_logodds'] for i in range(Q['random_directions'])],axis=0)
    primary=paired_mean_ci(results[strength+'_positive']['indirect_logodds']-randommean)
    signed=paired_mean_ci(results[strength+'_positive']['indirect_logodds']-results[strength+'_negative']['indirect_logodds'])
    write(D/f'results/{tag}.json',dict(primary_indirect_positive_vs_random=primary,secondary_positive_vs_negative=signed,conditions=report,
       selection='First 96 fixed test puzzles; one initially blank cell and alternative chosen using only pre-step9 features and a train/validation-fitted probe.',
       interpretation='Removing delta at readout is an intervention control, not identification of a unique natural mediation effect.'))
    np.savez_compressed(D/f'results/{tag}.npz',indices=te,cell=cell,current=A,target=B,direction=directions,random_directions=random,reference=ref,**arrays)
    log('steering_complete',source=tag,primary=primary)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--conditional',action='store_true');args=parser.parse_args();main(conditional=args.conditional)

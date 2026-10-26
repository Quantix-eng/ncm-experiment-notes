"""Batch/symmetry controls, independent corpus, substitutions and causal hooks."""
from common import *
from prepare import symkey
from solvers import backtracking, exact_cover_count
from scipy.stats import spearmanr
import argparse, random, pickle

def batch_trace(model,p,batch=16):
    out={}
    for lo in range(0,len(p),batch):
        t=trace(model,p[lo:lo+batch])
        for key,v in t.items(): out.setdefault(key,[]).append(v)
    return {key:np.concatenate(v) for key,v in out.items()}

def difference(a,b):
    return dict(hidden_max_abs=float(np.max(np.abs(a['hidden']-b['hidden']))),
       hidden_relative_l2=float(np.linalg.norm(a['hidden']-b['hidden'])/max(1e-12,np.linalg.norm(a['hidden']))),
       code_probability_max_abs=float(np.max(np.abs(a['code_probs']-b['code_probs']))),
       code_disagreements=int((a['codes']!=b['codes']).sum()),
       output_disagreements=int((a['decoded'][:,-1]!=b['decoded'][:,-1]).sum()))

def stability(model,p):
    start=time.perf_counter(); p=p[:64]; base=batch_trace(model,p[:32],32); results={}
    for j in range(3): results[f'repeat_{j}']=difference(base,batch_trace(model,p[:32],32))
    for b in [1,8]: results[f'batch_{b}']=difference(base,batch_trace(model,p[:32],b))
    rev=batch_trace(model,p[:32][::-1].copy(),32); results['reversed']=difference(base,{k:v[::-1] for k,v in rev.items()})
    mix=batch_trace(model,np.concatenate([p[:32],p[32:]]),64); results['composition_64']=difference(base,{k:v[:32] for k,v in mix.items()})
    write(ROOT/'results/stability.json',dict(contrasts=results,n=32,seconds=time.perf_counter()-start))
    progress('stability_complete',results=results)

def symmetry(model,p,s):
    ix=np.array(json.loads((ROOT/'results/splits.json').read_text())['test'])[:P['symmetry_n']]
    p=p[ix]; s=s[ix]; cells=p[:,1:-1][:,CELL]; ref=s[:,CELL]; rng=np.random.default_rng(P['seed'])
    mapping=np.r_[0,rng.permutation(np.arange(1,10))]; invers=np.argsort(mapping)
    variants=dict(identity=(cells,ref),digits=(mapping[cells],mapping[ref]),
      rotation=(np.rot90(cells.reshape(-1,9,9),1,axes=(1,2)).reshape(-1,81),np.rot90(ref.reshape(-1,9,9),1,axes=(1,2)).reshape(-1,81)),
      reflection=(cells.reshape(-1,9,9)[:,:,::-1].reshape(-1,81),ref.reshape(-1,9,9)[:,:,::-1].reshape(-1,81)))
    out={}; arrays={}
    for name,(c,r) in variants.items():
        prompt=tokens(c,True); sol=tokens(r); t=batch_trace(model,prompt); pred=t['decoded'][:,-1][:,CELL].copy()
        if name=='digits':
            pred=np.where(pred<=9,invers[np.minimum(pred,9)],pred)
        if name=='rotation': pred=np.rot90(pred.reshape(-1,9,9),-1,axes=(1,2)).reshape(-1,81)
        if name=='reflection': pred=pred.reshape(-1,9,9)[:,:,::-1].reshape(-1,81)
        arrays.update({name+'_'+k:v for k,v in t.items()}); arrays[name+'_prefix']=prompt; arrays[name+'_reference']=sol
        out[name]=dict(metrics=evaluate(t['decoded'][:,-1],sol,prompt),
            equivariance_cell_agreement=None if name=='identity' else float((pred==arrays['identity_decoded'][:,-1][:,CELL]).mean()),
            equivariance_whole_board_agreement=None if name=='identity' else float(np.all(pred==arrays['identity_decoded'][:,-1][:,CELL],axis=1).mean()))
        progress('symmetry',variant=name,**out[name]['metrics'])
    hidden=np.stack([arrays[name+'_hidden'] for name in variants],axis=1)
    intra=np.var(hidden,axis=1).mean(axis=(0,2)); inter=np.var(hidden.mean(axis=1),axis=0).mean(-1)
    write(ROOT/'results/symmetry.json',dict(n=len(p),indices=ix,variants=out,intra_variance_by_level=intra,inter_variance_by_level=inter,
          intra_inter_ratio=intra/np.maximum(inter,1e-12),digit_mapping=mapping))
    np.savez_compressed(ROOT/'results/symmetry_traces.npz',**arrays)

def cross_models():
    split=json.loads((ROOT/'results/splits.json').read_text()); report={}; arrays={}
    for modelname in ['easy','medium','hard']:
        model=SudokuModel(ROOT/f'artifacts/sudoku/{modelname}')
        for corpus in ['easy','medium','hard']:
            p,s=boards(corpus); ix=np.array(split['cross'][corpus]); p=p[ix]; s=s[ix]
            pred=[]
            for lo in range(0,len(p),16): pred.append(model.predict(torch.tensor(p[lo:lo+16])).ids.numpy())
            pred=np.concatenate(pred); result=evaluate(pred,s,p); result['solved_wilson95']=ci_rate(result['solved'],len(p))
            report[f'{modelname}_on_{corpus}']=result; arrays[f'{modelname}_on_{corpus}']=pred
            progress('cross_model',checkpoint=modelname,corpus=corpus,**result)
    write(ROOT/'results/cross_models.json',report); np.savez_compressed(ROOT/'results/cross_predictions.npz',**arrays)

def external(model):
    start=time.perf_counter(); excluded=set()
    for d in ['easy','medium','hard']:
        _,s=boards(d); excluded.update(symkey(row) for row in s[:,CELL])
    rng=random.Random(P['seed']); rows=[]; attempts=0; rejected=0
    while len(rows)<P['external_n']:
        attempts+=1
        if attempts>P['external_n']*100: raise RuntimeError('Generation cap reached')
        solution=backtracking([0]*81,limit=1,rng=rng)[0]; key=symkey(solution)
        if key in excluded: rejected+=1; continue
        puzzle=solution.copy(); positions=list(range(81)); rng.shuffle(positions)
        for pos in positions:
            old,puzzle[pos]=puzzle[pos],0
            if len(backtracking(puzzle))!=1: puzzle[pos]=old
            if sum(bool(v) for v in puzzle)==30: break
        if sum(bool(v) for v in puzzle)!=30: rejected+=1; continue
        assert exact_cover_count(puzzle)==1
        assert backtracking(puzzle,limit=1)[0]==solution
        excluded.add(key); rows.append(dict(puzzle=puzzle,solution=solution,key=key,attempt=attempts))
        if len(rows)%32==0: progress('external_generation',done=len(rows),attempts=attempts,seconds=time.perf_counter()-start)
    write(ROOT/'data/external.json',dict(seed=P['seed'],rows=rows,attempts=attempts,rejections=rejected,
        uniqueness='MRV backtracking + independent Algorithm X',excluded='official solution D4 + digit-normalization keys; no exhaustive isomorphism exclusion'))
    p=tokens(np.array([r['puzzle'] for r in rows]),True); s=tokens(np.array([r['solution'] for r in rows])); t=batch_trace(model,p)
    np.savez_compressed(ROOT/'results/external.npz',prefix=p,reference=s,**t)
    met=evaluate(t['decoded'][:,-1],s,p); met['solved_wilson95']=ci_rate(met['solved'],len(p))
    write(ROOT/'results/external.json',dict(metrics=met,difficulty=difficulty(p).mean(0),seconds=time.perf_counter()-start))
    progress('external_complete',**met)

def official_uniqueness():
    report={}
    for d in ['hard','easy','medium']:
        start=time.perf_counter(); p,_=boards(d); counts=[]
        for j,row in enumerate(p[:,1:-1][:,CELL]):
            counts.append(exact_cover_count(row))
            if j%256==0: progress('official_uniqueness',corpus=d,done=j)
        report[d]=dict(counts=np.bincount(counts,minlength=3),per_puzzle=counts,count_capped_at=2,seconds=time.perf_counter()-start)
        write(ROOT/'results/official_uniqueness.json',report)

@torch.inference_mode()
def functional(model):
    _,s=boards('hard'); ix=np.array(json.loads((ROOT/'results/splits.json').read_text())['train'])[:P['functional_substitution_n']]
    idx=model.vqvae.encode(torch.tensor(s[ix])); out=[]; offset=0
    for k,l in enumerate(model.net.levels):
        outputs=[]; probs=[]
        # Sixteen interventions per context suffice for every one of the 120 pairs.
        for code in range(16):
            mod=idx.clone(); mod[:,offset]=code; zs,=model.vqvae.vq.idx_to_zhats(mod,return_zlevels=False)
            logits=model.vqvae.decoder(zs[-1]); outputs.append(logits.argmax(-1).numpy()[:,CELL]); probs.append(logits.softmax(-1).numpy()[:,CELL])
        outputs=np.array(outputs); probs=np.array(probs); disagree=np.zeros((16,16)); tv=np.zeros((16,16))
        for a in range(16):
            for b in range(16):
                disagree[a,b]=(outputs[a]!=outputs[b]).mean(); tv[a,b]=np.abs(probs[a]-probs[b]).sum(-1).mean()/2
        emb=model.vqvae.vq.embeddings[k].weight.numpy(); dist=np.linalg.norm(emb[:,None]-emb[None,:],axis=-1); upper=np.triu_indices(16,1)
        rho=float(spearmanr(dist[upper],disagree[upper]).statistic)
        out.append(dict(level=k+1,contexts=len(ix),position_within_level=0,disagreement=disagree,total_variation=tv,
            embedding_effect_spearman=None if not np.isfinite(rho) else rho,min_cell_disagreement=float(disagree[upper].min()),
            median_cell_disagreement=float(np.median(disagree[upper])),max_cell_disagreement=float(disagree[upper].max()),
            zero_disagreement_pairs=int((disagree[upper]==0).sum())))
        offset+=l; progress('functional_substitutions',level=k+1)
    write(ROOT/'results/functional_substitutions.json',dict(indices=ix,levels=out,
      caveat='Conditional on 32 gold-encoded solutions, replacing the first slot only; may be out of distribution and cannot prove global interchangeability.'))

@torch.inference_mode()
def causal(model):
    start=time.perf_counter(); p,s=boards('hard'); te=np.array(json.loads((ROOT/'results/splits.json').read_text())['test']); p=p[te]; s=s[te]
    frozen=np.load(ROOT/'results/causal_direction.npz'); u=torch.tensor(frozen['direction'],dtype=torch.float32); mu=torch.tensor(frozen['mean'],dtype=torch.float32)
    rng=np.random.default_rng(P['seed']); controls=[]
    for _ in range(P['causal_random_directions']):
        r=rng.normal(size=512); r/=np.linalg.norm(r); controls.append(torch.tensor(r,dtype=torch.float32))
    report={}; arrays={}; baseline=[]
    for lo in range(0,len(p),16): baseline.append(model.predict(torch.tensor(p[lo:lo+16])).ids.numpy())
    baseline=np.concatenate(baseline); gold=np.all(baseline==s,axis=1); arrays['baseline']=baseline
    for name,r in [('sham',None),('confidence',u)]+[(f'random_{i}',r) for i,r in enumerate(controls)]:
        predictions=[]; norms=[]; calls=[0]
        def hook(mod,args,out):
            call=calls[0]; calls[0]+=1
            if call!=P['primary_level']: return out # prefill call=0, level 6 call=6
            if r is None: return out
            centered=out-mu; projection=(centered@u).unsqueeze(-1)
            if name=='confidence': delta=projection*u
            else:
                sign=torch.where((centered@r)>=0,1.0,-1.0).unsqueeze(-1)
                delta=projection.abs()*sign*r # exact per-token norm match to confidence ablation
            assert out.shape[1]==model.net.levels[P['primary_level']-1]
            norms.append(dict(delta_rms=float(delta.square().mean().sqrt()),state_rms=float(out.square().mean().sqrt()),
                confidence_projection_rms_before=float(projection.square().mean().sqrt()),
                confidence_projection_rms_after=float(((out-delta-mu)@u).square().mean().sqrt())))
            return out-delta
        for lo in range(0,len(p),16):
            calls[0]=0; handle=model.net.transformer.blocks[-1].register_forward_hook(hook)
            try: pred=model.predict(torch.tensor(p[lo:lo+16])).ids.numpy()
            finally: handle.remove()
            assert calls[0]==13
            predictions.append(pred)
        pred=np.concatenate(predictions); arrays[name]=pred; correct=np.all(pred==s,axis=1)
        diff=correct.astype(float)-gold.astype(float); brng=np.random.default_rng(P['seed']); boots=diff[brng.integers(0,len(diff),(P['bootstrap'],len(diff)))].mean(1)
        report[name]=dict(metrics=evaluate(pred,s,p),introduced_errors=int((gold&~correct).sum()),rescued=int((~gold&correct).sum()),
            changed_boards=int(np.any(pred!=baseline,axis=1).sum()),success_status_changes=int((gold!=correct).sum()),
            success_status_change_wilson95=ci_rate(int((gold!=correct).sum()),len(gold)),delta_solved=float(diff.mean()),delta_ci95=np.quantile(boots,[.025,.975]),
            perturbation=norms)
        progress('causal_intervention',condition=name,**report[name]['metrics'])
    assert np.array_equal(arrays['sham'],baseline)
    np.savez_compressed(ROOT/'results/causal_predictions.npz',indices=te,reference=s,**arrays)
    write(ROOT/'results/causal.json',dict(level=P['primary_level'],layer=8,conditions=report,seconds=time.perf_counter()-start,
        norm_matching='same per-token absolute projection on confidence direction at the unmodified intervention state',
        caveat='External probe direction; intervention tests influence on resolution, not endogenous metacognitive control. Five random directions do not support precise tail probabilities.'))

def score_frozen():
    from analyze import score_metrics, paired_auc
    models=pickle.loads((ROOT/'results/probe_models.pkl').read_bytes()); k=P['primary_level']-1; report={}
    for kind in ['external','symmetry_traces']:
        arr=np.load(ROOT/f'results/{kind}.npz')
        variants=['identity','digits','rotation','reflection'] if kind=='symmetry_traces' else ['']
        for var in variants:
            key=var+'_' if var else ''; h=arr[key+'hidden'][:,k].astype(float); b=arr[key+'baseline'][:,k].astype(float)
            y=np.all(arr[key+'decoded'][:,-1]==arr[key+'reference'],axis=1).astype(int)
            result={}
            for name,x in [('baseline',b),('hidden',h),('combined',np.column_stack([b,h]))]:
                pp=models[(k,name)].predict_proba(x)[:,1]; result[name]=score_metrics(y,pp)
                if name=='combined':
                    if var=='identity': identity=pp; identity_y=y
                    elif var:
                        changes=np.abs(pp-identity); delta=y.astype(float)-identity_y; rr=np.random.default_rng(P['seed'])
                        ix=rr.integers(0,len(y),(P['bootstrap'],len(y)))
                        result['confidence_vs_identity']=dict(mean_abs_change=float(changes.mean()),spearman=float(spearmanr(pp,identity).statistic),
                            mean_abs_change_ci95=np.quantile(changes[ix].mean(1),[.025,.975]),
                            delta_success=float(delta.mean()),delta_success_ci95=np.quantile(delta[ix].mean(1),[.025,.975]),
                            resampling='paired by source puzzle; all variants stay together')
            report[kind+('_'+var if var else '')]=result
    write(ROOT/'results/transfer_confidence.json',report)
    # Replicate frozen hard probes on each model's own corpus: feature transfer, no refit or test tuning.
    transfer={}
    for d in ['easy','medium']:
        a=np.load(ROOT/f'results/{d}.npz'); overlap=json.loads((ROOT/'results/cross_corpus_overlap.json').read_text())
        test_families=json.loads((ROOT/'results/splits.json').read_text())['test']
        ix=np.flatnonzero(np.isin(overlap['solution_family_to_hard_index'][d],test_families))
        h=a['hidden'][ix,k].astype(float); b=a['baseline'][ix,k].astype(float); y=np.all(a['decoded'][ix,-1]==a['reference'][ix],axis=1).astype(int)
        transfer[d]={name:score_metrics(y,models[(k,name)].predict_proba(x)[:,1]) for name,x in [('baseline',b),('hidden',h),('combined',np.column_stack([b,h]))]}
    write(ROOT/'results/cross_checkpoint_probe_transfer.json',dict(results=transfer,caveat='Direct transfer between unaligned hidden coordinate systems, evaluated only on the 600 solution families held out from hard probe training/validation; failure does not imply absence of decodable information.'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['pre','post','all','causal','scores'],default='all'); args=parser.parse_args()
    if args.stage in ['pre','all']:
        model=SudokuModel(ROOT/'artifacts/sudoku/hard'); p,s=boards('hard')
        stability(model,p); symmetry(model,p,s); external(model); functional(model); del model
        cross_models(); official_uniqueness()
    if args.stage in ['post','all','causal']:
        model=SudokuModel(ROOT/'artifacts/sudoku/hard'); causal(model)
    if args.stage in ['post','all','scores']: score_frozen()
    progress('experiments_'+args.stage+'_complete')

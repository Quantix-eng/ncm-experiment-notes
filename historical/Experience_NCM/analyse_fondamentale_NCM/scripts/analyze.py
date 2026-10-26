"""Frozen splits, gold-free confidence probes, descriptive hierarchy analyses."""
from common import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import warnings, pickle
threadpool_limits(1)
warnings.filterwarnings('error',category=ConvergenceWarning)

def fit_probe(x,y,train,val,shuffle=False):
    yy=y.copy()
    if shuffle: yy[train]=np.random.default_rng(P['seed']).permutation(yy[train])
    fits=[]
    for C in P['logistic_C']:
        m=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=2000,solver='lbfgs',random_state=P['seed']))
        m.fit(x[train],yy[train]); score=m.predict_proba(x[val])[:,1]
        fits.append((log_loss(y[val],score,labels=[0,1]),C,m))
    loss,C,m=min(fits,key=lambda a:a[0])
    return m,dict(C=C,validation_logloss=float(loss),n_train=len(train),train_successes=int(y[train].sum()),
                  solver_iterations=int(m[-1].n_iter_[0]))

def score_metrics(y,p):
    y=np.asarray(y); p=np.asarray(p); auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
    bins=np.minimum((p*10).astype(int),9); reliability=[]; ece=0
    for b in range(10):
        mask=bins==b
        if mask.any():
            acc=float(y[mask].mean()); conf=float(p[mask].mean()); ece+=mask.mean()*abs(acc-conf)
            reliability.append(dict(bin=b,n=int(mask.sum()),confidence=conf,accuracy=acc))
    # Include entire tied-score groups rather than selecting arbitrary ties.
    order=np.argsort(-p,kind='stable'); ends=np.flatnonzero(np.r_[p[order][1:]!=p[order][:-1],True]); cum=np.cumsum(1-y[order])
    curve=[dict(coverage=float((e+1)/len(y)),risk=float(cum[e]/(e+1)),threshold=float(p[order[e]])) for e in ends]
    return dict(n=len(y),successes=int(y.sum()),failures=int((1-y).sum()),auc=auc,brier=float(np.mean((p-y)**2)),
                ece10=float(ece),reliability=reliability,risk_coverage=curve)

def paired_auc(y,p,q,groups=None):
    rng=np.random.default_rng(P['seed']); values=[]; raw=[]
    if groups is None: groups=np.arange(len(y))
    unique=np.unique(groups); members=[np.flatnonzero(groups==g) for g in unique]
    for _ in range(P['bootstrap']):
        ix=np.concatenate([members[j] for j in rng.integers(0,len(unique),len(unique))])
        if len(np.unique(y[ix]))<2: continue
        a=roc_auc_score(y[ix],p[ix]); b=roc_auc_score(y[ix],q[ix]); values.append(a-b); raw.append([a,b])
    return dict(delta=float(roc_auc_score(y,p)-roc_auc_score(y,q)),delta_ci95=np.quantile(values,[.025,.975]).tolist(),
        auc_ci95=np.quantile(raw,[.025,.975],axis=0).tolist(),bootstrap_valid=len(values),method='paired percentile group bootstrap; fixed fitted models')

def token_stats(a,lengths):
    out=[]; offset=0
    for k,l in enumerate(lengths):
        codes=a[:,offset:offset+l]; counts=np.bincount(codes.ravel(),minlength=16); probs=counts/counts.sum()
        entropy=float(-(probs[probs>0]*np.log(probs[probs>0])).sum())
        position_entropies=[]
        for slot in range(l):
            freq=np.bincount(codes[:,slot],minlength=16)/len(codes)
            position_entropies.append(float(-(freq[freq>0]*np.log(freq[freq>0])).sum()))
        cov={str(n):int(len(np.unique(codes[:n]))) for n in [16,64,256,1000,2000] if n<=len(codes)}
        out.append(dict(level=k+1,length=l,counts=counts,entropy_nats=entropy,normalized_entropy=entropy/np.log(16),
           conditional_entropy_given_position_nats=float(np.mean(position_entropies)),code_position_mutual_information_nats=float(entropy-np.mean(position_entropies)),
           perplexity=float(np.exp(entropy)),utilization=float((counts>0).mean()),unobserved=np.flatnonzero(counts==0),coverage=cov))
        offset+=l
    return out

def main(primary_only=False):
    start=time.perf_counter(); split=json.loads((ROOT/'results/splits.json').read_text()); tr,va,te=[np.array(split[n]) for n in ['train','validation','test']]
    z=np.load(ROOT/'results/hard.npz'); y=np.all(z['decoded'][:,-1]==z['reference'],axis=1).astype(int)
    preds={}; models={}; probe_report=[]; maturity=[]; solutions=z['reference'][:,CELL]-1
    projection=np.random.default_rng(P['seed']).normal(size=(1024,128))/np.sqrt(128)
    targets=np.eye(9)[solutions].reshape(len(y),-1)
    puzzle=np.eye(10)[z['prefix'][:,1:-1][:,CELL]].reshape(len(y),-1)
    puzzle_projection=np.random.default_rng(P['seed']+1).normal(size=(810,128))/np.sqrt(128)
    puzzle_ridge=make_pipeline(StandardScaler(),Ridge(alpha=P['solution_ridge_alpha'],solver='cholesky'))
    puzzle_ridge.fit(puzzle[tr]@puzzle_projection,targets[tr])
    puzzle_pred=puzzle_ridge.predict(puzzle[te]@puzzle_projection).reshape(len(te),81,9).argmax(-1)
    puzzle_blank_acc=float((puzzle_pred==solutions[te])[z['prefix'][te,1:-1][:,CELL]==0].mean())
    for k in range(12):
        h=z['hidden'][:,k].astype(float); b=z['baseline'][:,k].astype(float)
        features=dict(difficulty=b[:,:6],baseline=b,hidden=h,combined=np.column_stack([b,h]),shuffled_hidden=h)
        result={}; preds[k]={}
        for name,x in features.items():
            m,info=fit_probe(x,y,tr,va,shuffle=name=='shuffled_hidden'); pp=m.predict_proba(x)[:,1]
            models[(k,name)]=m; preds[k][name]=pp
            result[name]=dict(**score_metrics(y[te],pp[te]),fit=info)
        prior=np.full(len(y),y[tr].mean()); preds[k]['prior']=prior; result['prior']=score_metrics(y[te],prior[te])
        probe_report.append(dict(level=k+1,models=result))
        # Fixed projection and fixed regularization; no test-based selection.
        x=h@projection; ridge=make_pipeline(StandardScaler(),Ridge(alpha=P['solution_ridge_alpha'],solver='cholesky'))
        ridge.fit(x[tr],targets[tr]); scores=ridge.predict(x[te]).reshape(len(te),81,9); ids=scores.argmax(-1)
        maturity.append(dict(level=k+1,linear_cell_accuracy=float((ids==solutions[te]).mean()),
           decoded_cell_accuracy=float((z['decoded'][te,k][:,CELL]==z['reference'][te][:,CELL]).mean()),
           puzzle_only_linear_blank_accuracy=puzzle_blank_acc,
           linear_blank_accuracy=float((ids==solutions[te])[z['prefix'][te,1:-1][:,CELL]==0].mean())))
        progress('probes',level=k+1,baseline_auc=result['baseline']['auc'],combined_auc=result['combined']['auc'])
    k=P['primary_level']-1
    primary=paired_auc(y[te],preds[k]['combined'][te],preds[k]['baseline'][te])
    # Additional direction uses mean states only; maps exactly to the manipulated activation space.
    direction_model,direction_info=fit_probe(z['hidden'][:,k,:512],y,tr,va)
    scaler,clf=direction_model; direction=clf.coef_[0]/scaler.scale_; direction/=np.linalg.norm(direction)
    np.savez(ROOT/'results/causal_direction.npz',direction=direction,mean=scaler.mean_)
    models['causal_direction']=direction_model
    with (ROOT/'results/probe_models.pkl').open('wb') as f: pickle.dump(models,f)
    np.savez_compressed(ROOT/'results/probe_predictions.npz',y=y,test=te,projection=projection,puzzle_projection=puzzle_projection,
        **{f'level{k+1}_{name}':p for k,d in preds.items() for name,p in d.items()})
    # Association with difficulty stratified by training-derived quartiles; descriptive, not causal adjustment.
    diff=z['baseline'][:,0,1]; edges=np.unique(np.quantile(diff[tr],[0,.25,.5,.75,1])); strata=[]
    bins=np.digitize(diff[te],edges[1:-1])
    for i in range(len(edges)-1):
        ix=te[bins==i]
        if not len(ix): continue
        strata.append(dict(bin=i,n=len(ix),success_rate=float(y[ix].mean()),difficulty_range=[float(edges[i]),float(edges[i+1])],
            hidden_auc=score_metrics(y[ix],preds[k]['hidden'][ix])['auc'],
            baseline_auc=score_metrics(y[ix],preds[k]['baseline'][ix])['auc'],combined_auc=score_metrics(y[ix],preds[k]['combined'][ix])['auc']))
    write(ROOT/'results/probes.json',dict(primary=primary,primary_level=k+1,levels=probe_report,maturity=maturity,
          difficulty_strata=strata,causal_direction_fit=direction_info,seconds=time.perf_counter()-start,
          caveats=['Bootstrap conditions on a single fitted split; no training-seed uncertainty.',
                   'Other levels, strata, maturity and mechanisms exploratory; no familywise significance claims.']))
    # Local naked-single decoder: final-level states at cells; labels are computed only from the initial prompt.
    local=z['local_hidden']; c=z['prefix'][:,1:-1][:,CELL]; count=np.array([candidate_counts(row) for row in c]); label=(count==1).astype(int)
    trainp=tr[:512]; valp=va[:128]; testp=te[:256]
    def flat(ix):
        mask=c[ix]==0; return local[ix][mask],label[ix][mask]
    tx,ty=flat(trainp); vx,vy=flat(valp); ex,ey=flat(testp)
    x=np.concatenate([tx,vx,ex]); yy=np.concatenate([ty,vy,ey]); a=np.arange(len(tx)); b=np.arange(len(tx),len(tx)+len(vx))
    m,info=fit_probe(x,yy,a,b); ep=m.predict_proba(ex)[:,1]
    shuffle,si=fit_probe(x,yy,a,b,shuffle=True); sp=shuffle.predict_proba(ex)[:,1]
    group=np.repeat(testp,(c[testp]==0).sum(1))
    contrast=paired_auc(ey,ep,sp,group)
    write(ROOT/'results/naked_single.json',dict(test_puzzles=len(testp),test_cells=len(ey),positives=int(ey.sum()),
          hidden=score_metrics(ey,ep),shuffled_labels=score_metrics(ey,sp),contrast=contrast,fit=info,
          explicit_candidate_baseline_auc=1.0,interpretation='Decodability of an initial constraint at final generation; no demonstrated monospecific neuron or rule circuit.'))
    with (ROOT/'results/naked_single_probe.pkl').open('wb') as f: pickle.dump(m,f)
    if not primary_only: finish()
    progress('primary_analysis_complete',primary=primary)

def finish():
    split=json.loads((ROOT/'results/splits.json').read_text()); tr,va,te=[np.array(split[n]) for n in ['train','validation','test']]
    k=P['primary_level']-1
    models=pickle.loads((ROOT/'results/probe_models.pkl').read_bytes())
    replications={}
    for d in ['easy','medium']:
        arr=np.load(ROOT/f'results/{d}.npz'); yy=np.all(arr['decoded'][:,-1]==arr['reference'],axis=1).astype(int)
        hh=arr['hidden'][:,k]; bb=arr['baseline'][:,k]; rr={}; pp={}
        for name,xx in [('baseline',bb),('hidden',hh),('combined',np.column_stack([bb,hh]))]:
            mm,info=fit_probe(xx,yy,tr,va); pp[name]=mm.predict_proba(xx[te])[:,1]
            rr[name]=dict(**score_metrics(yy[te],pp[name]),fit=info)
            models[(d,k,name)]=mm
        rr['contrast']=paired_auc(yy[te],pp['combined'],pp['baseline']) if len(np.unique(yy[te]))==2 else None
        replications[d]=rr
    write(ROOT/'results/probe_replications.json',replications)
    with (ROOT/'results/probe_models.pkl').open('wb') as f: pickle.dump(models,f)
    # Audit each tokenizer and reconstruction progression on its own reference corpus.
    report={}
    for d in ['hard','easy','medium']:
        arr=np.load(ROOT/f'results/{d}.npz'); model=SudokuModel(ROOT/f'artifacts/sudoku/{d}'); lengths=list(model.net.levels)
        dist=[]
        for emb in model.vqvae.vq.embeddings:
            e=emb.weight.detach().numpy(); distances=np.sqrt(np.maximum(((e[:,None]-e[None,:])**2).sum(-1),0)); vals=distances[np.triu_indices(16,1)]
            dist.append(dict(min=float(vals.min()),median=float(np.median(vals)),max=float(vals.max()),matrix=distances))
        reconstructed=arr['reconstructed']; decoded=arr['decoded']; ref=arr['reference']; transitions=[]
        rec=[]; gen=[]
        for k in range(12):
            rec.append(dict(level=k+1,token_accuracy=float((reconstructed[:,k]==ref).mean()),cell_accuracy=float((reconstructed[:,k][:,CELL]==ref[:,CELL]).mean()),
                exact_rate=float(np.all(reconstructed[:,k]==ref,axis=1).mean())))
            gen.append(dict(level=k+1,**evaluate(decoded[:,k],ref,arr['prefix'])))
            if k:
                old=decoded[:,k-1][:,CELL]==ref[:,CELL]; new=decoded[:,k][:,CELL]==ref[:,CELL]
                transitions.append(dict(level=k+1,correct_to_incorrect=int((old&~new).sum()),incorrect_to_correct=int((~old&new).sum()),
                    changed_cells=int((decoded[:,k-1][:,CELL]!=decoded[:,k][:,CELL]).sum())))
        report[d]=dict(encoded=token_stats(arr['encoded'],lengths),generated=token_stats(arr['codes'],lengths),
            reconstruction=rec,generation=gen,transitions=transitions,embedding_distances=dist)
    write(ROOT/'results/tokenizer.json',report)
    progress('supplementary_analysis_complete')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['all','primary','supplementary'],default='all'); args=parser.parse_args()
    if args.stage=='supplementary': finish()
    else: main(primary_only=args.stage=='primary')

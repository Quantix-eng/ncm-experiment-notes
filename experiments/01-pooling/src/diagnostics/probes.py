"""Linear prospective readouts. No step-9 observation enters feature construction."""
from base import *
from scipy.linalg import solve
from sklearn.metrics import f1_score,roc_auc_score
from threadpoolctl import threadpool_limits
import pickle
threadpool_limits(1)

def feature_blocks(a,global8,projection):
    prefix=a['prefix'];given=prefix[:,1:-1][:,CELL];n=len(prefix)
    prob=a['prob8'];old=a['decoded8'][:,CELL]
    candidate=np.zeros((n,81,9),dtype=np.float32)
    for b in range(n):
        for i in range(81):
            allowed=set(range(1,10))-{int(given[b,j]) for j in PEERS[i]}
            for d in allowed:candidate[b,i,d-1]=1
    coords=np.concatenate([np.repeat(np.eye(9),9,axis=0),np.tile(np.eye(9),(9,1))],axis=1)
    xy=np.broadcast_to(coords,(n,81,18))
    baseline=np.concatenate([prob,np.eye(12)[old],candidate,xy,np.broadcast_to(global8[:,None,:],(n,81,20))],-1).astype(np.float64)
    state=np.concatenate([a['z8'],np.broadcast_to((a['h8']@projection)[:,None,:],(n,81,64))],-1).astype(np.float64)
    return baseline,state,xy,given==0

def labels(a):
    old=a['decoded8'][:,CELL];new=a['decoded9'][:,CELL];ref=a['reference'][:,CELL]
    changed=old!=new;before=old==ref;after=new==ref
    quality=np.zeros_like(old);quality[changed&~before&after]=1;quality[changed&before&~after]=2;quality[changed&~before&~after]=3
    assert np.all(~changed|(quality!=0))
    return new,quality,changed

def fit(x,y,mask,train,val,nclasses,forbidden=None):
    # Fit statistics and moments on train only; all rows of a source puzzle stay together.
    tx=x[train][mask[train]];ty=y[train][mask[train]];vx=x[val][mask[val]];vy=y[val][mask[val]]
    mean=tx.mean(0);scale=tx.std(0);scale[scale<1e-12]=1
    xx=(tx-mean)/scale;yy=np.eye(nclasses)[ty];intercept=yy.mean(0)
    gram=xx.T@xx;rhs=xx.T@(yy-intercept);vz=(vx-mean)/scale;best=None;trials=[]
    for alpha in Q['ridge_alpha']:
        w=solve(gram+np.eye(len(mean))*alpha,rhs,assume_a='pos',check_finite=False)
        score=vz@w+intercept
        if forbidden is not None:
            banned=forbidden[val][mask[val]];score[np.arange(len(banned)),banned]=-1e6
        for temp in Q['temperature']:
            p=softmax(score,temp);loss=float(-np.log(np.maximum(p[np.arange(len(vy)),vy],1e-15)).mean())
            trials.append(dict(alpha=alpha,temperature=temp,validation_logloss=loss))
            if best is None or loss<best[0]:best=(loss,alpha,temp,w.copy())
    loss,alpha,temp,w=best
    return dict(mean=mean,scale=scale,weight=w,intercept=intercept,temperature=temp,
                fit=dict(alpha=alpha,temperature=temp,validation_logloss=loss,n_train_puzzles=len(train),n_train_cells=len(ty),
                         train_class_counts=np.bincount(ty,minlength=nclasses),trials=trials))

def predict(m,x,forbidden=None):
    shape=x.shape[:-1];flat=x.reshape(-1,x.shape[-1]);score=((flat-m['mean'])/m['scale'])@m['weight']+m['intercept']
    if forbidden is not None:score[np.arange(len(flat)),np.asarray(forbidden).ravel()]=-1e6
    return softmax(score,m['temperature']).reshape(*shape,-1)

def metrics(y,p,mask,quality=False):
    t=y[mask];pp=p[mask];pred=pp.argmax(-1)
    out=dict(n_puzzles=len(y),n_cells=len(t),accuracy=float((pred==t).mean()),logloss=float(-np.log(np.maximum(pp[np.arange(len(t)),t],1e-15)).mean()),
             macro_f1=float(f1_score(t,pred,labels=np.arange(p.shape[-1]),average='macro',zero_division=0)))
    if quality:
        out['corrected_auc']=float(roc_auc_score(t==1,pp[:,1])) if len(np.unique(t==1))==2 else None
        out['class_counts']=np.bincount(t,minlength=4)
    return out

def ratio_ci(numer,denom):
    numer=np.asarray(numer,dtype=float);denom=np.asarray(denom,dtype=float);rng=np.random.default_rng(Q['seed']);ix=rng.integers(0,len(numer),(Q['bootstrap'],len(numer)))
    ds=denom[ix].sum(1);v=numer[ix].sum(1)[ds>0]/ds[ds>0]
    return dict(delta=float(numer.sum()/denom.sum()),ci95=np.quantile(v,[.025,.975]),n_puzzles=len(numer),method='paired ratio bootstrap by source puzzle')

def main():
    start=time.perf_counter();official=np.load(D/'results/revision_official.npz');projection=np.random.default_rng(Q['seed']).normal(size=(1024,64))/8
    cached=np.load(ROOT/'results/hard.npz');b,s,xy,mask=feature_blocks(official,cached['baseline'][:,7],projection)
    features=dict(baseline=b,state=np.concatenate([s,xy],-1),combined=np.concatenate([b,s],-1))
    tr,va,te=[np.array(SPLIT[n]) for n in ['train','validation','test']];nxt,quality,changed=labels(official)
    models={};report={};predictions={};rng=np.random.default_rng(Q['seed']);permutation=rng.permutation(tr)
    # Negative control permutes only train states, retaining the original baseline.
    shuffled=features['combined'].copy();shuffled[tr,:,b.shape[-1]:]=s[permutation]
    features['shuffled_states']=shuffled
    for task,y,nclasses in [('next',nxt,12),('quality',quality,4)]:
        result={};taskpred={}
        for name,x in features.items():
            model=fit(x,y,mask,tr,va,nclasses);pp=predict(model,x[te]);models[task+'_'+name]=model
            taskpred[name]=pp;predictions[task+'_'+name]=pp
            result[name]=dict(all=metrics(y[te],pp,mask[te],quality=task=='quality'),fit=model['fit'])
            if task=='next':result[name]['changed_only']=metrics(y[te],pp,mask[te]&changed[te])
            log('prospective_probe',task=task,features=name,test=result[name]['all'])
        if task=='next':
            cmask=mask[te]&changed[te];ca=taskpred['combined'].argmax(-1)==y[te];ba=taskpred['baseline'].argmax(-1)==y[te]
            result['primary_changed_contrast']=ratio_ci(((ca.astype(float)-ba)*cmask).sum(1),cmask.sum(1))
            result['all_contrast']=ratio_ci(((ca.astype(float)-ba)*mask[te]).sum(1),mask[te].sum(1))
            old=official['decoded8'][te][:,CELL];p8=official['prob8'][te].copy();p8[np.arange(len(te))[:,None],np.arange(81)[None,:],old]=-np.inf
            second=p8.argmax(-1);result['second_decoder_choice_changed_accuracy']=float((second==y[te])[cmask].mean())
            result['persistence_all_accuracy']=float((old==y[te])[mask[te]].mean())
        report[task]=result
    metadata=dict(baseline_dim=b.shape[-1],state_local_dim=256,global_projection=projection,features='prob8 + current onehot + given candidates + row/col coordinates + observable global8; then local z8 and projected h8')
    with (D/'results/probe_models.pkl').open('wb') as f:pickle.dump(dict(models=models,metadata=metadata),f)
    np.savez_compressed(D/'results/probe_predictions_test.npz',indices=te,prefix=official['prefix'][te],reference=official['reference'][te],next_target=nxt[te],quality_target=quality[te],changed=changed[te],blank=mask[te],**predictions)
    write(D/'results/probes.json',dict(results=report,primary=report['next']['primary_changed_contrast'],seconds=time.perf_counter()-start,
       note='Changed-only is a future-defined evaluation stratum, not an oracle available to a deployed predictor. Classes absent from data count as zero in 12-class macro F1.'))
    ext=np.load(D/'results/revision_external.npz');ecached=np.load(ROOT/'results/external.npz');eb,es,exy,em=feature_blocks(ext,ecached['baseline'][:,7],projection)
    ef=dict(baseline=eb,state=np.concatenate([es,exy],-1),combined=np.concatenate([eb,es],-1),shuffled_states=np.concatenate([eb,es],-1));en,eq,ec=labels(ext)
    er={};epreds={}
    for task,y in [('next',en),('quality',eq)]:
        er[task]={}
        for name,x in ef.items():
            pp=predict(models[task+'_'+name],x);epreds[task+'_'+name]=pp;er[task][name]=dict(all=metrics(y,pp,em,quality=task=='quality'))
            if task=='next':er[task][name]['changed_only']=metrics(y,pp,em&ec)
        if task=='next':
            cp=epreds['next_combined'].argmax(-1)==y;bp=epreds['next_baseline'].argmax(-1)==y;mm=em&ec
            er[task]['changed_contrast']=ratio_ci(((cp.astype(float)-bp)*mm).sum(1),mm.sum(1))
    np.savez_compressed(D/'results/probe_predictions_external.npz',next_target=en,quality_target=eq,blank=em,changed=ec,**epreds)
    write(D/'results/external_probes.json',er);log('prospective_probes_complete',primary=report['next']['primary_changed_contrast'],external=er['next']['changed_contrast'])

if __name__=='__main__':main()

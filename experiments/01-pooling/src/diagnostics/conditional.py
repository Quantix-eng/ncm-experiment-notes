"""Exploratory amendment: prospective alternatives conditional on a revision."""
from base import *
from probes import feature_blocks,labels,fit,predict,metrics,ratio_ci
from sklearn.metrics import roc_auc_score,average_precision_score
import pickle

def blocks(a,global8,projection):
    b,s,xy,mask=feature_blocks(a,global8,projection);p=a['prob8'];old=a['decoded8'][:,CELL]
    sort=np.sort(p,axis=-1);ent=-(p*np.log(np.maximum(p,1e-30))).sum(-1);chosen=np.take_along_axis(p,old[...,None],-1)[...,0]
    extra=np.stack([ent,sort[:,:,-1],sort[:,:,-2],sort[:,:,-1]-sort[:,:,-2],chosen,b[:,:,24:33].sum(-1)/9],-1)
    b=np.concatenate([b,extra],-1)
    return b,s,xy,mask

def joint(gate,alt,old):
    p=gate[...,1,None]*alt
    p[np.arange(len(old))[:,None],np.arange(81)[None,:],old]+=gate[...,0]
    assert np.allclose(p.sum(-1),1)
    return p

def main():
    a=np.load(D/'results/revision_official.npz');old=np.load(ROOT/'results/hard.npz');proj=np.random.default_rng(Q['seed']).normal(size=(1024,64))/8
    b,s,xy,blank=blocks(a,old['baseline'][:,7],proj);y,quality,change=labels(a);current=a['decoded8'][:,CELL]
    features=dict(baseline=b,state=np.concatenate([s,xy],-1),combined=np.concatenate([b,s],-1));tr,va,te=[np.array(SPLIT[n]) for n in ['train','validation','test']]
    sh=features['combined'].copy();sh[tr,:,b.shape[-1]:]=s[np.random.default_rng(Q['seed']).permutation(tr)];features['shuffled_states']=sh
    models={};report={};pred={}
    for name,x in features.items():
        alt=fit(x,y,blank&change,tr,va,12,forbidden=current);gate=fit(x,change.astype(int),blank,tr,va,2)
        pa=predict(alt,x[te],forbidden=current[te]);pg=predict(gate,x[te]);pj=joint(pg,pa,current[te]);models[name]=dict(alternative=alt,gate=gate)
        pred[name+'_alternative']=pa;pred[name+'_gate']=pg;pred[name+'_joint']=pj
        report[name]=dict(changed_only=metrics(y[te],pa,blank[te]&change[te]),joint_all=metrics(y[te],pj,blank[te]),
           gate_auc=float(roc_auc_score(change[te][blank[te]],pg[:,:,1][blank[te]])),
           gate_average_precision=float(average_precision_score(change[te][blank[te]],pg[:,:,1][blank[te]])),
           fit_alternative=alt['fit'],fit_gate=gate['fit'])
        log('conditional_probe',features=name,changed_accuracy=report[name]['changed_only']['accuracy'],gate_auc=report[name]['gate_auc'])
    mm=blank[te]&change[te];cp=pred['combined_alternative'].argmax(-1)==y[te];bp=pred['baseline_alternative'].argmax(-1)==y[te]
    p8=a['prob8'][te].copy();p8[np.arange(len(te))[:,None],np.arange(81)[None,:],current[te]]=-1;second=p8.argmax(-1);sp=second==y[te]
    contrasts=dict(combined_minus_fitted_baseline=ratio_ci(((cp.astype(float)-bp)*mm).sum(1),mm.sum(1)),
      combined_minus_decoder_second=ratio_ci(((cp.astype(float)-sp)*mm).sum(1),mm.sum(1)),second_accuracy=float(sp[mm].mean()))
    meta=dict(global_projection=proj,baseline_dim=b.shape[-1],state_local_dim=256)
    with (D/'results/conditional_models.pkl').open('wb') as f:pickle.dump(dict(models=models,metadata=meta),f)
    np.savez_compressed(D/'results/conditional_predictions_test.npz',indices=te,target=y[te],blank=blank[te],changed=change[te],**pred)
    write(D/'results/conditional_probes.json',dict(results=report,contrasts=contrasts,amendment_sha256=sha(D/'AMENDEMENT_CONDITIONNEL.md')))
    ea=np.load(D/'results/revision_external.npz');ecached=np.load(ROOT/'results/external.npz');eb,es,exy,em=blocks(ea,ecached['baseline'][:,7],proj);en,eq,ec=labels(ea);eo=ea['decoded8'][:,CELL]
    ef=dict(baseline=eb,state=np.concatenate([es,exy],-1),combined=np.concatenate([eb,es],-1),shuffled_states=np.concatenate([eb,es],-1));er={};ep={}
    for name,x in ef.items():
        pa=predict(models[name]['alternative'],x,forbidden=eo);pg=predict(models[name]['gate'],x);pj=joint(pg,pa,eo)
        er[name]=dict(changed_only=metrics(en,pa,em&ec),joint_all=metrics(en,pj,em),gate_auc=float(roc_auc_score(ec[em],pg[:,:,1][em])))
        ep[name+'_alternative']=pa;ep[name+'_gate']=pg;ep[name+'_joint']=pj
    mm=em&ec;cp=ep['combined_alternative'].argmax(-1)==en;bp=ep['baseline_alternative'].argmax(-1)==en
    p8=ea['prob8'].copy();p8[np.arange(len(eo))[:,None],np.arange(81)[None,:],eo]=-1;sp=p8.argmax(-1)==en
    er['contrasts']=dict(combined_minus_fitted_baseline=ratio_ci(((cp.astype(float)-bp)*mm).sum(1),mm.sum(1)),
      combined_minus_decoder_second=ratio_ci(((cp.astype(float)-sp)*mm).sum(1),mm.sum(1)),second_accuracy=float(sp[mm].mean()))
    np.savez_compressed(D/'results/conditional_predictions_external.npz',target=en,blank=em,changed=ec,**ep);write(D/'results/conditional_external.json',er)
    log('conditional_complete',contrasts=contrasts,external=er['contrasts'])

if __name__=='__main__':main()

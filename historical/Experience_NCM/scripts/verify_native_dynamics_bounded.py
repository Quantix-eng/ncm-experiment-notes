"""Bounded-memory independent replay of native study, no torch, inference or fitting."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/metacognition'))
from features import CELL_POS,observed_counts
BASE=ROOT/'audit/metacognition/native-dynamics-v1'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb')as handle:
        for chunk in iter(lambda:handle.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def read(path):return json.loads(path.read_text())
def ci(x,seed,coverage=.975):
    rng=np.random.default_rng(seed);a=(1-coverage)/2
    boot=np.array([rng.choice(x,len(x),replace=True).mean()for _ in range(10000)])
    return np.quantile(boot,[a,1-a])

def direct_auc(y,s):
    diff=s[y==1,None]-s[None,y==0]
    return float(((diff>0)+.5*(diff==0)).mean())

def main():
    lock=read(BASE/'lock.json');checked=0
    for p,h in lock['source_files'].items():assert digest(ROOT/p)==h,p;checked+=1
    assert digest(BASE/'fit.json')==lock['fit_sha256']
    assert digest(BASE/'external/manifest.json')==lock['corpus_manifest_sha256']
    manifest=read(BASE/'external/manifest.json');assert digest(BASE/'external/corpus.npz')==manifest['corpus_sha256']
    for name,h in lock['models'].items():assert digest(BASE/'models'/name)==h;checked+=1
    extraction=read(BASE/'external/traces/report.json')
    for name,h in extraction['arrays'].items():assert digest(BASE/'external/traces'/name)==h;checked+=1
    corpus=np.load(BASE/'external/corpus.npz',allow_pickle=False)
    report=read(BASE/'external/analysis/report.json');assert digest(BASE/'external/analysis/predictions.npz')==report['predictions_sha256']
    data={p.stem:np.load(p,mmap_mode='r',allow_pickle=False)for p in (BASE/'external/traces').glob('*.npy')}
    scores=np.load(BASE/'external/analysis/predictions.npz',allow_pickle=False);rows=scores['rows'];group=scores['group'];y=scores['persistent']
    assert np.array_equal(data['group'][rows],group)and np.array_equal(data['persistent'][rows],y)
    assert data['current_error'][rows].all()
    permutation=np.arange(len(data['group']))
    for g in np.unique(data['group']):
        r=np.flatnonzero(data['group']==g)
        permutation[r]=np.random.default_rng(np.random.SeedSequence([20261206,int(g)])).permutation(r)
    aucs={};maximum=0
    for model in ['visible','static','dynamic','noise','shuffle']:
        m=np.load(BASE/f'models/{model}.npz',allow_pickle=False)
        def scale(pre,x):
            keep=m[pre+'_keep'];return (x[:,keep]-m[pre+'_mean'][keep])/m[pre+'_scale']
        def pca(pre,x):return (scale(pre+'_hidden',x.astype(float))@m[pre+'_pca_axes'].T-m[pre+'_pca_mean'])/m[pre+'_pca_scale']
        for start in range(0,len(rows),1024):
            r=rows[start:start+1024];x=scale('base',data['visible'][r])
            if model!='visible':x=np.column_stack([x,scale('amplitude',data['amplitude'][r]),pca('static',data['static'][r])])
            if model in ['dynamic','shuffle']:
                h=r if model=='dynamic'else permutation[r]
                x=np.column_stack([x,pca('delta',data['delta'][h])])
            if model=='noise':
                ns=np.empty((len(r),16));g=data['group'][r]
                for value in np.unique(g):
                    mask=g==value
                    ns[mask]=np.random.default_rng(np.random.SeedSequence([20261205,int(value)])).standard_normal((81,16))[data['cell'][r[mask]]]
                x=np.column_stack([x,scale('noise',ns)])
            replay=x@m['weight']+m['intercept']
            error=float(np.max(abs(replay-scores[model][start:start+len(r)])))
            assert error<1e-9;maximum=max(maximum,error)
        aucs[model]={}
        for g in np.unique(group):
            mask=group==g
            if len(np.unique(y[mask]))==2:aucs[model][int(g)]=direct_auc(y[mask],scores[model][mask])
        assert np.isclose(np.mean(list(aucs[model].values())),report['results'][model]['within_puzzle_auc'],atol=1e-14)
        print('bounded score replay',model,flush=True)
    for key,v in report['contrasts'].items():
        a,b=key.split('-');delta=np.array([aucs[a][g]-aucs[b][g]for g in sorted(aucs[a])])
        assert np.isclose(delta.mean(),v['delta'],atol=1e-14)
        assert np.allclose(ci(delta,20261202),v['interval_97_5'],rtol=0,atol=1e-12)
    causal=read(BASE/'causal/report.json');assert digest(BASE/'causal/outcomes.npz')==causal['outcomes_sha256']
    for name,h in causal['latent_checks'].items():assert digest(BASE/'causal'/name)==h;checked+=1
    out=np.load(BASE/'causal/outcomes.npz',allow_pickle=False);puzzles,gold=corpus['puzzles'][:1500],corpus['solutions'][:1500]
    assert np.array_equal(out['group'],corpus['groups'][:1500])
    latent_errors={}
    for name in ['runner_up','matched']:
        for variant in ['active','fixed']:
            ids=out[name+'_'+variant+'_ids'];_,_,_,_,valid=observed_counts(ids,puzzles)
            exact=(ids[:,CELL_POS]==gold).all(1)
            assert not (valid&~exact).any(),'Valid solution differs from unique reference'
            assert np.array_equal(valid&exact,out[name+'_'+variant+'_success'])
            n=int(out[name+'_'+variant+'_success'].sum());assert n==causal['results'][name][variant+'_resolutions']
        delta=out[name+'_active_success'].astype(int)-out[name+'_fixed_success'].astype(int)
        result=causal['results'][name]
        assert np.isclose(delta.mean(),result['resolution_gain'],atol=1e-14)
        assert int((delta==1).sum())==result['adaptive_only']and int((delta==-1).sum())==result['fixed_only']
        assert np.allclose(ci(delta,20261204),result['interval_97_5'],rtol=0,atol=1e-12)
        with np.load(BASE/f'causal/latent-check-{name}.npz',allow_pickle=False)as z:
            e,r=z['initial_effect'],z['future_response']
            assert np.array_equal(r,(z['active_final'].astype(float)-z['fixed_final'].astype(float)).reshape(len(e),-1))
            compensation=-(e*r).sum(1)/(e*e).sum(1)
            assert np.allclose(compensation,out[name+'_compensation'][:len(e)],atol=1e-12)
            rounding=float(np.max(abs((z['fixed_final'].astype(float)-z['clean_final'].astype(float)).reshape(len(e),-1)-e)))
            latent_errors[name]=rounding;assert rounding<1e-5
        for metric,seed in [('compensation',20261208)]:
            assert np.allclose(ci(out[name+'_'+metric],seed,.95),result[metric+'_interval_95'],rtol=0,atol=1e-12)
    assert lock['utc']<extraction['utc']and lock['utc']<causal['utc']
    result=dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),checked_files=checked,
        max_abs_score_replay_error=maximum,block_size=1024,auc_direct_pairwise_recomputed=True,bootstrap_recomputed=True,
        strict_causal_success_recomputed=True,latent_compensation_sample_recomputed=True,fixed_tail_rounding_max_abs=latent_errors,
        source_sha256=digest(Path(__file__)),report_sha256=digest(BASE/'external/analysis/report.json'),
        causal_report_sha256=digest(BASE/'causal/report.json'),fitting=False,model_execution=False,
        caveat='Independent arithmetic implementation in same environment; not replication by another team. Bounded variant added for system resource pressure, primary frozen study unchanged.')
    (BASE/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()

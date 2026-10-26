"""Independent arithmetic audit of saved outcomes; no NCM execution or fitting."""
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from run_pilot import ROOT,file_hash,write_json
from features import CELL_POS,observed_counts
from confirmation_corpus import known_transform_key

BASE=ROOT/'audit/metacognition/native-dynamics-v1'

def direct_auc(y,s):
    d=s[y==1,None]-s[None,y==0]
    return float(((d>0)+.5*(d==0)).mean())

def interval(x,seed,coverage=.975):
    rng=np.random.default_rng(seed)
    means=[rng.choice(x,len(x),replace=True).mean()for _ in range(10000)]
    a=(1-coverage)/2
    return np.quantile(means,[a,1-a])

def main():
    lock=json.loads((BASE/'lock.json').read_text());checked=0
    for relative,h in lock['source_files'].items():assert file_hash(ROOT/relative)==h;checked+=1
    for n,h in lock['models'].items():assert file_hash(BASE/'models'/n)==h;checked+=1
    manifest=json.loads((BASE/'external/manifest.json').read_text())
    assert file_hash(BASE/'external/corpus.npz')==manifest['corpus_sha256']
    corpus=np.load(BASE/'external/corpus.npz',allow_pickle=False)
    assert len(set(known_transform_key(s.tolist())for s in corpus['solutions']))==2000
    extraction=json.loads((BASE/'external/traces/report.json').read_text())
    for n,h in extraction['arrays'].items():assert file_hash(BASE/'external/traces'/n)==h;checked+=1
    report=json.loads((BASE/'external/analysis/report.json').read_text())
    assert file_hash(BASE/'external/analysis/predictions.npz')==report['predictions_sha256']
    scores=np.load(BASE/'external/analysis/predictions.npz',allow_pickle=False);g=scores['group'];y=scores['persistent']
    # Reimplement scaling/PCA replay from coefficients, without the fitting/transform routine.
    data={p.stem:np.load(p,mmap_mode='r',allow_pickle=False)for p in (BASE/'external/traces').glob('*.npy')}
    rows=scores['rows'];maximum=0;aucs={}
    for name in ['visible','static','dynamic','noise','shuffle']:
        saved=np.load(BASE/f'models/{name}.npz',allow_pickle=False)
        def scale(pre,x):
            keep=saved[pre+'_keep'];return (x[:,keep]-saved[pre+'_mean'][keep])/saved[pre+'_scale']
        def pca(pre,x):return (scale(pre+'_hidden',x.astype(float))@saved[pre+'_pca_axes'].T-saved[pre+'_pca_mean'])/saved[pre+'_pca_scale']
        x=scale('base',data['visible'][rows])
        if name!='visible':x=np.column_stack([x,scale('amplitude',data['amplitude'][rows]),pca('static',data['static'][rows])])
        if name in ['dynamic','shuffle']:
            source=rows.copy()
            if name=='shuffle':
                permutation=np.arange(len(data['group']))
                for group in np.unique(data['group']):
                    r=np.flatnonzero(data['group']==group)
                    permutation[r]=np.random.default_rng(np.random.SeedSequence([20261206,int(group)])).permutation(r)
                source=permutation[rows]
            x=np.column_stack([x,pca('delta',data['delta'][source])])
        if name=='noise':
            noise=np.empty((len(rows),16))
            for group in np.unique(g):
                r=g==group
                noise[r]=np.random.default_rng(np.random.SeedSequence([20261205,int(group)])).standard_normal((81,16))[data['cell'][rows[r]]]
            x=np.column_stack([x,scale('noise',noise)])
        replay=x@saved['weight']+saved['intercept'];error=float(np.max(abs(replay-scores[name])))
        assert error<1e-9;maximum=max(maximum,error)
        aucs[name]={}
        for group in np.unique(g):
            r=g==group
            if len(np.unique(y[r]))==2:aucs[name][int(group)]=direct_auc(y[r],scores[name][r])
        assert np.isclose(np.mean(list(aucs[name].values())),report['results'][name]['within_puzzle_auc'],atol=1e-14)
    for contrast,result in report['contrasts'].items():
        a,b=contrast.split('-');delta=np.array([aucs[a][k]-aucs[b][k]for k in sorted(aucs[a])])
        assert np.isclose(delta.mean(),result['delta'],atol=1e-14)
        assert np.array_equal(interval(delta,20261202),result['interval_97_5'])
    causal=json.loads((BASE/'causal/report.json').read_text())
    assert file_hash(BASE/'causal/outcomes.npz')==causal['outcomes_sha256']
    for n,h in causal['latent_checks'].items():assert file_hash(BASE/'causal'/n)==h;checked+=1
    outcomes=np.load(BASE/'causal/outcomes.npz',allow_pickle=False)
    puzzles,gold=corpus['puzzles'][:1500],corpus['solutions'][:1500]
    for name in ['runner_up','matched']:
        for variant in ['active','fixed']:
            ids=outcomes[f'{name}_{variant}_ids'];_,_,_,_,valid=observed_counts(ids,puzzles)
            success=valid&((ids[:,CELL_POS]==gold).all(1))
            assert np.array_equal(success,outcomes[f'{name}_{variant}_success'])
        delta=outcomes[name+'_active_success'].astype(int)-outcomes[name+'_fixed_success'].astype(int)
        r=causal['results'][name]
        assert np.isclose(delta.mean(),r['resolution_gain'],atol=1e-14)
        assert np.array_equal(interval(delta,20261204),r['interval_97_5'])
        with np.load(BASE/f'causal/latent-check-{name}.npz',allow_pickle=False)as z:
            e=z['initial_effect'];response=z['future_response']
            assert np.array_equal(response,(z['active_final'].astype(float)-z['fixed_final'].astype(float)).reshape(len(e),-1))
            compensation=-(e*response).sum(1)/(e*e).sum(1)
            assert np.allclose(compensation,outcomes[name+'_compensation'][:len(e)],atol=1e-12)
    assert lock['utc']<extraction['utc'] and lock['utc']<causal['utc']
    result=dict(status='passed',utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        checked_files=checked,max_abs_score_replay_error=maximum,auc_direct_pairwise_recomputed=True,
        bootstrap_recomputed=True,strict_causal_success_recomputed=True,latent_compensation_sample_recomputed=True,
        lock_precedes_completed_external_inference=True,model_execution=False,fitting=False,
        source_sha256=file_hash(Path(__file__)),report_sha256=file_hash(BASE/'external/analysis/report.json'),
        causal_report_sha256=file_hash(BASE/'causal/report.json'),caveat='Arithmetic audit in same environment, not independent team replication')
    write_json(BASE/'verification.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()

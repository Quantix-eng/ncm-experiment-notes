"""Fit on 799 original TRAINING groups, evaluate on its untouched fold0."""
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from probes import stratified_folds
from run_pilot import ROOT,file_hash,write_json
from learning import controls,fit_logistic,grouped_metrics,prepare

HERE=Path(__file__).resolve().parent
BASE=ROOT/'audit/metacognition/correction-v0.1'


def main():
    torch.set_num_threads(4);torch.manual_seed(20261101)
    output=BASE/'development';output.mkdir(exist_ok=False)
    protocol=json.loads((HERE/'protocol-v0.1.json').read_text())
    extraction=json.loads((BASE/'training/report.json').read_text())
    assert extraction['status']=='complete' and extraction['protocol_sha256']==file_hash(HERE/'protocol-v0.1.json')
    for name,entry in extraction['arrays'].items():assert file_hash(BASE/'training'/name)==entry['sha256']
    data={key:np.load(BASE/f'training/{key}.npy',mmap_mode='r',allow_pickle=False) for key in
        ['visible','hidden','group','cell','current_error','persistent']}
    split=json.loads((ROOT/'audit/metacognition/split-v0.2.json').read_text())
    groups=np.unique(data['group'])
    assert groups.tolist()==split['partition_indices']['train']
    held=np.array([g for g in groups if split['train_cv_fold'][str(g)]==0])
    pool=np.array([g for g in groups if split['train_cv_fold'][str(g)]!=0])
    mixed=np.array([int(np.any(data['current_error'][data['group']==g]&data['persistent'][data['group']==g]))for g in pool])
    cv=stratified_folds(mixed,20261101)
    noise,shuffle=controls(data)
    models=protocol['models'];grid=protocol['lambda_grid']
    report=dict(status='running',scope='original_training_only_799_fit_201_development_holdout',protocol_sha256=file_hash(HERE/'protocol-v0.1.json'),
        source_files={name:file_hash(HERE/name)for name in ['run_development.py','learning.py','cell_features.py','extract_training.py']},
        pool_groups=pool.tolist(),holdout_groups=held.tolist(),inner_seed=20261101,cv={},fits=[],selection={},models=models)
    write_json(output/'report.json',report);start=time.perf_counter()
    try:
        values={m:{s:[] for s in grid}for m in models}
        for k in range(5):
            train,valid=pool[cv!=k],pool[cv==k]
            report['cv'][str(k)]=dict(train=train.tolist(),valid=valid.tolist())
            matrices,t,v=prepare(data,train,valid,noise,shuffle)
            for name,(a,b,_)in matrices.items():
                for strength in grid:
                    fit=fit_logistic(a,data['persistent'][t],data['group'][t],strength,protocol['optimizer'])
                    score=b@fit['weight']+fit['intercept']
                    metric=grouped_metrics(data['persistent'][v],score,data['group'][v])
                    assert metric['within_puzzle_auc']is not None
                    values[name][strength].append(metric['within_puzzle_auc'])
                    report['fits'].append(dict(fold=k,model=name,strength=strength,within_auc=metric['within_puzzle_auc'],
                        log_loss=metric['weighted_log_loss'],iterations=fit['iterations'],max_abs_gradient=fit['max_abs_gradient']))
            print(f'correction grouped CV {k+1}/5 complete',flush=True)
        matrices,t,v=prepare(data,pool,held,noise,shuffle)
        scores={}
        for name,(a,b,preprocessing)in matrices.items():
            means={s:float(np.mean(values[name][s]))for s in grid}
            strength=max(s for s in grid if means[s]>=max(means.values())-1e-8)
            fit=fit_logistic(a,data['persistent'][t],data['group'][t],strength,protocol['optimizer'])
            scores[name]=b@fit['weight']+fit['intercept']
            np.savez_compressed(output/f'model-{name}.npz',weight=fit['weight'],intercept=np.array(fit['intercept']),**preprocessing)
            report['selection'][name]=dict(strength=strength,cv_mean_within_auc=means,iterations=fit['iterations'],max_abs_gradient=fit['max_abs_gradient'])
        y,g=data['persistent'][v],data['group'][v]
        result={m:grouped_metrics(y,s,g)for m,s in scores.items()}
        a={r['group']:r['auc']for r in result['visible']['per_group']}
        b={r['group']:r['auc']for r in result['visible_hidden']['per_group']}
        paired=np.array([b[k]-a[k]for k in sorted(a)])
        rng=np.random.default_rng(20261105)
        delta=np.array([rng.choice(paired,len(paired),replace=True).mean()for _ in range(5000)])
        np.savez_compressed(output/'predictions.npz',group=g,cell=data['cell'][v],persistent=y,rows=v,**scores)
        report.update(status='complete',results=result,primary_delta_within_auc=float(paired.mean()),
            paired_group_bootstrap_95=np.quantile(delta,[.025,.975]).tolist(),mixed_holdout_groups=len(paired),
            wall_seconds=time.perf_counter()-start,external_test_used=False,
            caveat='Oracle-defined current-error cohort; development holdout, not independent confirmation; group bootstrap conditional on fixed probes',
            artifacts={p.name:file_hash(p)for p in output.glob('*.npz')})
        for name,digest in report['source_files'].items():assert file_hash(HERE/name)==digest
    except Exception:
        report.update(status='failed',error=traceback.format_exc());write_json(output/'report.json',report);raise
    write_json(output/'report.json',report)
    print(json.dumps(dict(status=report['status'],results={m:r['within_puzzle_auc']for m,r in result.items()},
        delta=report['primary_delta_within_auc'],interval=report['paired_group_bootstrap_95'],mixed_groups=len(paired))))

if __name__=='__main__':main()

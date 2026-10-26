"""Numerical replay of cell probes without fitting or model execution."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_pilot import ROOT,file_hash,write_json
from learning import grouped_metrics

BASE=ROOT/'audit/metacognition/correction-v0.1'


def transform(saved,data,name,rows,shuffle=None):
    keep=saved['base_keep']
    x=(data['visible'][rows][:,keep]-saved['base_mean'][keep])/saved['base_scale']
    if name in ('visible_hidden','visible_shuffle'):
        source=shuffle[rows]if name=='visible_shuffle'else rows
        h=data['hidden'][source].astype(np.float64)
        keep=saved['hidden_keep']
        h=(h[:,keep]-saved['hidden_mean'][keep])/saved['hidden_scale']
        z=(h@saved['pca_axes'].T-saved['pca_mean'])/saved['pca_scale']
        x=np.column_stack([x,z])
    elif name=='visible_noise':
        noise=np.empty((len(rows),16))
        g=data['group'][rows];cells=data['cell'][rows]
        for group in np.unique(g):
            selected=g==group
            noise[selected]=np.random.default_rng(np.random.SeedSequence([20261104,int(group)])).standard_normal((81,16))[cells[selected]]
        keep=saved['noise_keep']
        z=(noise[:,keep]-saved['noise_mean'][keep])/saved['noise_scale']
        x=np.column_stack([x,z])
    return x


def shuffled_rows(groups):
    shuffle=np.arange(len(groups))
    for group in np.unique(groups):
        rows=np.flatnonzero(groups==group)
        shuffle[rows]=np.random.default_rng(np.random.SeedSequence([20261103,int(group)])).permutation(rows)
    return shuffle


def check_scores(report,scores):
    all_metrics={m:grouped_metrics(scores['persistent'],scores[m],scores['group'])for m in report['models']}
    for model,result in all_metrics.items():
        expected=report['results'][model]
        for key in ('auc','within_puzzle_auc','weighted_log_loss','weighted_brier'):
            assert np.isclose(result[key],expected[key],rtol=1e-10,atol=1e-12)
    a={r['group']:r['auc']for r in all_metrics['visible']['per_group']}
    b={r['group']:r['auc']for r in all_metrics['visible_hidden']['per_group']}
    paired=np.array([b[k]-a[k]for k in sorted(a)])
    rng=np.random.default_rng(20261105)
    delta=[rng.choice(paired,len(paired),replace=True).mean()for _ in range(5000)]
    assert np.isclose(paired.mean(),report['primary_delta_within_auc'])
    assert np.array_equal(np.quantile(delta,[.025,.975]),report['paired_group_bootstrap_95'])


def main():
    directory=BASE/'development';report=json.loads((directory/'report.json').read_text())
    assert report['status']=='complete'
    for name,digest in report['artifacts'].items():assert file_hash(directory/name)==digest
    for name,digest in report['source_files'].items():assert file_hash(Path(__file__).with_name(name))==digest
    data={key:np.load(BASE/f'training/{key}.npy',mmap_mode='r',allow_pickle=False)for key in
        ['visible','hidden','group','cell','current_error','persistent']}
    shuffle=shuffled_rows(data['group']);maximum=0.
    with np.load(directory/'predictions.npz',allow_pickle=False)as scores:
        rows=scores['rows'];assert set(scores['group'])<=set(report['holdout_groups'])
        assert not set(scores['group'])&set(report['pool_groups'])
        assert data['current_error'][rows].all()
        for model in report['models']:
            with np.load(directory/f'model-{model}.npz',allow_pickle=False)as saved:
                x=transform(saved,data,model,rows,shuffle)
                replay=x@saved['weight']+saved['intercept']
            maximum=max(maximum,float(np.max(abs(replay-scores[model]))))
            assert np.allclose(replay,scores[model],rtol=1e-12,atol=1e-10)
        check_scores(report,scores)
    output=dict(status='passed',report_sha256=file_hash(directory/'report.json'),max_abs_replay_error=maximum,
        fitting=False,model_execution=False,paired_group_bootstrap_recomputed=True)
    write_json(directory/'verification.json',output);print(json.dumps(output))

if __name__=='__main__':main()

"""Read-only numerical replay of early-control feasibility, no fits or inference."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'audit/metacognition/early-control-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def direct_auc(y,s):
    a=s[y==1];b=s[y==0]
    return float(np.mean((a[:,None]>b[None,:])+.5*(a[:,None]==b[None,:])))
def main():
    dev=read(BASE/'development.json');r=read(BASE/'profiling/report.json');split=read(ROOT/'audit/metacognition/split-v0.2.json')
    tech=read(BASE/'technical.json');assert tech['status']=='passed' and tech['n']==32
    engine='experiments/metacognition/early_control/engine.py'
    assert tech['source_files'][engine]==r['source_files'][engine]
    assert tech['model_state_sha256_before_and_after']==r['model_state_sha256_before_and_after']
    assert dev['scores_sha256']==sha(BASE/'development-scores.npz')
    assert r['development_sha256']==sha(BASE/'development.json')
    for rel,h in r['source_files'].items():assert sha(ROOT/rel)==h,rel
    for name,h in dev['models'].items():assert sha(BASE/'models'/name)==h
    with np.load(BASE/'development-scores.npz',allow_pickle=False)as z:d={k:z[k].copy()for k in z.files}
    assert d['indices'].tolist()==split['partition_indices']['train']
    assert len(set(d['indices']))==1000 and d['failure'].sum()==137
    np.testing.assert_array_equal(d['fold'],[split['train_cv_fold'][str(i)]for i in d['indices']])
    results={}
    for name in dev['auc']:
        assert abs(direct_auc(d['failure'],d[name])-dev['auc'][name])<1e-12
        for allowance,front in dev['frontiers'][name].items():
            rejected=np.zeros(1000,dtype=bool)if front['threshold']is None else d[name]>front['threshold']
            assert rejected.sum()==front['rejected']
            assert ((d['failure']==0)&rejected).sum()==front['lost_resolutions']
            # Independently evaluate every attainable whole-score threshold, including ties.
            candidate_counts=[]
            for t in np.r_[np.inf,np.unique(d[name])]:
                mask=d[name]>t
                if ((d['failure']==0)&mask).sum()<=int(allowance):candidate_counts.append(int(mask.sum()))
            assert max(candidate_counts)==front['rejected']
        results[name]={'auc':dev['auc'][name],'frontiers':dev['frontiers'][name]}
    plain=[a for a in r['stage_records']if a['mode']=='none']
    shares=[a['tail']/(a['first9']+a['tail']+a['verify'])for a in plain]
    assert abs(np.median(shares)-r['stage_tail_share_median'])<1e-12
    gate=any(dev['frontiers'][n]['10']['rejected']/1000*max(shares)>=.05 for n in ['hidden','codes_hidden'])
    assert gate==r['promising_under_optimistic_gate']
    assert r['policy_timing_repeats_completed']==(3 if gate else 1)
    assert r['keep_none_technical_timing']['tail_batches']==0
    reference={a['repeat']:a['total_seconds']for a in r['runtime']if a['mode']=='fixed'}
    staged={a['repeat']:a['total_seconds']for a in r['runtime']if a['mode']=='keep_all'}
    timing=[]
    for a in r['runtime']:
        entry=dict(mode=a['mode'],repeat=a['repeat'],seconds=a['total_seconds'],relative_saving=1-a['total_seconds']/reference[a['repeat']],relative_saving_vs_staged=1-a['total_seconds']/staged[a['repeat']])
        if a['mode']!='fixed':
            p=BASE/f"profiling/retention-{a['repeat']}-{a['mode']}.npz"
            with np.load(p,allow_pickle=False)as z:
                keep=z['retained'];v=z['verified'];s=z['scores'];np.testing.assert_array_equal(z['indices'],d['indices'])
            assert keep.sum()==a['retained'] and (~keep).sum()==a['rejected']
            assert (keep.sum()+7)//8==a['tail_batches']
            lost=int(((d['failure']==0)&~keep).sum());assert lost==a['lost_resolutions']
            np.testing.assert_array_equal(v,keep&(d['failure']==0))
            assert v.sum()==a['verified_solved']
            if a['mode']in dev['policies']:
                t=dev['policies'][a['mode']]['threshold'];expected=np.ones(1000,dtype=bool)if t is None else d[a['mode']]<=t
                np.testing.assert_array_equal(keep,expected);np.testing.assert_allclose(s,d[a['mode']],rtol=1e-11,atol=1e-10)
            elif a['mode']=='random':
                expected=np.ones(1000,dtype=bool);expected[np.random.default_rng(20261222).permutation(1000)[:dev['policies']['codes_hidden']['rejected']]]=False
                np.testing.assert_array_equal(keep,expected)
            else:assert keep.all()
            entry.update(verified_resolutions=int(v.sum()),lost_resolutions=lost,retained=int(keep.sum()))
        timing.append(entry)
    output=dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),n=1000,scope='same-environment independent arithmetic, not external confirmation',
        stage_means={n:{k:float(np.mean([a[k]for a in r['stage_records']if a['mode']==n]))for k in ['first9','score','tail','verify','total']}for n in ['none','codes','hidden','codes_hidden','summary_archived']},
        auc_and_frontiers=results,tail_share_median=float(np.median(shares)),tail_share_max=max(shares),runtime=timing,
        decision=r['decision'],source_sha256=sha(Path(__file__)),profile_sha256=sha(BASE/'profiling/report.json'),
        evidence_sha256={str(p.relative_to(ROOT)):sha(p)for p in sorted(BASE.rglob('*'))if p.is_file()and p.name not in ['verification.json','campaign-summary.json']})
    (BASE/'verification.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':'passed','runtime':timing,'decision':r['decision']}))
if __name__=='__main__':main()

"""Family-level paired analysis, with per-template descriptions and explicit abstentions."""
from support import *
import argparse

def summary(stage,head=''):
    obs=read('results/'+stage+head+'_observations.json');rows=read('data/selected.json')[stage]
    x=np.array([[v['nll'] for v in o['values']] for o in obs['crossed']])
    fam=lambda a: np.asarray(a).reshape(len(rows),2).mean(1)
    contrasts=dict(D=fam(x[:,1]-x[:,0]),R=fam(x[:,1]-x[:,2]),I=fam(x[:,1]+x[:,3]-x[:,2]-x[:,0]),
        S=fam(x[:,4]-x[:,2]),B=fam(x[:,5]-x[:,6]),
        block_backups_minus_adapted=fam(x[:,5]-x[:,2]),block_controls_minus_adapted=fam(x[:,6]-x[:,2]),
        self_repair=fam([o['self_repair'] for o in obs['crossed']]),
        delta_DE=fam([o['delta_DE_original_LN'] for o in obs['crossed']]),
        ordinary_ablation_damage=fam(x[:,2]-x[:,0]))
    rng=np.random.default_rng(2026102518);idx=rng.integers(len(rows),size=(20000,len(rows)))
    out={}
    for k,v in contrasts.items():
        means=v[idx].mean(1)
        out[k]=dict(mean=float(v.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
            ci9875=np.quantile(means,[.00625,.99375]).tolist())
    accuracy={name:{m:sum(o['values'][k][m] for o in obs['crossed']) for m in
        ('correct_candidate','correct_vocab')} for k,name in enumerate(obs['conditions'])}
    # These are operational effect claims, not six-way mechanism labels.
    claims={k:bool(out[k]['ci9875'][0]>margin) for k,margin in [('D',.05),('R',.01),('S',.01),('B',.01)]}
    local={k:int((v>dict(D=.05,R=.01,S=.01,B=.01).get(k,.0)).sum()) for k,v in contrasts.items()
           if k in ('D','R','S','B')}
    claim_coverage=dict(complete=sum((contrasts['D']>.05)&(contrasts['R']>.01)&(contrasts['S']>.01)),
        without_specificity=sum((contrasts['D']>.05)&(contrasts['R']>.01)),
        without_frozen='cannot identify necessity of adaptation from Y00/Y11 alone',
        self_repair_positive=sum(contrasts['self_repair']>.0))
    claim_coverage={k:int(v) if isinstance(v,(np.integer,int)) else v for k,v in claim_coverage.items()}
    by_template={}
    for template in sorted({r['template'] for r in rows}):
        mask=np.array([r['template']==template for r in rows])
        by_template[template]=dict(n=int(mask.sum()),**{k:float(v[mask].mean()) for k,v in contrasts.items()})
    er={}
    for other in (1,2):
        conditions=[r for r in obs['E'] if r['other']==other and r['recipient']!=r['donor']]
        for policy in ('regenerated','recipient_native','donor_native'):
            values=[r for r in conditions if r['policy']==policy or
                (policy=='recipient_native' and r['policy']==('fixed_left' if r['recipient']==0 else 'fixed_right')) or
                (policy=='donor_native' and r['policy']==('fixed_left' if r['donor']==0 else 'fixed_right'))]
            er[f'other{other}_{policy}']=dict(n=len(values),recipient_candidate_correct=sum(r['values']['correct_candidate'] for r in values),
                recipient_vocab_correct=sum(r['values']['correct_vocab'] for r in values))
    result=dict(stage=stage,head=head or 'L9H9',n_families=len(rows),n_resamples=len(x),
        contrasts=out,accuracy=accuracy,operational_claims_adjusted=claims,
        per_family_threshold_counts=local,claim_coverage=claim_coverage,by_template=by_template,E=er,
        gate=obs['gate'],elapsed_seconds=obs['elapsed_seconds'],
        status='development descriptive' if stage=='dev' else 'reserved evaluation; conditionally independent source families')
    write('results/'+stage+head+'_summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('by_template',)},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='dev');p.add_argument('--suffix',default='');a=p.parse_args()
    summary(a.split,a.suffix)

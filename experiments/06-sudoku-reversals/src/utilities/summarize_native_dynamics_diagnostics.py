"""Post-hoc descriptors and conservative rare-discordance CI; no model execution."""
from pathlib import Path
import hashlib,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'audit/metacognition/native-dynamics-v1/causal'

def cdf(k,n,p):
    if p==0:return 1.
    if p==1:return float(k>=n)
    return min(1.,sum(math.exp(math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)+j*math.log(p)+(n-j)*math.log1p(-p))for j in range(k+1)))

def inverse(k,n,target):
    lo,hi=0.,1.
    for _ in range(80):
        mid=(lo+hi)/2
        if cdf(k,n,mid)>target:lo=mid
        else:hi=mid
    return (lo+hi)/2

def cp(k,n):return (0. if k==0 else inverse(k-1,n,1-.00625),1. if k==n else inverse(k,n,.00625))

def main():
    report=json.loads((BASE/'report.json').read_text());out=np.load(BASE/'outcomes.npz',allow_pickle=False)
    result=dict(status='complete',scope='posthoc descriptive geometry and exact paired diagnostics; no fitting or generation',treatments={})
    for name in ['runner_up','matched']:
        d=out[name+'_active_success'].astype(int)-out[name+'_fixed_success'].astype(int)
        win,lose=int((d>0).sum()),int((d<0).sum());n=win+lose
        w,l=cp(win,len(d)),cp(lose,len(d))
        result['treatments'][name]=dict(quantiles_0_05_50_95_100={k:np.quantile(out[name+'_'+k],[0,.05,.5,.95,1]).tolist()for k in ['compensation','relative_final_latent_norm','forced_probability']},
            future_codes_unchanged=int((out[name+'_future_codes_changed']==0).sum()),
            exact_paired_p=min(1.,2*sum(math.comb(n,k)for k in range(min(win,lose)+1))/2**n)if n else 1.,
            conservative_exact_difference_interval_at_least_97_5=[w[0]-l[1],w[1]-l[0]])
    result.update(norm_match_mismatch_quantiles_0_50_90_100=report['norm_match_relative_mismatch_quantiles'],
        exact_interval_method='difference of marginal Clopper-Pearson98.75 percent intervals, four tail probabilities0.00625, joint Bonferroni >=97.5 percent for difference; posthoc sensitivity, not replacement of primary bootstrap',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outcomes_sha256=hashlib.sha256((BASE/'outcomes.npz').read_bytes()).hexdigest())
    (BASE/'descriptive-diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:dict(exact_paired_p=v['exact_paired_p'],conservative_interval=v['conservative_exact_difference_interval_at_least_97_5'])for k,v in result['treatments'].items()}))

if __name__=='__main__':main()

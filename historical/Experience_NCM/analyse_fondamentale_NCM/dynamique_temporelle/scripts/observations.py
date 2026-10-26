"""Native stops and reversals from previously verified cumulative decodings."""
from base import *
LENGTHS=np.array([1,2,4,6,9,13,18,26,37,50,67,89])

def filehash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def ratio_ci(numer,denom):
    numer=np.asarray(numer,dtype=float);denom=np.asarray(denom,dtype=float);rng=np.random.default_rng(Q['seed']);ix=rng.integers(0,len(numer),(Q['bootstrap'],len(numer)))
    ds=denom[ix].sum(1);valid=ds>0;v=numer[ix].sum(1)[valid]/ds[valid]
    return dict(rate=float(numer.sum()/denom.sum()) if denom.sum() else None,ci95=np.quantile(v,[.025,.975]) if len(v) else None,numerator=int(numer.sum()),denominator=int(denom.sum()),n_puzzles=len(numer))

def change_fits(q):
    rng=np.random.default_rng(Q['seed']);curves=np.stack([q[rng.integers(0,len(q),len(q))].mean(0) for _ in range(Q['bootstrap'])]);result={};y=q.mean(0)
    for name,xx in [('level',np.arange(1,13)),('cumulative_codes',LENGTHS.cumsum())]:
        x=(xx-xx.min())/(xx.max()-xx.min());design=np.column_stack([np.ones(12),x]);linear=design@np.linalg.lstsq(design,y,rcond=None)[0];fits=[];errors=[]
        for k in range(3,11):
            a=np.column_stack([np.ones(12),x,np.maximum(x-x[k-1],0)]);projection=a@np.linalg.pinv(a);pred=projection@y;err=((curves-curves@projection.T)**2).sum(1)
            fits.append(dict(knot=k,sse=float(((y-pred)**2).sum()),prediction=pred));errors.append(err)
        best=min(fits,key=lambda v:v['sse']);pref=np.argmin(np.stack(errors,1),1)
        result[name]=dict(best=best,linear_prediction=linear,linear_sse=float(((y-linear)**2).sum()),fits=fits,bootstrap_preference={str(k):float((pref==i).mean()) for i,k in enumerate(range(3,11))},warning='Same 12 points select and fit the knot; not a significance test or finite-size scaling.')
    return result

def describe(a,indices,label):
    ix=np.asarray(indices);d=a['decoded'][ix];ref=a['reference'][ix];prefix=a['prefix'][ix];cells=d[:,:,CELL];truth=ref[:,CELL];correct=cells==truth[:,None,:];blank=prefix[:,1:-1][:,CELL]==0
    profiles=[];transition=[]
    for k in range(12):
        m=evaluate(d[:,k],ref,prefix);errors=(~correct[:,k]).sum(1)
        profiles.append(dict(level=k+1,length=int(LENGTHS[k]),cumulative_codes=int(LENGTHS[:k+1].sum()),**m,cell_accuracy_blank=float(correct[:,k][blank].mean()),
          solved_cells_only=int((errors==0).sum()),bad_separator_grids=int(np.any(d[:,k][:,SEP]!=10,axis=1).sum()),cell_error_quantiles=np.quantile(errors,[0,.1,.25,.5,.75,.9,1])))
        if k:
            before=correct[:,k-1];after=correct[:,k];changed=cells[:,k-1]!=cells[:,k]
            def counts(mask):return dict(corrected=int((~before&after&mask).sum()),degraded=int((before&~after&mask).sum()),wrong_to_wrong=int((~before&~after&changed&mask).sum()),changed=int((changed&mask).sum()))
            transition.append(dict(level=k+1,all_cells=counts(np.ones_like(blank,bool)),blank_cells=counts(blank),corrected_per_added_code_per_puzzle=float((~before&after).sum()/(len(ix)*LENGTHS[k]))))
    corrected9=~correct[:,7]&correct[:,8];everwrong=(~correct[:,9:]).any(1);finalwrong=~correct[:,-1];everchange=(cells[:,9:]!=cells[:,8,None,:]).any(1);return8=(cells[:,9:]==cells[:,7,None,:]).any(1)
    reversal=dict(corrected9_then_wrong_any=ratio_ci((corrected9&everwrong).sum(1),corrected9.sum(1)),corrected9_then_wrong_final=ratio_ci((corrected9&finalwrong).sum(1),corrected9.sum(1)),
      correct9_then_wrong_any=ratio_ci((correct[:,8]&everwrong).sum(1),correct[:,8].sum(1)),corrected9_then_back_to_value8_any=ratio_ci((corrected9&return8).sum(1),corrected9.sum(1)),
      fraction_values9_changed_later=float(everchange.mean()),grids9_identical_to12=int(np.all(d[:,8]==d[:,-1],axis=1).sum()),
      solved9_lost_by12=int((np.all(d[:,8]==ref,axis=1)&~np.all(d[:,-1]==ref,axis=1)).sum()))
    last=np.where(cells[:,1:]!=cells[:,:-1],np.arange(2,13)[None,:,None],1).max(1);last_counts=np.bincount(last.ravel(),minlength=13)[1:]
    success=np.all(d==ref[:,None,:],axis=-1);stops={str(k):paired_mean_ci(success[:,k-1].astype(float)-success[:,-1]) for k in [8,9,10,11]}
    q=correct.mean(2);acceleration=paired_mean_ci((q[:,8]-q[:,7])-.5*((q[:,7]-q[:,6])+(q[:,9]-q[:,8])))
    examples=[]
    for i,c in np.argwhere(corrected9&everwrong)[:12]:
        examples.append(dict(source_index=int(ix[i]),cell=int(c),row=int(c//9+1),column=int(c%9+1),reference=int(truth[i,c]),values8_to12=cells[i,7:,c],puzzle=prefix[i,1:-1][CELL]))
    return dict(label=label,n=len(ix),profiles=profiles,transitions=transition,reversals=reversal,stop_minus_final=stops,secondary_acceleration=acceleration,last_observed_change_histogram=last_counts,
      counterexamples=examples,descriptive_fits=change_fits(q),note='Last change is measured only within the fixed 12-level horizon; not an irreversible convergence time.')

def main():
    if not (D/'results/protocol_frozen.json').exists():freeze()
    roots=[ROOT,ROOT/'directions_mecanistes',ROOT/'generalisation_architecturale',ROOT/'correction_pooling'];write(D/'sources/previous_manifests.json',{str((r/'results/final_manifest.json').relative_to(ROOT)):filehash(r/'results/final_manifest.json') for r in roots})
    result={}
    for task in ['hard','easy','medium']:
        a=np.load(ROOT/f'results/{task}.npz');result[task+'_all']=describe(a,np.arange(len(a['reference'])),task+'_all')
        if task=='hard':result['hard_test']=describe(a,np.array(SPLIT['test']),'hard_test')
    a=np.load(ROOT/'results/external.npz');result['external']=describe(a,np.arange(len(a['reference'])),'external')
    write(D/'results/observations.json',result);log('observations_complete',test_reversals=result['hard_test']['reversals'],test_success=[v['solved'] for v in result['hard_test']['profiles']])

if __name__=='__main__':main()

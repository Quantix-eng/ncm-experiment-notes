"""Separate omission of a fixed contribution from omission of its prediction."""
from base import *
from runtime import Engine

@torch.inference_mode()
def main():
    model=SudokuModel(ROOT/'artifacts/sudoku/hard');report={};saved={}
    for name in ['official_test','external']:
        if name=='official_test':
            p,s=boards('hard');ix=np.array(SPLIT['test']);p=p[ix];s=s[ix];cached=np.load(ROOT/'results/hard.npz')
            native=cached['decoded'][ix,-1];nativecodes=cached['codes'][ix]
        else:
            p,s=external_boards();ix=np.arange(len(p));cached=np.load(ROOT/'results/external.npz');native=cached['decoded'][:,-1];nativecodes=cached['codes']
        current={}
        for condition in ['fixed_code11_keep_residual','zero_residual_regenerate']:
            ids=[];codes=[];times=[];calls=[]
            for lo in range(0,len(p),16):
                t=time.perf_counter()
                with Engine(model,p[lo:lo+16]) as engine:
                    out=engine.run(fixed5=condition.startswith('fixed'),zero5=condition.startswith('zero'))
                    calls.append(engine.calls)
                ids.append(out['final_ids'].numpy());codes.append(out['codes'].numpy());times.append(time.perf_counter()-t)
            ids=np.concatenate(ids);codes=np.concatenate(codes);gold=np.all(native==s,axis=1);new=np.all(ids==s,axis=1);delta=new.astype(float)-gold
            status=int((gold!=new).sum())
            current[condition]=dict(metrics=evaluate(ids,s,p),changed_grids=int(np.any(ids!=native,axis=1).sum()),changed_codes=int((codes!=nativecodes).sum()),
               introduced_errors=int((gold&~new).sum()),rescued=int((~gold&new).sum()),success_status_change_wilson95=ci_rate(status,len(gold)),
               delta_success=paired_mean_ci(delta),generator_calls_per_batch=sorted(set(calls)),wall_seconds=sum(times),
               caveat='Runtime is descriptive; conditions not randomized/interleaved, concurrent workloads may differ.')
            saved[name+'_'+condition+'_ids']=ids;saved[name+'_'+condition+'_codes']=codes
            log('level5_intervention',corpus=name,condition=condition,**current[condition]['metrics'])
        report[name]=current;saved[name+'_indices']=ix;saved[name+'_reference']=s;saved[name+'_native']=native
        write(D/'results/level5_interventions.json',report)
    np.savez_compressed(D/'results/level5_interventions.npz',**saved);log('level5_interventions_complete')

if __name__=='__main__':main()

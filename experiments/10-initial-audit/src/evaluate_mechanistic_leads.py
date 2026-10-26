"""Read-only reassessment of two hypotheses from the saved experiment traces.

No model inference, probe fitting, intervention or modification of prior results.
"""
import os,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CELL=np.array([i for i in range(89) if i%10!=9])

def stats(old,new,ref,mask):
    oldok=old==ref;newok=new==ref;changed=old!=new
    fixed=mask&~oldok&newok;broken=mask&oldok&~newok
    def rate(a,b):return float(a/b) if b else None
    nwrong=int((mask&~oldok).sum());ncorrect=int((mask&oldok).sum());nchanged=int((mask&changed).sum())
    assert int(fixed.sum()+broken.sum()+(mask&changed&~oldok&~newok).sum())==nchanged
    per=fixed.sum(1)
    return dict(cells=int(mask.sum()),previously_wrong=nwrong,previously_correct=ncorrect,
        corrected=int(fixed.sum()),broken=int(broken.sum()),changed=nchanged,
        wrong_to_other_wrong=int((mask&changed&~oldok&~newok).sum()),
        repair_rate_given_wrong=rate(fixed.sum(),nwrong),damage_rate_given_correct=rate(broken.sum(),ncorrect),
        net_accuracy_gain=rate(fixed.sum()-broken.sum(),mask.sum()),
        fraction_of_changes_that_correct=rate(fixed.sum(),nchanged),
        puzzles_with_correction=int((per>0).sum()),corrections_per_puzzle_quantiles=np.quantile(per,[0,.25,.5,.75,1]).tolist())

def main():
    result={}
    for d in ['hard','easy','medium']:
        a=np.load(ROOT/f'results/{d}.npz');dec=a['decoded'][:,:,CELL];ref=a['reference'][:,CELL];given=a['prefix'][:,1:-1][:,CELL]
        scopes={'all':np.ones_like(given,dtype=bool),'initially_blank':given==0,'given':given!=0}
        transitions=[]
        for k in range(1,12):
            transitions.append(dict(level=k+1,**{key:stats(dec[:,k-1],dec[:,k],ref,mask) for key,mask in scopes.items()}))
        # Simple comparator for predicting next values: current values, without fitting.
        next8=dec[:,8];current8=dec[:,7];final=dec[:,-1];blank=given==0
        result[d]=dict(transitions=transitions,level8_to9=dict(
           persist_next_value_accuracy=float((current8[blank]==next8[blank]).mean()),
           next_value_equals_final_value_on_changed_blanks=float((next8[blank&(current8!=next8)]==final[blank&(current8!=next8)]).mean()),
           fraction_of_initially_blank_cells_changed=float((current8[blank]!=next8[blank]).mean())))
    old=json.loads((ROOT/'results/constant_level.json').read_text())
    result['constant_level_ablation']=old
    result['notes']=['All counts describe offline decoder readouts, not an explicit grid state fed back to the generator.',
       'Transition analyses count individual cells; inferential resampling must group by puzzle.',
       'A prospective probe must exclude all step-9 codes, logits and decoded predictions if its observation time is before step 9.',
       'This is post-hoc assessment of hypotheses; no confirmatory testing or causal learning-history conclusion.']
    out=ROOT/'results/pistes_mecanistes/assessment.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    files=[Path(__file__),out,ROOT/'results/final_manifest.json']
    manifest={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (out.parent/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for d in ['hard','easy','medium']:
        r=result[d]
        for k in [9,12]:
            t=r['transitions'][k-2];print(d,k,json.dumps(t,ensure_ascii=False))

if __name__=='__main__':main()

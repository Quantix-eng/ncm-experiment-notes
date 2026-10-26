"""Post-hoc clarification: compare network and oracle on the same defined mask.

Adds a reported quantity; does not change predictions, thresholds or C2 scores.
Also recomputes C3 native argmax and conformity counts from saved logits.
"""
from pathlib import Path
import json
import numpy as np
D=Path(__file__).resolve().parent
def read(p):return json.loads((D/p).read_text())
if __name__=='__main__':
    rows=read('results/c2_falsification.json');pred={r['query']:r for r in read('results/c2_predictions.json')}
    methods=list(rows[0]['confidence']);result={};checks=0
    for case in [13,18,19,21]:
        valid=[r for r in rows if r['case']==case and r['scorable_oracle']];result[str(case)]={}
        for method in methods:
            emitted=[r for r in valid if r['confidence'][method]>=1.]
            bad=0
            for r in emitted:
                proposed=pred[r['query']]['prediction']
                bad+=any(proposed[i]!=r['learned_prediction'][i] for i,t in enumerate(r['oracle_targets']) if t is not None)
                assert r['learned_oracle_agreement']==all(r['learned_prediction'][i]==t for i,t in enumerate(r['oracle_targets']) if t is not None)
                checks+=1
            result[str(case)][method]=dict(scorable_claims=len(emitted),network_contradictions_same_mask=bad,
                risk_network_same_mask=None if not emitted else bad/len(emitted))
    c3=read('results/c3_padding.json');logits=np.load(D/'results/c3_padding_logits.npz')
    for r in c3['observations']:
        a=logits[r['id']];assert np.isfinite(a).all()
        actual=a[0,1:len(r['targets'])+1].argmax(-1).tolist();assert actual==r['network_prediction']
        match=all(actual[i]==t for i,t in enumerate(r['targets']) if t is not None)
        assert match==r['padded_network_program'];checks+=2
    out=dict(passed=True,posthoc=True,C2_predictions_changed=False,C2_primary_metrics_changed=False,
        task_results=result,scalar_checks=checks,C3_native_argmax_audited=1024,
        purpose='separate all-useful-position network risk from same-oracle-mask network risk',new_neural_forward=False)
    (D/'results/c2_mask_and_c3_audit.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('Mask and C3 audit passed',checks)

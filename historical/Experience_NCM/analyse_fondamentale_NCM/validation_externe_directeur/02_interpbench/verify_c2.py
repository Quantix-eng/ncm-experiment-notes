"""Scalar prediction reconstruction, read-only logits and provenance audit."""
from pathlib import Path
import json,hashlib
import numpy as np
D=Path(__file__).resolve().parent
def read(p):return json.loads((D/p).read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
    for p,h in read('results/c2_frozen.json')['files'].items():assert digest(Path(p))==h,p
    for p,h in read('results/c2_predictions_frozen.json')['files'].items():assert digest(D/p)==h,p
    probes=read('results/c2_probes.json');preds={r['query']:r for r in read('results/c2_predictions.json')}
    native={r['id']:r for r in read('results/c2_native.json')};rows=read('results/c2_falsification.json')
    logits=np.load(D/'results/c2_future_logits.npz');count=0
    for p in probes:
        original=native[p['source']]['learned_prediction']
        same=[[cats==original for cats in q['argmax']] for q in p['probes']]
        expected={'O_prudent':0.,'O_naive':1.,'C_without_frozen':sum(q[2] for q in same)/2,
            'C_without_mismatch':sum(sum(q[:4]) for q in same)/8,'C_full':sum(sum(q) for q in same)/10}
        assert preds[p['query']]['confidence']==expected;count+=5
        assert preds[p['query']]['prediction']==original
    for r in rows:
        a=logits[r['query']+'-future'];assert np.isfinite(a).all()
        actual=a[0,1:len(r['oracle_targets'])+1].argmax(-1).tolist()
        assert actual==r['learned_prediction'];p=preds[r['query']]
        assert r['prediction_contradicted_by_network']==(p['prediction']!=actual)
        if r['scorable_oracle']:
            different=any(p['prediction'][i]!=t for i,t in enumerate(r['oracle_targets']) if t is not None)
            assert r['prediction_contradicted_by_oracle']==different
        assert p['future_donor']==r['donor'];count+=3
    E=read('results/c2_E.json');reg=[r for r in E if r['policy']=='regenerated']
    assert all(r['donor_native_logit_max_error']==0 for r in reg)
    history=read('sources/history_audit/history_comparison.json')
    assert history['c5645ddbbbb7cb4f37d88ad175e517f22e029039']==history['a1242a84f8a4d07d1be9a8f5ec710415198016b2']
    out=dict(passed=True,scalar_checks=count,families=len(native),queries=len(rows),E_regenerated_checks=len(reg),
        all_future_logits_finite=True,prediction_seal_intact=True,input_seal_intact=True,
        case13_history_checked=True,new_neural_forward=False,external_replication=False,
        scope='checks scoring and provenance; oracle and model forward implemented internally')
    (D/'results/c2_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

"""Finite task-level coverage and contradictions; no iid query significance."""
from pathlib import Path
import json,hashlib
import numpy as np
D=Path(__file__).resolve().parent
def read(p):return json.loads((D/p).read_text())
def ratio(a,b):return None if not b else a/b
def score(rows,method,threshold):
    emitted=[r for r in rows if r['confidence'][method]>=threshold]
    valid=[r for r in emitted if r['scorable_oracle']]
    bad_network=sum(r['prediction_contradicted_by_network'] for r in emitted)
    bad_oracle=sum(r['prediction_contradicted_by_oracle'] for r in valid)
    return dict(total=len(rows),emitted=len(emitted),coverage=ratio(len(emitted),len(rows)),
        contradicted_by_network=bad_network,risk_network=ratio(bad_network,len(emitted)),
        oracle_scorable_emitted=len(valid),contradicted_by_oracle=bad_oracle,
        risk_oracle=ratio(bad_oracle,len(valid)))

if __name__=='__main__':
    rows=read('results/c2_falsification.json');native=read('results/c2_native.json')
    methods=list(rows[0]['confidence']);tasks={}
    for case in [13,18,19,21]:
        rs=[r for r in rows if r['case']==case];ns=[r for r in native if r['case']==case]
        valid=[r for r in rs if r['scorable_oracle']];nvalid=[r for r in ns if r['defined']]
        alignment={}
        for kind in ['aligned','unaligned']:
            selected=[r for r in valid if r['type']==kind]
            alignment[kind]=dict(n=len(selected),agreement=sum(r['learned_oracle_agreement'] for r in selected),
                fraction=ratio(sum(r['learned_oracle_agreement'] for r in selected),len(selected)))
        tasks[str(case)]=dict(families=len(ns),native_scorable=len(nvalid),
            native_exact_defined=sum(r['exact_defined'] for r in nvalid),queries=len(rs),
            oracle_scorable=len(valid),craft_RASP_disagreements=sum(not r['craft_RASP_agreement'] for r in rs),
            empty_masks=sum(not any(r['defined_mask']) for r in rs),interchange_conformity=alignment,
            methods={m:{str(t):score(rs,m,t) for t in [.5,.75,1.]} for m in methods})
    macro={}
    for method in methods:
        macro[method]={}
        for threshold in [.5,.75,1.]:
            values=[tasks[str(c)]['methods'][method][str(threshold)] for c in [13,18,19,21]]
            mean=lambda k:None if not any(v[k] is not None for v in values) else float(np.mean([v[k] for v in values if v[k] is not None]))
            macro[method][str(threshold)]=dict(coverage=mean('coverage'),risk_network=mean('risk_network'),
                risk_oracle=mean('risk_oracle'),n_tasks_with_oracle_claims=sum(v['risk_oracle'] is not None for v in values))
    E=read('results/c2_E.json');reg=[r for r in E if r['policy']=='regenerated']
    out=dict(tasks=tasks,macro_average_over_tasks=macro,n_queries=len(rows),n_families=len(native),
        E=dict(conditions=len(E),regenerated_conditions=len(reg),
            max_regenerated_donor_native_error=max(r['donor_native_logit_max_error'] for r in reg)),
        interpretation='invariance prediction on future donor after test-time probes; not mechanism classification',
        estimators='finite counts; no population intervals; methods are internal baselines',
        definitions='score is agreement fraction, not a calibrated probability; missing risk at zero coverage is null')
    (D/'results/c2_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'n_queries':len(rows),'tasks':{k:{x:v[x] for x in ['families','native_scorable','native_exact_defined','craft_RASP_disagreements','interchange_conformity']} for k,v in tasks.items()},'macro_strict':{m:v['1.0'] for m,v in macro.items()},'E':out['E']},indent=2))

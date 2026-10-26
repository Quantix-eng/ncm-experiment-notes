"""Re-execute analytical controls on the included fixed source families.

This recomputes interventions and diagnostic predictions, rather than reading
saved predictions. It reuses the source families and is not a new dataset.
"""
import json
import hashlib
import time
from support import D,Q,np,write
from controls import observe
from methods import METHODS,project
from run import score


def main():
    families=json.loads((D/'data/families.json').read_text())['rows']
    records=[];truth={};start=time.monotonic()
    for index,source in enumerate(families):
        models=[(name,None) for name in Q['pure_controls']]
        if source['split']=='evaluation':models += [('mixed',alpha) for alpha in Q['mixture_alphas']]
        for name,alpha in models:
            case_id=hashlib.sha256(f"{Q['seed']}:{source['source_id']}:{name}:{alpha}".encode()).hexdigest()[:20]
            record,_=observe(source,name,alpha,np.random.default_rng(Q['seed']+1009*index))
            record['case_id']=case_id
            record['predictions']={method:function(project(record,regime)) for method,(regime,function) in METHODS.items()}
            truth[case_id]=dict(source_id=source['source_id'],family=source['family'],split=source['split'],label=name,alpha=alpha)
            records.append(record)
    scores=score(records,truth)
    result={'scope':'fresh analytical execution on the included source families; not independent confirmation',
            'families':len(families),'instances':len(records),'scores':scores,'seconds':time.monotonic()-start}
    assert scores['complete_intervention']['all']['singleton_correct']==len(records)
    write(D/'results/fixed_controls.json',result)
    print(json.dumps({'passed':True,'families':len(families),'instances':len(records),'complete':scores['complete_intervention']['all']},indent=2))


if __name__=='__main__':main()

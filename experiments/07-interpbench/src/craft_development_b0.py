"""B0 development audit: external Craft and RASP oracle versus learned weights."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from official_audit_a1 import build
from pilot import encode,CASES
from craft_oracle import CraftOracle,check_native,rasp

HERE=Path(__file__).resolve().parent
torch.set_num_threads(1)


def patch_hook(variable,donor):
    layer,kind,head=variable
    name=f'blocks.{layer}.attn.hook_result' if kind=='attn' else f'blocks.{layer}.hook_mlp_out'
    def patch(value,hook):
        result=value.clone()
        if kind=='attn':result[:,:,head]=donor[name][:,:,head]
        else:result=donor[name]
        return result
    return name,patch


@torch.inference_mode()
def run():
    start=time.perf_counter();seal=json.loads((HERE/'results/craft_b0_frozen.json').read_text())
    for p,h in seal['sha256'].items():assert hashlib.sha256((HERE/p).read_bytes()).hexdigest()==h,p
    data=json.loads((HERE/'data/dev.json').read_text());rows=[];schemas={}
    for case in [8,33]:
        oracle=CraftOracle(case);schemas[str(case)]=oracle.schema()
        model,_=build(case,torch.float32)
        subset=[r for r in data if r['case']==case]
        native=[model.run_with_cache(encode(row)) for row in subset]
        craft_states=[oracle.native(row['tokens']) for row in subset]
        for i,row in enumerate(subset):
            ok,sem=check_native(oracle,row['tokens']);assert ok,('native oracle',row['id'],sem)
            expected=oracle.target_ids(sem);assert expected==row['target_ids']
            same_length=[j for j,r in enumerate(subset) if len(r['tokens'])==len(row['tokens']) and j!=i]
            assert len(same_length)>=2
            for donor in same_length[:2]:
                for variable in [(0,'attn',0),(0,'mlp',None),(1,'attn',0)]:
                    semantic=oracle.semantic(row['tokens'],variable,subset[donor]['tokens'])
                    targets=oracle.target_ids(semantic);mask=np.array([x is not None for x in targets])
                    expected=np.array([0 if x is None else x for x in targets])
                    key=variable
                    patch=None if key not in craft_states[donor][1] else (key,craft_states[donor][1][key])
                    scores,_=oracle.native(row['tokens'],patch)
                    craft_match=bool(np.array_equal(scores[1:].argmax(-1)[mask],expected[mask]))
                    assert craft_match,('Craft/RASP intervention',row['id'],donor,variable,semantic)
                    actual=model.run_with_hooks(encode(row),fwd_hooks=[patch_hook(variable,native[donor][1])])
                    predicted=actual[0,1:len(targets)+1].argmax(-1).numpy()
                    low_match=bool(np.array_equal(predicted[mask],expected[mask]))
                    rows.append({'source':row['id'],'donor':subset[donor]['id'],'case':case,'variable':list(variable),
                                 'oracle_semantic_ids':targets,'oracle_defined_mask':mask.tolist(),
                                 'craft_semantic_agreement':craft_match,'learned_oracle_agreement':low_match,
                                 'learned_prediction':predicted.tolist(),'changed':bool(targets!=row['target_ids'])})
        del model,native
    from circuits_benchmark.benchmark.common_programs import shift_by
    shifted=shift_by(1,rasp.tokens)
    assert shifted(['a','b','c'])==[None,'a','b']
    mapped=rasp.Map(lambda x:x.upper(),shifted)
    assert mapped(['a','b','c'])==[None,'A','B']
    out={'status':'development_oracle_audit_complete','schemas':schemas,'interventions':rows,
         'native_sources':len(data),'oracle_intervention_count':len(rows),
         'learned_oracle_agreement_count':sum(r['learned_oracle_agreement'] for r in rows),
         'changed_oracle_cases':sum(r['changed'] for r in rows),'none_propagation_unit_checks':2,
         'reserved_network_forwards':0,'elapsed_seconds':time.perf_counter()-start,
         'scope':'external program/Craft/learned alignment on development; not reserved task validation'}
    (HERE/'results/craft_development_b0.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['schemas','interventions']},indent=2))


if __name__=='__main__':run()

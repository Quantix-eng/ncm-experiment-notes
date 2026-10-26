"""Post-hoc canonical-padding control; no replacement of the C2 held-out test."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import torch
from official_audit_a1 import build
from pilot import vocabulary,CASES
from craft_oracle_c2 import CraftOracle
D=Path(__file__).resolve().parent
def read(p):return json.loads((D/p).read_text())
if __name__=='__main__':
    seal=read('results/c3_frozen.json')
    for p,h in seal['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    torch.set_num_threads(1);start=time.perf_counter();old={r['id']:r for r in read('results/c2_native.json')}
    results=[];arrays={}
    with torch.inference_mode():
        for case in [13,18,19,21]:
            model,_=build(case,torch.float32);oracle=CraftOracle(case)
            vocab={v:i for i,v in enumerate(vocabulary(case))}
            ctx=CASES[str(case)]['transformer_cfg']['n_ctx']
            for row in [r for r in read('data/reserved_inputs.json') if r['case']==case]:
                useful=row['tokens'];padded=useful+['PAD']*(ctx-1-len(useful));assert len(padded)==ctx-1
                yy=model(torch.tensor([[vocab['BOS']]+[vocab[v] for v in padded]]))
                assert torch.isfinite(yy).all()
                actual=yy[0,1:len(useful)+1].argmax(-1).numpy()
                scores,_=oracle.native(padded);craft=scores[1:len(useful)+1].argmax(-1)
                targets=old[row['id']]['targets'];mask=np.array([t is not None for t in targets])
                wanted=np.array([0 if t is None else t for t in targets])
                exact=lambda a,b:bool(np.array_equal(a[mask],b[mask]))
                results.append(dict(id=row['id'],case=case,useful_length=len(useful),defined=int(mask.sum()),
                    unpadded_network_program=old[row['id']]['exact_defined'],
                    padded_network_program=exact(actual,wanted),padded_Craft_program=exact(craft,wanted),
                    padded_network_Craft=exact(actual,craft),network_prediction=actual.tolist(),
                    craft_prediction=craft.tolist(),targets=targets))
                arrays[row['id']]=yy.numpy()
            del model
    summary={}
    for case in [13,18,19,21]:
        subset=[r for r in results if r['case']==case and r['defined']]
        summary[str(case)]=dict(scorable=len(subset),**{name:sum(r[name] for r in subset) for name in
            ['unpadded_network_program','padded_network_program','padded_Craft_program','padded_network_Craft']})
    out=dict(posthoc=True,observations=results,summary=summary,elapsed_seconds=time.perf_counter()-start,
        C2_results_replaced=False,scientific_thresholds_changed=False)
    (D/'results/c3_padding.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(D/'results/c3_padding_logits.npz',**arrays)
    print(json.dumps(summary,indent=2))

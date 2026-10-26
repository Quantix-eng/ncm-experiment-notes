"""Preserves the frozen failed test; records its extent without changing criteria."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_verifier',HERE/'verify.py')
verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
rows=json.loads((HERE/'data/dev.json').read_text())
logits=np.load(HERE/'results/development_logits.npz')
records=[]
for case in [8,33]:
    weights=torch.load(HERE/f'sources/models/{case}/ll_model.pth',map_location='cpu',weights_only=True)
    vocab=list(range(11))+['BOS','PAD'] if case==8 else ['BOS','J','LB','PAD','TPSI','V','b','no','oCLrZaW','poiVg']
    index={v:i for i,v in enumerate(vocab)}
    for row in [r for r in rows if r['case']==case]:
        ids=[index['BOS']]+[index[x] for x in row['tokens']]+[index['PAD']]*(9-len(row['tokens']))
        reference=verifier.forward(np.array(ids),weights)
        observed=logits[row['id']+'-native'][0]
        records.append({'id':row['id'],'case':case,
                        'max_abs_difference':float(np.max(np.abs(reference-observed))),
                        'frozen_tolerance_pass':bool(np.allclose(reference,observed,atol=2e-4,rtol=2e-5)),
                        'all_argmax_equal':bool(np.array_equal(reference.argmax(-1),observed.argmax(-1)))})
seal=json.loads((HERE/'results/protocol_frozen.json').read_text())
integrity=all(hashlib.sha256((HERE/p).read_bytes()).hexdigest()==h for p,h in seal['sha256'].items())
original=json.loads((HERE/'results/development.json').read_text())
cells=[]
for case in [8,33]:
    for cut in [0,1]:
        obs=[r for r in original['interventions'] if r['case']==case and r['cut']==cut]
        cells.append({'case':case,'cut':cut,'n':len(obs),
                      'exact':{k:sum(r['conditions'][k]['exact'] for r in obs) for k in ['Y00','Y10','Y11','Y01','Y1wrong']},
                      'mean_contrasts':{k:float(np.mean([r['contrasts'][k] for r in obs])) for k in ['D','R','I','S']}})
output={'status':'failed_frozen_numeric_audit_branch_stopped',
        'initial_failure':{'source':'dev-8-017','max_abs_difference':0.0005538618895570835},
        'posthoc':True,'criteria_changed':False,'threshold_pass_count':sum(r['frozen_tolerance_pass'] for r in records),
        'argmax_equal_count':sum(r['all_argmax_equal'] for r in records),'n':len(records),
        'max_abs_difference':max(r['max_abs_difference'] for r in records),
        'frozen_hashes_unchanged':integrity,'records':records,'cells':cells,
        'official_loader_parity_not_established':True,'reserved_tasks_executed':[]}
(HERE/'results/numerical_failure_diagnostic.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({k:v for k,v in output.items() if k not in ['records','cells']},indent=2))

"""Prepare inputs and donor identities; no network or task-oracle evaluation."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
meta=json.loads((HERE/'sources/benchmark_metadata.json').read_text())
cases={int(x['case_id']):x for x in meta['cases'] if x['case_id'].isdigit()}
rows=[]
for case in [13,18,19,21]:
    rng=np.random.default_rng(20261125+case)
    vocab=cases[case]['vocab'];used=set()
    for i in range(256):
        while True:
            n=int(rng.integers(cases[case]['min_seq_len']-1,cases[case]['max_seq_len']))
            seq=[vocab[int(k)] for k in rng.integers(0,len(vocab),n)]
            if tuple(seq) not in used:used.add(tuple(seq));break
        rows.append({'id':f'reserved-{case}-{i:03d}','case':case,'tokens':seq,
                     'proposal_donors':[f'reserved-{case}-{(i+j)%256:03d}' for j in [1,2]],
                     'falsification_donor':f'reserved-{case}-{(i+128)%256:03d}'})
dst=HERE/'data/reserved_inputs.json'
if dst.exists():raise SystemExit('Preserve existing reserved inputs; refuse overwrite')
dst.write_text(json.dumps(rows,indent=2)+'\n')
out={'status':'prepared_not_evaluated_not_final_experiment_freeze','families':len(rows),
     'per_task':256,'cases':[13,18,19,21],'outputs_computed':False,
     'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),
     'notes':['Sources are unique within task; same input may act as donor in correlated queries.',
              'Proposal and falsification donors differ within a query; this is not a claim of globally disjoint families.',
              'Before any prediction fit, implement source-family split or prequential restrictions preventing donor-output leakage.',
              'No reference outputs or learned model outputs are included.']}
(HERE/'results/reserved_preparation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

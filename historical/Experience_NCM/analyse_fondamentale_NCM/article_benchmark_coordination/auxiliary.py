"""Post-hoc exact finite-design information and sensitivity; no model calls."""
import hashlib
import json
import itertools
import math
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
ROOT = D.parent
NCM = ROOT/'benchmark_ncm_reel'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()


def write(name,value):
    p=D/name
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))


def input_paths():
    return [D/'auxiliary.py',D/'PROTOCOLE_AUXILIAIRE.md',
            NCM/'results/native.npz',NCM/'results/summary.json',
            NCM/'results/output_quality.json',NCM/'protocol.json',
            ROOT/'benchmark_causal_revision/data/families.json',
            ROOT/'benchmark_causal_revision/results/summary.json']


def check():
    freeze=json.loads((D/'results/auxiliary_frozen.json').read_text())
    for name,digest in freeze['files'].items(): assert sha(ROOT/name)==digest,name


def entropy(values):
    counts=Counter(values)
    return -sum((n/len(values))*math.log2(n/len(values)) for n in counts.values())


def mutual(x,y):
    return entropy(x)+entropy(y)-entropy(list(zip(x,y)))


def main():
    if sys.argv[1:]==['--freeze']:
        assert not (D/'results/auxiliary_frozen.json').exists()
        write('results/auxiliary_frozen.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            status='post-hoc descriptive freeze; original NCM outcomes already known',
            files={str(p.relative_to(ROOT)):sha(p) for p in input_paths()}))
        return
    assert not sys.argv[1:]
    check()
    assert not (D/'results/auxiliary.json').exists(),'Refusing to replace auxiliary observations'
    with np.load(NCM/'results/native.npz') as archive:
        codes=archive['codes'][:,:2]
        offsets=archive['offsets']
        levels=archive['levels']
    report=json.loads((NCM/'results/summary.json').read_text())
    families=[]
    for f in range(16):
        coarse=[tuple(row[:int(offsets[9])]) for row in codes[f]]
        fine=[tuple(row[int(offsets[9]):]) for row in codes[f]]
        level_values=[]
        for k,length in enumerate(levels):
            pair=codes[f,:,int(offsets[k]):int(offsets[k+1])]
            vectors=[tuple(row) for row in pair]
            information=mutual([0,1],vectors)
            assert information == float(not np.array_equal(pair[0],pair[1]))
            level_values.append(dict(level=k+1,whole_vector_MI_bits=information,
                                     mean_position_MI_bits=float((pair[0]!=pair[1]).mean())))
        cf=mutual(coarse,fine)
        assert cf==float(coarse[0]!=coarse[1] and fine[0]!=fine[1])
        families.append(dict(source_id=report['E']['per_family'][f]['source_id'],levels=level_values,
                             branch_coarse_MI_bits=mutual([0,1],coarse),branch_fine_MI_bits=mutual([0,1],fine),
                             coarse_fine_MI_bits=cf))
    by_level=[dict(level=k+1,positions=int(levels[k]),
                   distinguishable_families=sum(f['levels'][k]['whole_vector_MI_bits']>0 for f in families),
                   average_branch_MI_bits=float(np.mean([f['levels'][k]['whole_vector_MI_bits'] for f in families])),
                   mean_position_MI_bits=float(np.mean([f['levels'][k]['mean_position_MI_bits'] for f in families])),
                   observed_code_ids=sorted(map(int,np.unique(codes[:,:,int(offsets[k]):int(offsets[k+1])])))) for k in range(12)]
    sensitivity=[]
    for threshold in (.70,.80,.90,.95):
        eligible=[]
        for r in report['E']['per_family']:
            ratio=r['input_ratio']
            good=(r['native_exact_A'] and r['native_exact_B'] and r['native_span']>.1
                  and abs(r['interaction_ratio'])<=.2 and ratio is not None and 0<=ratio<=1)
            eligible.append(good)
        sensitivity.append(dict(threshold=threshold,eligible=sum(eligible),
            input_dominant=sum(ok and r['input_ratio']>=threshold for ok,r in zip(eligible,report['E']['per_family'])),
            total_families=16))
    parity=[]
    for n in range(2,11):
        vectors=[v for v in itertools.product((0,1),repeat=n) if sum(v)%2==0]
        h=entropy(vectors)
        tc=sum(entropy([v[i] for v in vectors]) for i in range(n))-h
        pair_mi=[mutual([v[i] for v in vectors],[v[j] for v in vectors]) for i in range(n) for j in range(i)]
        assert np.isclose(h,n-1) and np.isclose(tc,1)
        if n>=3: assert np.allclose(pair_mi,0)
        # Any non-singleton Cartesian rectangle includes a parity-changing edge.
        safe=0
        for supports in itertools.product(((0,),(1,),(0,1)),repeat=n):
            corners=list(itertools.product(*supports))
            if all(sum(v)%2==0 for v in corners):
                assert all(len(s)==1 for s in supports)
                safe+=1
        assert safe==2**(n-1)
        parity.append(dict(n=n,joint_entropy_bits=h,total_correlation_bits=tc,
                           max_pairwise_MI_bits=max(pair_mi),safe_product_supports=safe,
                           status='Enumeration supports general analytical proof; no numerical Wyner optimization'))
    rows=[r for r in json.loads((ROOT/'benchmark_causal_revision/data/families.json').read_text())['rows'] if r['family']=='sudoku']
    target=[]
    for r in rows:
        a,b=r['oracle_solutions']
        changed=[j for j in range(81) if a[j]!=b[j]]
        assert len(changed)>=2
        mixed=a.copy();mixed[changed[0]]=b[changed[0]]
        assert mixed not in (a,b)
        target.append(dict(source_id=r['source_id'],differing_cells=len(changed),
                           hypothetical_uniform_two_solution_entropy_bits=1,
                           hypothetical_uniform_two_solution_Wyner_bits=1,
                           unique_branch_conditional_Wyner_bits=0,
                           status='Exact analytical target distribution, not sampled NCM output distribution'))
    result=dict(status='passed',scope='Post-hoc finite paired-input design; no population MI estimate or Wyner measurement of NCM',
                levels=by_level,per_family=families,
                mean_coarse_branch_MI_bits=float(np.mean([f['branch_coarse_MI_bits'] for f in families])),
                mean_fine_branch_MI_bits=float(np.mean([f['branch_fine_MI_bits'] for f in families])),
                mean_coarse_fine_MI_given_family_bits=float(np.mean([f['coarse_fine_MI_bits'] for f in families])),
                conditional_on_full_input='Zero analytically under deterministic argmax; do not infer from one observation per input',
                threshold_sensitivity=sensitivity,parity_checks=parity,two_solution_target_checks=target)
    write('results/auxiliary.json',result)
    check()
    print(json.dumps({k:result[k] for k in ('levels','mean_coarse_fine_MI_given_family_bits','threshold_sensitivity')},indent=2))


if __name__=='__main__': main()

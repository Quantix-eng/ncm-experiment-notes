"""Recalculate this study's recorded-data checks; no model inference."""
from pathlib import Path
import json
import math
from statistics import mean
ROOT = Path(__file__).resolve().parents[1]
def load(name):
    config = json.loads((ROOT/'experiment.json').read_text())
    paths = [ROOT.parents[1]/x for x in config['records'] if Path(x).name==name]
    if not paths:
        paths = list((ROOT/'recorded').rglob(name))
    if len(paths) != 1:
        raise ValueError(f'Expected one {name}, found {len(paths)}')
    return json.loads(paths[0].read_text())
def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-12),(a,b)
def emit(result):
    print(json.dumps({'passed':True,'scope':'recorded-data recalculation, not a new model run','result':result},indent=2))

SUFFIXES=['']
results=[]
for suffix in SUFFIXES:
    a=load('test'+suffix+'_observations.json');s=load('test'+suffix+'_summary.json')
    families={}
    for row in a['crossed']:
        values=[x['nll'] for x in row['values']]
        y00,y10,y11,y01,wrong,backups,controls=values
        d={'D':y10-y00,'R':y10-y11,'I':y10+y01-y11-y00,'S':wrong-y11,'B':backups-controls}
        families.setdefault(row['family'],[]).append(d)
    assert len(families)==s['n_families']
    means={k:mean(mean(v[k] for v in group) for group in families.values()) for k in ['D','R','I','S','B']}
    for k,v in means.items():close(v,s['contrasts'][k]['mean'])
    results.append({'head':s['head'],'families':len(families),'recomputed_means':means,'matching_interval':s['contrasts']['S']['ci95']})
emit(results)

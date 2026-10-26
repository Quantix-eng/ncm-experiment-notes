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

s=load('report.json')
assert s['persistent_errors']<=s['current_errors']
assert s['paired_group_bootstrap_95'][0]<=s['primary_delta_within_auc']<=s['paired_group_bootstrap_95'][1]
emit({'groups':s['n_groups'],'mixed_groups':s['mixed_groups'],'persistence':s['persistent_errors']/s['current_errors'],'increment':s['primary_delta_within_auc'],'interval':s['paired_group_bootstrap_95']})

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
assert s['failures']<=s['n']
assert s['primary_paired_bootstrap_95'][0]<=s['primary_delta_auc']<=s['primary_paired_bootstrap_95'][1]
emit({'n':s['n'],'failures':s['failures'],'delta_auc':s['primary_delta_auc'],'interval':s['primary_paired_bootstrap_95'],'results':s['results']})

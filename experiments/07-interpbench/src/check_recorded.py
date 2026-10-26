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

s=load('c2_summary.json')
assert sum(x['queries'] for x in s['tasks'].values())==s['n_queries']
assert sum(x['families'] for x in s['tasks'].values())==s['n_families']
assert sum(x['craft_RASP_disagreements'] for x in s['tasks'].values())==0
assert s['E']['max_regenerated_donor_native_error']==0
emit({'queries':s['n_queries'],'families':s['n_families'],'macro_averages':s['macro_average_over_tasks']})

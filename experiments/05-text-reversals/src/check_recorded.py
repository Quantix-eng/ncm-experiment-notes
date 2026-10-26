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

s=load('lm1b_generation.json')
assert len(s['all_positions']['agreement_by_level'])==len(s['levels'])
assert all(x['codes_equal'] and x['all_readouts_equal'] for x in s['replays'])
assert s['all_positions']['n_positions']==s['all_positions']['n_trajectories']*128
emit({'transition_at_level9':s['all_positions']['transitions'][7],'anchor_warning':s['anchor_warning'],'trajectories':s['all_positions']['n_trajectories']})

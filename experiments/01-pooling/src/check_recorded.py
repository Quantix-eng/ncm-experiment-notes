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

p=load('pooling_algebra.json')
for w in p['windows']:
    start=w['start_token'];stop=w['stop_token_exclusive']
    assert start==math.floor(w['slot']*89/9) and stop==math.ceil((w['slot']+1)*89/9)
    counts=w['token_counts']
    assert all(counts[d]==1 for d in range(1,10))
a=load('generator_alignment.json')
assert a['native']['level5_match']==1 and a['weighted_1']['level5_match']<0.02
emit({'histogram_windows_checked':len(p['windows']),'native_generator_agreement':a['native']['level5_match'],'weighted_generator_agreement':a['weighted_1']['level5_match']})

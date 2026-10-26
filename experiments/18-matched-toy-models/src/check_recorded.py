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

c=load('calibration.json');s=load('summary.json')
assert c['status']=='no_common_informative_dose_fallback'
assert s['calibration']['selected_rho']==c['selected_rho']
models={(x['architecture'],x['seed']) for x in s['cells']}
assert len(models)==6
emit({'models':len(models),'damage_gate':c['status'],'selected_fallback_dose':c['selected_rho'],'recovery_by_model_and_cut':[{'architecture':x['architecture'],'seed':x['seed'],'cut':x['cut'],'R':x['C']['R']['mean']} for x in s['cells']]})

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

checked=[]
for path in sorted((ROOT/'recorded').rglob('*_audit.json')):
    data=json.loads(path.read_text())
    for name,stratum in data.items():
        if not isinstance(stratum,dict) or 'levels' not in stratum:continue
        for level in stratum['levels']:
            o=level['occupancy'];assert sum(v>0 for v in o['counts'])==o['used']
            assert not o['empirical_constant_level']
            checked.append({'task':path.stem,'stratum':name,'level':level['level'],'used_codes':o['used']})
assert checked
emit({'native_levels_and_strata_checked':len(checked),'checks':checked})

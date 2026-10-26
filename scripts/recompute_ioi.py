"""Recompute means from saved per-intervention observations, without model calls."""
from pathlib import Path
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT/'historical/Experience_NCM/analyse_fondamentale_NCM/validation_externe_directeur/01_ioi/results'


def summarize(suffix):
    observations = json.loads((RECORDS/f'test{suffix}_observations.json').read_text())
    saved = json.loads((RECORDS/f'test{suffix}_summary.json').read_text())
    values = {k: [] for k in ('D','R','I','S','B')}
    families = {}
    for row in observations['crossed']:
        families[row['family']] = families.get(row['family'], 0)+1
        a,b,c,d,e,f,g = [v['nll'] for v in row['values']]
        for key, value in {'D':b-a, 'R':b-c, 'I':b+d-c-a, 'S':e-c, 'B':f-g}.items():
            if not math.isfinite(value): raise ValueError('Non-finite observation')
            values[key].append(value)
    assert len(families) == saved['n_families'] == 512
    assert set(families.values()) == {2}
    means = {key: statistics.mean(v) for key,v in values.items()}
    for key,value in means.items():
        assert math.isclose(value, saved['contrasts'][key]['mean'], rel_tol=0, abs_tol=1e-12), key
    return {'head':saved['head'], 'families':len(families), 'interventions':len(observations['crossed']), 'means_nats':means}


if __name__ == '__main__':
    print(json.dumps({'passed':True, 'scope':'recalculated means; model outputs and confidence intervals are not rerun',
                      'results':[summarize(s) for s in ('','_L9H6','_L10H0')]}, indent=2))

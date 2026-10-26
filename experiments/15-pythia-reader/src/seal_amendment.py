"""Explicit software amendment: preserve original seal and all observations; never alter tolerances."""
from common import *
import datetime

archive=D/'archives/v1_native_parity_failure_20261025T174632Z'
original=json.loads((archive/'protocol.json').read_text())
protocol=dict(original)
protocol['version']='1.1'
protocol['software_amendment']='AMENDEMENT_LOGICIEL.md: native LayerNorm kernel plus denominator ratio; fixed thresholds unchanged'
protocol['amended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
assert protocol['stop']==original['stop']
write('protocol.json',protocol)
old=json.loads((archive/'results/protocol_frozen.json').read_text())
paths=[Path(p) for p in old['files']]+[D/'AMENDEMENT_LOGICIEL.md',D/'scripts/seal_amendment.py']
write('results/protocol_frozen.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    version='1.1',previous_seal=str(archive/'results/protocol_frozen.json'),
    caveat='software amendment after development; before any test output',
    files={str(p.resolve()):sha(p) for p in paths},environment=old['environment']))
print('amendment sealed')

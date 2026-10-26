"""Seal the explicitly documented measurement-only FP64 amendment, before test."""
from common import *
import datetime
archive=D/'archives/v1_1_before_fp64_measurement'
original=json.loads((archive/'protocol.json').read_text())
protocol=dict(original);protocol['version']='1.2'
protocol['software_amendment_fp64']='AMENDEMENT_FP64.md: direct-effect measurement only; native model FP32 and scientific criteria unchanged'
protocol['amended_fp64_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
assert protocol['stop']==original['stop']
write('protocol.json',protocol)
old=json.loads((archive/'results/protocol_frozen.json').read_text())
paths=[Path(p) for p in old['files']]+[D/'AMENDEMENT_FP64.md',D/'scripts/seal_fp64.py',D/'scripts/recompute_de_fp64.py']
write('results/protocol_frozen.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    version='1.2',previous_seal=str(archive/'results/protocol_frozen.json'),
    caveat='measurement hardening after development; before any test output',
    files={str(p.resolve()):sha(p) for p in paths},environment=old['environment']))
print('FP64 measurement amendment sealed')

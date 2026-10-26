"""Prepare official-partition inputs without reading model predictions; seal code and assets."""
from common import *
import datetime
import pyarrow.parquet as pq

assert sha(D/'weights/model.safetensors')=='29d2e457a664e41c12c735f20a36dc0956a665f614a54ce5db21a32e75965270'
tok=AutoTokenizer.from_pretrained(D/'weights',local_files_only=True)
sources={}
seen=set()
for split,n,seed in [('validation',128,2026102501),('test',512,2026102502)]:
    eligible=[]
    for row,text in enumerate(pq.read_table(D/'data'/f'{split}.parquet')['text'].to_pylist()):
        ids=tok(text,add_special_tokens=False)['input_ids']
        if len(ids)<33: continue
        key=hashlib.sha256(text.encode()).hexdigest()
        if key in seen: continue
        seen.add(key)
        eligible.append(dict(row=row,text_sha256=key,tokens=ids[:33]))
    assert len(eligible)>=n,(split,len(eligible))
    order=np.random.default_rng(seed).permutation(len(eligible))[:n]
    records=[eligible[int(i)] for i in order]
    # Deterministic distinct donors, fixed before scores; no source's own activation is its donor.
    for i,r in enumerate(records): r['donors']=[(i+1)%n,(i+17)%n]
    sources[split]=dict(eligible=len(eligible),selected=n,records=records)
write('data/sequences.json',sources)
protocol=dict(version='1.0',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    model='EleutherAI/pythia-160m',model_revision='50f5173d932e8e61f858120bcb800b97af589f46',
    dataset='Salesforce/wikitext',config='wikitext-2-raw-v1',
    sampling='one first-33-token window per eligible nonduplicate row, deterministic random sample without replacement',
    dev='128 official validation rows',test='512 official test rows',
    length_context=32,target='token33',layers=[11],heads=[2,10],donors_per_sequence=2,
    conditions=['Y00','Y10','Y11','Y01','Y1wrong'],
    readout_contrast='same residual crossed with native / own-corruption / other-donor final-LN denominator',
    neural_continuation='empty after final parallel attention contribution; MLP independent of intervention',
    metrics=['NLL','whole-vocabulary exactness','centered target logit','published raw self-repair','LN scale ratio'],
    high_direct_effect_selection=False,ratio_clipping=False,
    bootstrap='10000 sequence-family resamples; descriptive95%; shared donor dependence not removed',
    stop='technical parity >2e-4 logit abs, residual max >1e-5, or nonzero downstream-MLP change; preserve failures',
    classification='No universal-class prediction. Separate learned continuation from final reader.',
    test_gate='development technical checks pass; no damage or recovery favorable-effect gate for structural falsifier')
write('protocol.json',protocol)
paths=[D/'protocol.json',D/'data/sequences.json']
paths+=sorted((D/'scripts').glob('*.py'))
paths+=sorted((D/'weights').glob('*'))
paths+=sorted((D/'data').glob('*.parquet'))
paths+=sorted((D/'sources').glob('*.json'))
paths+=[Path(__import__('inspect').getfile(transformers.models.gpt_neox.modeling_gpt_neox))]
write('results/protocol_frozen.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    caveat='local checksum seal, not independent preregistration',files={str(p.resolve()):sha(p) for p in paths},
    environment=dict(torch=torch.__version__,numpy=np.__version__,transformers=transformers.__version__)))
print(json.dumps({k:{'eligible':v['eligible'],'selected':v['selected']} for k,v in sources.items()}))

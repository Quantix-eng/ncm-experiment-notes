"""Editorial QA: source values, semantic consistency and immutable evidence.

No PyTorch import, model execution, refitting or new inferential analysis.
"""
from pathlib import Path
import json,hashlib,re,urllib.parse
from datetime import datetime,timezone
D=Path(__file__).resolve().parent;V=D.parent
BASE=D/'archives/avant_integration_resultats_20261025'
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()
text=(D/'PAPER_REVISE.md').read_text();before=(BASE/'PAPER_REVISE.md').read_text()
baseline=read(BASE/'baseline.json');assert sha(BASE/'PAPER_REVISE.md')==baseline['previous_manuscript_sha256']
for p,h in baseline['scientific_files_before_editorial_integration'].items():assert sha(V/p)==h,p
claims=read(D/'CLAIMS_INTEGRATION.json')['generated_numeric_claims']
for claim in claims:
    value=read(V/claim['source'])
    for k in claim['keys']:value=value[k]
    assert value==claim['value'] and format(value,claim['format'])==claim['rendered']
    assert claim['rendered'] in text,claim

def section(s,start,end):return s[s.index(start):s.index(end)]
old=section(before,'## 5. Results','## 6.')
new=section(text,'## 5. Results','### 5.8 IOI')
new=new.replace('### 5.2 NCM native performance is at ceiling','### 5.2 Native performance is at ceiling')
assert old.strip()==new.strip(),'Historical results changed'
assert section(before,'## 7.','## 8.')==section(text,'## 7.','## 8.'),'Theory changed'

static_checks=0
native=read(V/'01_ioi/results/test_observations.json')['native']
expected=[[512,476],[498,449],[506,446]]
for k in range(3):
    for m,metric in enumerate(['correct_candidate','correct_vocab']):
        assert sum(r[k][metric] for r in native)==expected[k][m];static_checks+=1
for suffix in ['', '_L9H6','_L10H0']:
    obs=read(V/f'01_ioi/results/test{suffix}_observations.json')
    s=read(V/f'01_ioi/results/test{suffix}_summary.json')
    # Recalculate recipient/donor policy alignment from individual E records.
    for other in (1,2):
        for policy in ('regenerated','recipient_native','donor_native'):
            records=[]
            for r in obs['E']:
                if r['other']!=other or r['recipient']==r['donor']:continue
                wanted=policy
                if policy=='recipient_native':wanted='fixed_left' if r['recipient']==0 else 'fixed_right'
                if policy=='donor_native':wanted='fixed_left' if r['donor']==0 else 'fixed_right'
                if r['policy']==wanted:records.append(r)
            entry=s['E'][f'other{other}_{policy}']
            assert entry['n']==len(records)
            for metric,out in [('correct_candidate','recipient_candidate_correct'),('correct_vocab','recipient_vocab_correct')]:
                assert sum(r['values'][metric] for r in records)==entry[out];static_checks+=1
assert read(V/'01_ioi/results/test_summary.json')['E']==read(V/'01_ioi/results/test_L9H6_summary.json')['E']
py=read(V/'03_pythia/results/test_summary.json');rows=read(V/'03_pythia/results/test_observations.json')
assert py['n_sequences']==512 and py['n_interventions']==2048
assert [sum(r['correct'][i] for r in rows) for i in range(5)]==[604,598,598,604,597]
assert all(r['correct'][1]==r['correct'][2] for r in rows);static_checks+=7
normalized=text.replace('−','-')
for metric,fmt in [('D','.6f'),('R','.6f'),('self_repair','.6f'),('reader_logit_effect','.6f'),('I','.9f'),('S','.9f'),('scale_ratio','.8f')]:
    assert format(py['metrics'][metric]['mean'],fmt) in normalized,(metric,fmt);static_checks+=1
for head in ['2','10']:
    for metric in ['D','R','S','delta_DE']:
        assert format(py['by_head'][head][metric]['mean'],'.8f') in normalized,(head,metric);static_checks+=1
ib=read(V/'02_interpbench/results/c2_summary.json');execution=read(V/'02_interpbench/results/c2_execution.json')
assert ib['n_families']==1024 and ib['n_queries']==5632
assert execution['oracle_disagreements']==0 and execution['empty_oracle_masks']==14
assert ib['E']['conditions']==55296 and ib['E']['regenerated_conditions']==18432
assert ib['E']['max_regenerated_donor_native_error']==0;static_checks+=7
assert [(ib['tasks'][str(c)]['native_exact_defined'],ib['tasks'][str(c)]['native_scorable']) for c in [13,18,19,21]]==[(256,256),(119,256),(253,253),(256,256)]
for c,aligned,unaligned in [(13,(1024,1024),(512,512)),(18,(116,256),(232,512)),(19,(507,507),(759,759)),(21,(768,768),(1277,1280))]:
    for kind,(agreement,n) in [('aligned',aligned),('unaligned',unaligned)]:
        row=ib['tasks'][str(c)]['interchange_conformity'][kind]
        assert (row['agreement'],row['n'])==(agreement,n);static_checks+=1
for method in ['C_without_frozen','C_without_mismatch','C_full']:
    for threshold in ['0.5','0.75','1.0']:
        row=ib['macro_average_over_tasks'][method][threshold]
        for metric in ['coverage','risk_network','risk_oracle']:
            assert f'{100*row[metric]:.2f}' in text;static_checks+=1
c3=read(V/'02_interpbench/results/c3_padding.json')['summary']
assert c3['18']['padded_network_program']==122 and c3['18']['padded_Craft_program']==256
assert ib['tasks']['18']['native_exact_defined']==119
assert read(V/'02_interpbench/results/c2_verification.json')['passed'];static_checks+=4

missing=[];links=0
for path in [D/'PAPER_REVISE.md',D/'PREUVES_INTEGRATION.md']:
    for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',path.read_text()):
        if '://' in target or target.startswith('#') or target.startswith('mailto:'):continue
        decoded=urllib.parse.unquote(target.split('#')[0].strip('<>'))
        assert decoded,(path,target)
        if not (path.parent/decoded).exists():missing.append([path.name,target])
        links+=1
assert not missing,missing
for stale in ['Prospective external-validation programme','The first planned external case','All proposed validation campaigns remain separate','Eight learned checkpoints are included','It reports no new experiment']:
    assert stale not in text,stale
for required in ['### 4.6 IOI','### 4.7 Pythia','### 4.8 InterpBench','### 5.8 IOI','### 5.9 Pythia','### 5.10 InterpBench','## Appendix A.','15 distinct learned checkpoints']:
    assert required in text,required
abstract=section(text,'## Abstract','## 1.')
assert len(abstract.split())<300
assert text.count('## 9. Conclusion')==1 and text.count('## References')==1
assert 'The fixed 0.01-nat specificity criterion is not met for L10H0.' in text
assert 'An incremental predictive advantage from all five conditions is not established.' in text
assert 'proxy for one-shot generation' in section(text,'## 9.','## References')
assert 'no independent human replication has been performed' in text
result=dict(utc=datetime.now(timezone.utc).isoformat(),passed=True,
    manuscript_sha256=sha(D/'PAPER_REVISE.md'),words=len(text.split()),
    generated_numeric_insertions_checked=len(claims),additional_static_checks=static_checks,
    scientific_artifacts_unchanged=len(baseline['scientific_files_before_editorial_integration']),
    historical_result_sections_unchanged=True,theory_section_unchanged=True,
    local_links_checked=links,missing_local_links=missing,stale_prospective_status_removed=True,
    new_neural_experiments=False,independent_scientific_review=False,
    scope='editorial consistency and evidence integrity, not a new scientific validation')
(D/'VERIFICATION_INTEGRATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))

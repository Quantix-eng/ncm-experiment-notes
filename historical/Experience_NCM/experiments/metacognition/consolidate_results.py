"""Read-only evidence audit; no NCM execution, fitting, new split or experiment."""
import json
from pathlib import Path
import re
import numpy as np
from run_pilot import ROOT, file_hash, load_boards, write_json, verify_assets
from confirmation_corpus import known_transform_key
from features import CELL_POS, GROUPS

BASE = ROOT / 'audit/metacognition'


def read(relative):
    return json.loads((BASE / relative).read_text())


def brute_auc(y, score):
    positive, negative = score[y == 1], score[y == 0]
    assert len(positive) and len(negative)
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0) + .5 * (comparisons == 0)).mean())


def main():
    verify_assets(ROOT, "hard", include_data=True)
    evidence = {}
    checked = {}
    def check(path, digest):
        assert path.exists(), str(path)
        assert file_hash(path) == digest, str(path)
        checked[str(path.relative_to(ROOT))] = digest
    def record(relative):
        path = BASE / relative
        value = json.loads(path.read_text())
        evidence[relative] = dict(sha256=file_hash(path), status=value.get('status'))
        return value
    def chain(directory, source_base, model_folder=None, model_key='model_files'):
        lock = record(f'{directory}/lock.json')
        for name,digest in lock['source_files'].items(): check(source_base / name, digest)
        if model_folder is not None:
            for name,entry in lock[model_key].items():
                check(BASE / directory / model_folder / name, entry if isinstance(entry,str) else entry['sha256'])
        return lock

    split = record('split-v0.2.json')
    partitions = split['partition_indices']
    assert [len(partitions[k]) for k in ('train','calibration','test')] == [1000,500,500]
    assert len(set(sum(partitions.values(),[]))) == 2000
    group_hashes = split['solution_group_sha256']
    solution_sets = [{group_hashes[i] for i in partitions[k]} for k in ('train','calibration','test')]
    assert not solution_sets[0] & solution_sets[1] and not solution_sets[0] & solution_sets[2] and not solution_sets[1] & solution_sets[2]

    chain('final-v0.3',ROOT,'')
    h1 = record('final-v0.3/analysis/report.json')
    h1_verification = record('final-v0.3/analysis/verification.json')
    assert h1['status']=='complete' and h1_verification['status']=='passed'
    check(BASE/'final-v0.3/analysis/report.json',h1_verification['report_sha256'])
    check(BASE/'final-v0.3/lock.json',h1['lock_sha256'])
    for name,digest in h1['artifacts'].items(): check(BASE/'final-v0.3/analysis'/name,digest)
    extraction = record('final-v0.3/held-out/report.json')
    for name,entry in extraction['artifacts'].items(): check(BASE/'final-v0.3/held-out'/name,entry['sha256'])
    assert extraction['status']=='complete'
    with np.load(BASE/'final-v0.3/analysis/test-predictions.npz',allow_pickle=False) as data:
        assert data['indices'].tolist()==partitions['test']
        assert len(data['failure'])==500 and int(data['failure'].sum())==68
        auc0=brute_auc(data['failure'],data['rich'])
        auc1=brute_auc(data['failure'],data['rich_hidden_primary'])
        summary=brute_auc(data['failure'],data['summary'])
        assert np.isclose(auc1-auc0,h1['primary_delta_auc'],atol=1e-12)
        assert np.isclose(auc0,h1['results']['rich']['raw']['auc'],atol=1e-12)
        assert np.isclose(summary,h1['results']['summary']['raw']['auc'],atol=1e-12)

    chain('sensitivity-v0.3',ROOT/'experiments/metacognition','',model_key='models_sha256')
    secondary=record('sensitivity-v0.3/analysis/report.json')
    secondary_verification=record('sensitivity-v0.3/analysis/verification.json')
    check(BASE/'sensitivity-v0.3/analysis/report.json',secondary_verification['report_sha256'])
    for name,digest in secondary['artifacts'].items(): check(BASE/'sensitivity-v0.3/analysis'/name,digest)
    assert secondary_verification['status']=='passed'

    local=record('correction-v0.1/development/report.json')
    local_verification=record('correction-v0.1/development/verification.json')
    check(BASE/'correction-v0.1/development/report.json',local_verification['report_sha256'])
    assert set(local['pool_groups'])|set(local['holdout_groups'])==set(partitions['train'])
    assert not set(local['pool_groups'])&set(local['holdout_groups'])
    for cv in local['cv'].values():
        assert not set(cv['train'])&set(cv['valid'])
        assert not (set(cv['train'])|set(cv['valid'])) & set(local['holdout_groups'])
    for name,digest in local['artifacts'].items(): check(BASE/'correction-v0.1/development'/name,digest)

    lock = chain('correction-v0.1/external',ROOT,'models')
    correction=record('correction-v0.1/external/analysis/report.json')
    verification=record('correction-v0.1/external/analysis/verification.json')
    check(BASE/'correction-v0.1/external/analysis/report.json',verification['report_sha256'])
    check(BASE/'correction-v0.1/external/lock.json',correction['lock_sha256'])
    assert verification['status']=='passed' and correction['status']=='complete'
    for name,digest in correction['artifacts'].items(): check(BASE/'correction-v0.1/external/analysis'/name,digest)
    manifest=record('correction-v0.1/external/corpus-manifest.json')
    check(BASE/'correction-v0.1/external/corpus-manifest.json',lock['corpus_manifest_sha256'])
    check(BASE/'correction-v0.1/external/corpus.npz',manifest['corpus_sha256'])
    traces=record('correction-v0.1/external/traces/report.json')
    check(BASE/'correction-v0.1/external/lock.json',traces['lock_sha256'])
    for name,digest in traces['arrays'].items():check(BASE/'correction-v0.1/external/traces'/name,digest)
    with np.load(BASE/'correction-v0.1/external/corpus.npz',allow_pickle=False) as corpus:
        puzzles,solutions=corpus['puzzles'],corpus['solutions']
        assert len(puzzles)==1000 and (np.count_nonzero(puzzles,axis=1)==30).all()
        assert np.all((puzzles==0)|(puzzles==solutions))
        assert all(np.all(np.sort(solutions[:,unit],axis=1)==np.arange(1,10)) for unit in GROUPS)
        old=load_boards(ROOT/'data/sudoku/hard/solutions.pickle')
        official_keys={known_transform_key([s[i]for i in CELL_POS])for s in old}
        new_keys={known_transform_key(s.tolist())for s in solutions}
        pilot=read('confirmation-generator-pilot/puzzles.json')
        pilot_keys={known_transform_key(row['solution'])for row in pilot}
        assert len(new_keys)==1000 and not new_keys & (official_keys|pilot_keys)

    with np.load(BASE/'correction-v0.1/external/analysis/predictions.npz',allow_pickle=False) as data:
        rows=data['rows'];groups=data['group'];target=data['persistent']
        raw=BASE/'correction-v0.1/external/traces'
        mask=np.load(raw/'current_error.npy',allow_pickle=False)
        persistent=np.load(raw/'persistent.npy',allow_pickle=False)
        assert np.array_equal(rows,np.flatnonzero(mask))
        assert np.array_equal(target,persistent[rows])
        assert len(rows)==19941 and int(target.sum())==1324
        means={};paired=[]
        for model in ('visible','visible_hidden'):
            per={}
            for g in np.unique(groups):
                selected=groups==g;y=target[selected]
                if y.min()!=y.max():per[int(g)]=brute_auc(y,data[model][selected])
            assert len(per)==145
            means[model]=float(np.mean(list(per.values())))
            assert np.isclose(means[model],correction['results'][model]['within_puzzle_auc'],atol=1e-12)
            if model=='visible':baseline=per
            else:paired=[per[g]-baseline[g]for g in sorted(per)]
        assert np.isclose(np.mean(paired),correction['primary_delta_within_auc'],atol=1e-12)

    previous=record('correction-v0.1/external/format-failure-01/lock.json')
    assert previous['source_files'].keys()==lock['source_files'].keys()
    changed=[k for k in previous['source_files'] if previous['source_files'][k]!=lock['source_files'][k]]
    assert changed==['experiments/metacognition/correction/external.py']
    check(BASE/'correction-v0.1/external/format-failure-01/external.py',previous['source_files'][changed[0]])
    assert previous['model_files']==lock['model_files']
    assert previous['corpus_manifest_sha256']==lock['corpus_manifest_sha256']
    check(BASE/'correction-v0.1/external/format-failure-01/lock.json',lock['pre_inference_format_repair']['previous_lock_sha256'])
    failure=record('correction-v0.1/external/format-failure-01/failure.json')
    assert failure['external_predictions_observed']==0
    invalid=record('correction-v0.1/training-uniqueness-invalid-format-01.json')
    assert invalid['status']=='invalid'
    uniqueness=record('correction-v0.1/training-uniqueness.json')
    assert uniqueness['counts']=={'0':0,'1':1000,'2':0}
    for relative in ['pilot-v0.2/report.json','development-primary-v0.2/report.json','development-controls-v0.2/report.json',
                     'development-v0.3/report.json','stability-v0.3/report.json','stability-v0.3-failed-01/report.json',
                     'correction-v0.1/training-label-audit.json']:
        record(relative)

    docs=['SYNTHESE_METACOGNITION.md','RAPPORT_RESULTATS_METACOGNITION.md','RAPPORT_CORRECTION_FUTURE.md',
          'RAPPORT_DEVELOPPEMENT_METACOGNITION.md','RAPPORT_PILOTE_METACOGNITION.md','RAPPORT_METACOGNITION.md',
          'PLAN_RECHERCHE_METACOGNITION.md','FEUILLE_DE_ROUTE.md']
    destination=BASE/'consolidation-v1.json'
    missing=[]
    for name in docs:
        path=ROOT/name
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' in target or target.startswith('#'):continue
            target=target.strip('<>').split('#')[0]
            resolved=path.parent/target
            if not resolved.exists() and resolved!=destination:missing.append(dict(document=name,target=target))
    assert not missing,missing
    out=dict(status='passed',date='2026-10-23',scope='consolidation_only_no_inference_no_training_no_new_bootstrap',
        auditor_sha256=file_hash(Path(__file__)),pinned_hard_assets_verified=True,evidence=evidence,verified_hashes=checked,documents=docs,document_sha256={name:file_hash(ROOT/name)for name in docs},
        results=dict(H1=dict(n=500,failures=68,baseline_auc=auc0,internal_auc=auc1,summary_auc=summary,
                            delta=h1['primary_delta_auc'],interval=h1['primary_paired_bootstrap_95']),
                     future_correction=dict(n_puzzles=1000,n_errors=19941,n_mixed_puzzles=145,
                         baseline_within_auc=means['visible'],internal_within_auc=means['visible_hidden'],
                         delta=correction['primary_delta_within_auc'],interval=correction['paired_group_bootstrap_95'])),
        checks=['group split disjointness','source/model/artifact hashes and verification-report links',
                'direct pairwise AUC replay, no refit','new corpus solution validity and known-transform exclusion',
                'same corpus and coefficients before/after pre-inference prompt repair','invalid audit excluded','local documentation links'],
        caveats=['Existing independent replays are arithmetic checks in this workspace, not a replication by another team or machine',
                 'No new uniqueness solving or statistical power simulation performed during consolidation',
                 'No native metacognition proof; neither result exhausts all observable predictors or internal readouts'])
    write_json(destination,out)
    assert destination.is_file()  # Resolve the audit document's own newly created link.
    print(json.dumps(dict(status='passed',evidence_records=len(evidence),verified_files=len(checked),broken_document_links=0,
                         output=str(destination))))

if __name__=='__main__':main()

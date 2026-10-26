"""Independent artifact checks, not a reimplementation of feature extraction."""
from common import *
import pickle
from sklearn.metrics import roc_auc_score

def main():
    checks=[]
    def check(name,condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    manifest=json.loads((ROOT/'results/input_manifest.json').read_text())
    for rel,digest in manifest.items(): check('input_sha256:'+rel,sha(ROOT/rel)==digest)
    split=json.loads((ROOT/'results/splits.json').read_text()); sets=[set(split[n]) for n in ['train','validation','test']]
    check('all_2000_partitioned',len(set.union(*sets))==2000 and sum(map(len,sets))==2000)
    for i in range(3):
        for j in range(i): check(f'partition_disjoint_{i}_{j}',not sets[i]&sets[j])
    for d in ['easy','medium','hard']:
        a=np.load(ROOT/f'results/{d}.npz'); y=np.all(a['decoded'][:,-1]==a['reference'],axis=1)
        met=json.loads((ROOT/f'results/reproduction_{d}.json').read_text())['metrics']
        check('reproduction_n_'+d,len(y)==met['n']==2000)
        check('exact_predictions_'+d,int(y.sum())==met['solved'])
        check('reference_fixed_positions_'+d,a['reference'].shape==(2000,89) and np.all(a['reference'][:,SEP]==10))
        # Independent Sudoku validity by set checks, without observed()/metrics().
        good=[]
        for pred,prompt in zip(a['decoded'][:,-1],a['prefix']):
            c=pred[CELL]; giv=prompt[1:-1][CELL]
            good.append(all(set(c[u])==set(range(1,10)) for u in UNITS) and np.all(pred[SEP]==10) and np.all((giv==0)|(giv==c)))
        check('independent_validity_'+d,abs(np.mean(good)-met['valid_and_clues_rate'])<1e-12)
        check('finite_features_'+d,np.isfinite(a['hidden']).all() and np.isfinite(a['baseline']).all())
    reconstruction_errors=json.loads((ROOT/'results/tokenizer_errors.json').read_text())
    check('rare_reconstruction_errors_upstream_replayed',all(case['upstream_identical'] for err in reconstruction_errors.values() for case in err['cases']))
    probes=json.loads((ROOT/'results/probes.json').read_text()); pred=np.load(ROOT/'results/probe_predictions.npz'); te=pred['test']; y=pred['y'][te]
    k=P['primary_level']; comb=pred[f'level{k}_combined'][te]; base=pred[f'level{k}_baseline'][te]
    check('primary_delta_recomputed',abs(roc_auc_score(y,comb)-roc_auc_score(y,base)-probes['primary']['delta'])<1e-12)
    check('constant_auc_half',abs(probes['levels'][k-1]['models']['prior']['auc']-.5)<1e-12)
    frozen=json.loads((ROOT/'results/protocol_frozen.json').read_text()); check('protocol_unmodified',sha(ROOT/'protocol.json')==frozen['protocol_sha256'])
    check('split_unmodified',sha(ROOT/'results/splits.json')==frozen['splits_sha256'])
    # TODO is updated for completion after study; its frozen digest is retained as provenance.
    models=pickle.loads((ROOT/'results/probe_models.pkl').read_bytes()); a=np.load(ROOT/'results/hard.npz')
    h=a['hidden'][:,k-1].astype(float); b=a['baseline'][:,k-1].astype(float)
    for name,x in [('baseline',b),('combined',np.column_stack([b,h]))]:
        pp=models[(k-1,name)].predict_proba(x[te])[:,1]
        check('probe_replay_'+name,np.allclose(pp,pred[f'level{k}_{name}'][te],rtol=1e-12,atol=1e-12))
    ext=json.loads((ROOT/'data/external.json').read_text()); check('external_256',len(ext['rows'])==P['external_n'])
    from solvers import exact_cover_count,backtracking
    for j in [0,1,31,63,127,255]:
        row=ext['rows'][j]; check('external_uniqueness_replay_'+str(j),exact_cover_count(row['puzzle'])==1 and backtracking(row['puzzle'],limit=1)[0]==row['solution'])
    constant=np.load(ROOT/'results/constant_level_predictions.npz')
    check('constant_level_sham_identical',np.array_equal(constant['baseline'],constant['sham']))
    ca=np.load(ROOT/'results/causal_predictions.npz'); check('causal_sham_identical',np.array_equal(ca['baseline'],ca['sham']))
    check('causal_test_only',np.array_equal(ca['indices'],te))
    causal_report=json.loads((ROOT/'results/causal.json').read_text())['conditions']
    check('confidence_projection_removed',all(v['confidence_projection_rms_after']<max(1e-5,1e-5*v['confidence_projection_rms_before']) for v in causal_report['confidence']['perturbation']))
    target=np.array([v['delta_rms'] for v in causal_report['confidence']['perturbation']])
    for i in range(P['causal_random_directions']):
        control=np.array([v['delta_rms'] for v in causal_report[f'random_{i}']['perturbation']])
        check('causal_norm_matching_'+str(i),np.allclose(target,control,rtol=1e-5,atol=1e-7))
    upstream=json.loads((ROOT/'results/upstream_parity.json').read_text())
    check('untouched_upstream_parity_all_checkpoints',all(v['codes_equal'] and v['outputs_equal'] for v in upstream.values()))
    stab=json.loads((ROOT/'results/stability.json').read_text())
    check('repeated_codes_stable',all(stab['contrasts'][f'repeat_{i}']['code_disagreements']==0 for i in range(3)))
    overlap=json.loads((ROOT/'results/cross_corpus_overlap.json').read_text())
    check('cross_corpus_shared_families_documented',all(v['solution_symmetry_overlap']==2000 and v['exact_solutions_same_order'] for v in overlap['overlap'].values()))
    transfer=json.loads((ROOT/'results/cross_checkpoint_probe_transfer.json').read_text())
    check('transfer_only_600_heldout_families',all(v['baseline']['n']==600 for v in transfer['results'].values()))
    # Save source and result hashes after the verification itself is fully written.
    final={str(f.relative_to(ROOT)):sha(f) for sub in ['scripts','results','sources','figures','vendor'] for f in (ROOT/sub).rglob('*')
        if f.is_file() and f.name not in ['final_manifest.json','verification.json','progress.json'] and '__pycache__' not in str(f) and '/.git/' not in str(f)}
    for f in [ROOT/'TODO.md',ROOT/'JOURNAL.md',ROOT/'protocol.json',ROOT/'run.sh',ROOT/'RAPPORT.md',ROOT/'README.md',ROOT/'requirements.txt',ROOT/'sources_utilisees.zip']:
        final[str(f.relative_to(ROOT))]=sha(f)
    write(ROOT/'results/final_manifest.json',final)
    write(ROOT/'results/verification.json',dict(status='passed',n_checks=len(checks),checks=checks,
          primary_delta=probes['primary']['delta'],manifest_sha256=sha(ROOT/'results/final_manifest.json')))
    progress('verification_complete',checks=len(checks))

if __name__=='__main__': main()

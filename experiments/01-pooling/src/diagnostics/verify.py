"""Verify saved evidence, replay fitted probes and independently decode interventions.

Only study-local outputs are written. Existing manifests are checked before a
fresh manifest is produced, so a later invocation also detects artifact changes.
"""
from base import *
from probes import feature_blocks,labels,predict,ratio_ci
from conditional import blocks,joint
from runtime import Engine
import pickle,re,zipfile

checks=[]
def check(name,value):
    if not bool(value):raise AssertionError(name)
    checks.append(name)
def read(name):return json.loads((D/'results'/name).read_text())
def close(a,b,atol=1e-10):return np.allclose(a,b,atol=atol,rtol=1e-10)
def small(a,ix):return {k:a[k][ix] for k in a.files}

@torch.inference_mode()
def main():
    # Never rewrite an input to make a checksum pass.
    parent=json.loads((ROOT/'results/final_manifest.json').read_text())
    for path,digest in parent.items():check('parent_unchanged:'+path,sha(ROOT/path)==digest)
    frozen=read('protocol_frozen.json')
    for key,path in [('protocol_sha256',D/'protocol.json'),('objectives_sha256',D/'OBJECTIFS_ET_PROTOCOLE.md'),
                     ('previous_manifest_sha256',ROOT/'results/final_manifest.json'),('split_sha256',ROOT/'results/splits.json')]:
        check('frozen:'+key,sha(path)==frozen[key])
    previous=D/'results/final_manifest.json'
    if previous.exists():
        for path,digest in json.loads(previous.read_text()).items():check('study_unchanged:'+path,sha(D/path)==digest)
    sets=[set(SPLIT[n]) for n in ['train','validation','test']]
    check('split_sizes',[len(s) for s in sets]==[1000,400,600])
    check('split_disjoint',all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i)))
    check('split_complete',set.union(*sets)==set(range(2000)))
    parity=read('runtime_parity.json');check('runtime_parity_status',parity['status']=='passed')
    check('runtime_parity_all',all(all(s[k] for k in ['codes','logits','z8','z9']) for s in parity['samples']))
    te=np.array(SPLIT['test']);tr=np.array(SPLIT['train'])
    sources={}
    for corpus,oldname in [('official','hard'),('external','external')]:
        a=np.load(D/f'results/revision_{corpus}.npz');old=np.load(ROOT/f'results/{oldname}.npz');sources[corpus]=(a,old)
        for key,newkey in [('codes','codes'),('prefix','prefix'),('reference','reference')]:
            check(corpus+':'+key,np.array_equal(a[newkey],old[key]))
        for k,newkey in [(7,'decoded8'),(8,'decoded9'),(11,'final_ids')]:
            check(corpus+':'+newkey,np.array_equal(a[newkey],old['decoded'][:,k]))
        check(corpus+':probability_normalized',close(a['prob8'].sum(-1),1,1e-6))
        check(corpus+':probability_argmax',np.array_equal(a['prob8'].argmax(-1),a['decoded8'][:,CELL]))
    a,old=sources['official'];diag=read('level5_diagnostic.json');pool=read('pooling_algebra.json');drop=read('dropout_control.json')
    dcodes=old['decoded'][:,:,CELL];ref=old['reference'][:,CELL];before=dcodes[:,7]==ref;after=dcodes[:,8]==ref
    counts=read('revision_transition_counts.json')
    check('revision_counts:corrected',int((~before&after).sum())==counts['corrected'])
    check('revision_counts:degraded',int((before&~after).sum())==counts['degraded'])
    check('revision_counts:wrong_to_wrong',int((~before&~after&(dcodes[:,7]!=dcodes[:,8])).sum())==counts['wrong_to_wrong'])
    code_offset=sum([1,2,4,6]);dc=np.load(D/'results/dropout_control.npz')
    for d in ['hard','easy','medium']:
        for i,v in enumerate(diag[d]['levels']):check(f'{d}:distances:{i}',v['direct_float64_assignment_disagreements']==0)
        check(d+':pool_histogram',pool['checkpoint_variances'][d]['9']['token_histogram_inter_puzzle_variance']==0)
        check(d+':raw_invariance',pool['checkpoint_variances'][d]['9']['raw_embedding_inter_puzzle_variance']<1e-12)
        check(d+':dropout_frozen',drop[d]['state_dict_before']==drop[d]['state_dict_after'] and drop[d]['quantizer_remained_eval'])
        cached=np.load(ROOT/f'results/{d}.npz');ix=drop[d]['indices']
        check(d+':dropout_native_codes',np.array_equal(dc[d+'_native_codes'],cached['encoded'][ix,code_offset:code_offset+9]))
        for r in drop[d]['dropout_runs']:
            counts=np.bincount(dc[f"{d}_{r['seed']}_codes"].ravel(),minlength=16)
            check(f"{d}:dropout_counts:{r['seed']}",np.array_equal(counts,r['codes']['counts']))
    _,sol=boards('hard')
    for w in pool['windows']:
        counts=np.array([np.bincount(row[w['start_token']:w['stop_token_exclusive']],minlength=12) for row in sol[pool['indices']]])
        check('pool_window:'+str(w['slot']),np.all(counts==w['token_counts']))
    l5=read('level5_interventions.json');saved=np.load(D/'results/level5_interventions.npz')
    for corpus,source,ix in [('official_test','official',te),('external','external',np.arange(256))]:
        aa,oo=sources[source];native=aa['final_ids'][ix];ref=aa['reference'][ix];success=np.all(native==ref,axis=1)
        check(corpus+':l5_indices',np.array_equal(saved[corpus+'_indices'],ix))
        for name,r in l5[corpus].items():
            ids=saved[corpus+'_'+name+'_ids'];codes=saved[corpus+'_'+name+'_codes'];new=np.all(ids==ref,axis=1)
            check(corpus+':'+name+':counts',int(np.any(ids!=native,axis=1).sum())==r['changed_grids'] and int((codes!=aa['codes'][ix]).sum())==r['changed_codes'])
            check(corpus+':'+name+':success',int(new.sum())==r['metrics']['solved'])
            check(corpus+':'+name+':ci',close(paired_mean_ci(new.astype(float)-success)['ci95'],r['delta_success']['ci95']))
            if name.startswith('fixed'):
                check(corpus+':fixed5_exact',np.array_equal(ids,native) and np.array_equal(codes,aa['codes'][ix]))
                check(corpus+':fixed5_calls',r['generator_calls_per_batch']==[11])
    for conditional in [False,True]:
        modelpath='conditional_models.pkl' if conditional else 'probe_models.pkl';pack=pickle.loads((D/'results'/modelpath).read_bytes());projection=pack['metadata']['global_projection']
        function=blocks if conditional else feature_blocks
        for corpus,indices in [('official',te[:16]),('external',np.arange(16))]:
            aa,oo=sources[corpus];sm=small(aa,indices);g=oo['baseline'][indices,7];b,s,xy,blank=function(sm,g,projection)
            poisoned={k:v.copy() for k,v in sm.items()};poisoned['reference'][:]=0;poisoned['decoded9'][:]=0;poisoned['final_ids'][:]=0
            pb,ps,pxy,pm=function(poisoned,g,projection)
            check(f'{conditional}:{corpus}:no_future_features',np.array_equal(b,pb) and np.array_equal(s,ps) and np.array_equal(blank,pm))
            feats=dict(baseline=b,state=np.concatenate([s,xy],-1),combined=np.concatenate([b,s],-1),shuffled_states=np.concatenate([b,s],-1))
            tag='test' if corpus=='official' else 'external';fn=f'conditional_predictions_{tag}.npz' if conditional else f'probe_predictions_{tag}.npz';pred=np.load(D/'results'/fn)
            nxt,q,changed=labels(sm);oldids=sm['decoded8'][:,CELL]
            check(f'{conditional}:{corpus}:targets',np.array_equal(pred['target' if conditional else 'next_target'][:16],nxt))
            for name,x in feats.items():
                if conditional:
                    m=pack['models'][name];p=predict(m['alternative'],x,forbidden=oldids);pg=predict(m['gate'],x)
                    check(f'conditional:{corpus}:{name}:replay',close(p,pred[name+'_alternative'][:16]) and close(pg,pred[name+'_gate'][:16]))
                    check(f'conditional:{corpus}:{name}:joint',close(joint(pg,p,oldids),pred[name+'_joint'][:16]))
                    check(f'conditional:{corpus}:{name}:exclude_current',np.all(np.take_along_axis(p,oldids[...,None],-1)==0))
                else:
                    for task in ['next','quality']:
                        p=predict(pack['models'][task+'_'+name],x)
                        check(f'v1:{corpus}:{task}:{name}:replay',close(p,pred[task+'_'+name][:16]))
            if conditional:
                mm=pred['blank']&pred['changed'];yy=pred['target'];cp=pred['combined_alternative'].argmax(-1)==yy;bp=pred['baseline_alternative'].argmax(-1)==yy
                contrast=ratio_ci(((cp.astype(float)-bp)*mm).sum(1),mm.sum(1));report=read('conditional_probes.json' if corpus=='official' else 'conditional_external.json')
                c=report['contrasts']['combined_minus_fitted_baseline'];check(corpus+':conditional_bootstrap',close(contrast['delta'],c['delta']) and close(contrast['ci95'],c['ci95']))
    # Replay indirect readout for 16 sources, independently from the branch runner.
    model=SudokuModel(ROOT/'artifacts/sudoku/hard');sidx=te[:Q['causal_n']]
    zlevels,=model.vqvae.vq.idx_to_zhats(torch.tensor(a['codes'][sidx[:16]]),return_zlevels=False)
    z8=zlevels[7];engine=Engine(model,a['prefix'][sidx[:16]])
    for tag in ['steering','steering_conditional']:
        ar=np.load(D/f'results/{tag}.npz');report=read(tag+'.json');cell=ar['cell'];current=ar['current'];target=ar['target'];rr=np.arange(len(sidx))
        check(tag+':indices',np.array_equal(ar['indices'],sidx))
        given=a['prefix'][sidx,1:-1][:,CELL]
        check(tag+':selection',np.all(given[rr,cell]==0) and np.all(current!=target) and np.all((target>=1)&(target<=9)))
        check(tag+':current',np.array_equal(current,a['decoded8'][sidx][:,CELL][rr,cell]))
        check(tag+':direction_unit',close(np.linalg.norm(ar['direction'],axis=-1),1))
        check(tag+':random_unit',close(np.linalg.norm(ar['random_directions'],axis=-1),1))
        check(tag+':sham_output',np.array_equal(ar['sham_final_ids'],a['final_ids'][sidx]) and np.array_equal(ar['sham_decoded9'],a['decoded9'][sidx]))
        for name,r in report['conditions'].items():
            code=ar[name+'_codes9'];ids=ar[name+'_decoded9'];final=ar[name+'_final_ids'];new=np.all(final==ar['reference'],axis=1)
            check(tag+':'+name+':metrics',int(new.sum())==r['final_metrics']['solved'] and int((code!=ar['sham_codes9']).sum())==r['changed_level9_codes'])
            unchanged=np.all(code==ar['sham_codes9'],axis=1)
            check(tag+':'+name+':unchanged_codes_zero_indirect',close(ar[name+'_indirect_logodds'][unchanged],ar['sham_indirect_logodds'][unchanged],1e-6))
            if name!='sham':
                dose=float(name.split('_')[0]);check(tag+':'+name+':matched_norm',close(ar[name+'_delta_norm'],dose*ar[name+'_native_state_norm'],2e-6))
            if name in ['sham','0.25_positive','0.25_random0']:
                dl=model.vqvae.decoder(z8+engine.contribution(8,torch.tensor(code[:16])));p=dl.softmax(-1).numpy();pos=CELL[cell[:16]];rows=np.arange(16)
                value=np.log(np.maximum(p[rows,pos,target[:16]],1e-15))-np.log(np.maximum(p[rows,pos,current[:16]],1e-15))
                check(tag+':'+name+':independent_decode',close(value,ar[name+'_indirect_logodds'][:16],1e-4))
        randommean=np.mean([ar[f"0.25_random{i}_indirect_logodds"] for i in range(Q['random_directions'])],axis=0)
        c=paired_mean_ci(ar['0.25_positive_indirect_logodds']-randommean);actual=report['primary_indirect_positive_vs_random']
        check(tag+':primary_bootstrap',close(c['mean'],actual['mean']) and close(c['ci95'],actual['ci95']))
    # Check report assets and archive before publishing their hashes.
    report=(D/'RAPPORT.md').read_text()
    for target in re.findall(r'\]\(([^)]+)\)',report):
        if not target.startswith('http'):check('report_link:'+target,(D/target).exists())
    for path in (D/'figures').glob('*.png'):check('figure_nonempty:'+path.name,path.stat().st_size>10000)
    with zipfile.ZipFile(D/'sources_directions_mecanistes.zip') as archive:
        check('source_zip_integrity',archive.testzip() is None)
        for path in (D/'scripts').glob('*.py'):
            check('source_zip_current:'+path.name,archive.read(str(path.relative_to(ROOT)))==path.read_bytes())
    files=list((D/'scripts').glob('*.py'))+list(D.glob('*.md'))+[D/'protocol.json',D/'run.sh',D/'sources_directions_mecanistes.zip']
    files+=list((D/'figures').glob('*'))
    files += [p for p in (D/'results').iterdir() if p.is_file() and p.name not in ['verification.json','final_manifest.json','progress.json']]
    manifest={str(p.relative_to(D)):sha(p) for p in sorted(set(files))};write(D/'results/final_manifest.json',manifest)
    write(D/'results/verification.json',dict(status='passed',n_checks=len(checks),checks=checks,parent_manifest_files=len(parent),
        manifest_files=len(manifest),manifest_sha256=sha(D/'results/final_manifest.json'),
        replay_scope='Saved probes: first 16 official test and external puzzles; independent indirect decoder: first 16 causal puzzles, sham/positive/random0 in both series. All saved traces and paired primary bootstrap contrasts checked.',
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print(json.dumps(dict(status='passed',checks=len(checks),manifest_files=len(manifest))),flush=True)

if __name__=='__main__':main()

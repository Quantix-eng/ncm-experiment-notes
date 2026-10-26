"""Independent counts/checker, frozen-input audit, native and intervention replay."""
from base import *
from experiment import symkey
from solvers import backtracking,exact_cover_count
from runtime import Engine
from policy import decode_native,decode_certified
import zipfile
checks=[]
def check(name,value):
    assert bool(value),name
    checks.append(name)
def close(name,a,b,tol=1e-7):check(name,np.allclose(a,b,rtol=tol,atol=tol))
def read(n):return json.loads((D/'results'/n).read_text())
def inputs():
    f=read('protocol_frozen.json')
    for key,p in [('protocol_sha256',D/'protocol.json'),('objectives_sha256',D/'PROTOCOLE_GELE.md'),('tables_sha256',D/'results/discovery_tables.json'),('pilot_sha256',D/'results/discovery_pilot.json')]:check('frozen.'+key,sha(p)==f[key])
    prior=json.loads((D/'sources/previous_manifests.json').read_text())
    for name,h in prior.items():check('previous_manifest.'+name,sha(ROOT/name)==h)
    root=json.loads((ROOT/'results/final_manifest.json').read_text());check('parent_input_manifest',sha(ROOT/'results/input_manifest.json')==root['results/input_manifest.json']);root.update(json.loads((ROOT/'results/input_manifest.json').read_text()));used={}
    for task in ['hard','easy','medium']:
        for name in [f'results/{task}.npz',f'artifacts/sudoku/{task}/config.yaml',f'artifacts/sudoku/{task}/tokenizer.pt',f'artifacts/sudoku/{task}/generator.pt']:
            h=sha(ROOT/name);check('input.'+name,h==root[name]);used['../'+name]=h
    local_sources=[str(p.relative_to(ROOT)) for p in (ROOT/'vendor/ncm_mac').rglob('*.py') if '__pycache__' not in str(p)]
    raw_boards=[f'data/sudoku/{task}/{kind}.pickle' for task in ['hard','easy','medium'] for kind in ['puzzles','solutions']]
    for name in ['scripts/common.py','scripts/solvers.py','results/splits.json','protocol.json',*local_sources,*raw_boards]:
        h=sha(ROOT/name);check('input.'+name,h==root[name]);used['../'+name]=h
    previous_new=json.loads((ROOT/'compensation_conditionnelle/results/final_manifest.json').read_text());check('input.previous_fresh',sha(ROOT/'compensation_conditionnelle/data/fresh.json')==previous_new['data/fresh.json']);used['../compensation_conditionnelle/data/fresh.json']=sha(ROOT/'compensation_conditionnelle/data/fresh.json')
    # The older raw external JSON was not in the original manifest; cross-check its
    # exact contents against the saved, historically pinned trace instead.
    previous_inputs=json.loads((ROOT/'compensation_conditionnelle/results/input_manifest.json').read_text());check('input.previous_inputs_manifest',sha(ROOT/'compensation_conditionnelle/results/input_manifest.json')==previous_new['results/input_manifest.json']);check('input.old_external_trace',sha(ROOT/'results/external.npz')==previous_inputs['../results/external.npz'])
    ext=json.loads((ROOT/'data/external.json').read_text())['rows'];tr=np.load(ROOT/'results/external.npz');check('input.old_external_raw_solutions',np.array_equal(np.array([r['solution'] for r in ext]),tr['reference'][:,CELL]));check('input.old_external_raw_puzzles',np.array_equal(np.array([r['puzzle'] for r in ext]),tr['prefix'][:,1:-1][:,CELL]));used['../data/external.json']=sha(ROOT/'data/external.json');used['../results/external.npz']=sha(ROOT/'results/external.npz')
    for name in prior:
        folder=(ROOT/name).parents[1];manifest=json.loads((ROOT/name).read_text())
        for report in ['RAPPORT.md','ANALYSE_ET_PISTES.md']:
            if report in manifest:check('previous_report.'+str((folder/report).relative_to(ROOT)),sha(folder/report)==manifest[report])
    write(D/'results/input_manifest.json',used)
def fresh_and_checker():
    data=json.loads((D/'data/fresh.json').read_text());check('fresh.count',len(data['rows'])==Q['fresh_n']);check('fresh.protocol',data['protocol_sha256']==sha(D/'protocol.json'));excluded=set()
    for task in ['hard','easy','medium']:
        _,s=boards(task);excluded.update(symkey(row[CELL]) for row in s)
    for p in [ROOT/'data/external.json',ROOT/'compensation_conditionnelle/data/fresh.json']:excluded.update(symkey(r['solution']) for r in json.loads(p.read_text())['rows'])
    p=tokens(np.array([r['puzzle'] for r in data['rows']]),True);ref=tokens(np.array([r['solution'] for r in data['rows']]))
    for i,r in enumerate(data['rows']):
        key=symkey(r['solution']);check(f'fresh.{i}.family',key not in excluded);excluded.add(key);check(f'fresh.{i}.clues',sum(v!=0 for v in r['puzzle'])==30);check(f'fresh.{i}.MRV',backtracking(r['puzzle'],1)[0]==r['solution']);check(f'fresh.{i}.AlgorithmX',exact_cover_count(r['puzzle'],2)==1)
    check('checker.true_references',certificate(ref,p).all())
    bad=ref.copy();bad[:,CELL[0]]=bad[:,CELL[1]];check('checker.reject_duplicate',not certificate(bad,p).any())
    for v in [0,10,11]:
        bad=ref.copy();bad[:,CELL[0]]=v;check('checker.reject_digit_'+str(v),not certificate(bad,p).any())
    bad=ref.copy();bad[:,SEP[0]]=0;check('checker.reject_separator',not certificate(bad,p).any())
    bad=ref.copy()
    for i,row in enumerate(p):
        g=row[1:-1][CELL];d=int(g[g>0][0]);e=d%9+1;vals=bad[i,CELL].copy();bad[i,CELL]=np.where(vals==d,e,np.where(vals==e,d,vals))
    check('checker.reject_valid_wrong_clues',not certificate(bad,p).any())

def readout_scores():
    t=read('discovery_tables.json');tr=np.array(SPLIT['train'])
    for task in ['hard','easy','medium']:
        old=np.load(ROOT/f'results/{task}.npz');cnt=np.zeros((16,12),int);np.add.at(cnt,(old['codes'][tr,-89:][:,CELL].ravel(),old['decoded'][tr,-1][:,CELL].ravel()),1);mapping=cnt[:,1:10].argmax(1)+1;alphabet=np.array([cnt[:,j].argmax() for j in range(1,10)]);check(task+'.train_mapping',np.array_equal(mapping,t[task]['mapping']));check(task+'.train_alphabet',np.array_equal(alphabet,t[task]['alphabet_by_digit']))
        a=np.load(D/f'results/{task}_readout.npz');q=read(task+'_readout.json');candidate=lookup_board(a['codes'],mapping);check(task+'.candidate',np.array_equal(candidate,a['candidate']));accept=certificate(candidate,a['prefix']);check(task+'.acceptance',np.array_equal(accept,a['accept']));hybrid=np.where(accept[:,None],candidate,a['native']);check(task+'.policy',np.array_equal(hybrid,a['hybrid']))
        # Independent Python set-based Sudoku and clue validation.
        reference_accept=[]
        for pred,prefix in zip(candidate,a['prefix']):
            board=pred[CELL];giv=prefix[1:-1][CELL];grid=board.reshape(9,9);want=set(range(1,10))
            # Derive rows/columns/boxes anew rather than sharing the vectorized checker's unit indices.
            units=list(grid)+list(grid.T)+[grid[r:r+3,c:c+3].ravel() for r in [0,3,6] for c in [0,3,6]]
            reference_accept.append(all(set(map(int,u))==want for u in units) and all(g==0 or g==v for g,v in zip(giv,board)) and all(pred[j]==10 for j in SEP))
        check(task+'.independent_checker',np.array_equal(reference_accept,accept));check(task+'.certificates_correct',np.all(np.all(candidate==a['reference'],1)[accept]));check(task+'.no_introduced',not np.any(np.all(a['native']==a['reference'],1)&~np.all(hybrid==a['reference'],1)))
        for name,ids in [('native',a['native']),('hybrid',hybrid),('lookup_only',candidate)]:check(task+'.'+name+'.metrics',evaluate(ids,a['reference'],a['prefix'])==q[name])
        b=bootstrap(accept);close(task+'.coverage.mean',b['mean'],q['coverage']['mean']);close(task+'.coverage.ci',b['ci_familywise'],q['coverage']['ci_familywise'])

def causal_scores():
    a=np.load(D/'results/writing_interventions.npz');q=read('writing_interventions.json');N=len(a['native']);idx=np.arange(N);pos=a['cell'];old=a['native'][idx,CELL[pos]];eligible=np.arange(1,10)[None]!=old[:,None];check('writing.eligibility',np.array_equal(eligible,a['eligible']));check('writing.blank',np.all(a['prefix'][idx,1+CELL[pos]]==0));norm=a['delta_norm_position'];rn=a['random_norm_position'];close('writing.norm_match',norm,rn,1e-6)
    hits={}
    for name in ['patched','random']:
        pred=a[name];hit=pred[idx[:,None],np.arange(9)[None],CELL[pos][:,None]]==np.arange(1,10)[None];check('writing.'+name+'.hits',np.array_equal(hit,a[name+'_hits']));hits[name]=hit;diff=pred[:,:,CELL]!=a['native'][:,None,CELL];diff[idx[:,None],np.arange(9)[None],pos[:,None]]=False;check('writing.'+name+'.collateral',np.array_equal(diff.sum(2),a[name+'_collateral']))
    num=(hits['patched']&eligible).sum(1);rnum=(hits['random']&eligible).sum(1);den=eligible.sum(1)
    for key,v in [('target_writing',ratio_boot(num,den)),('random_target',ratio_boot(rnum,den)),('paired_advantage',bootstrap((num-rnum)/den))]:
        close('writing.'+key+'.mean',v['mean'],q[key]['mean']);close('writing.'+key+'.ci',v['ci_familywise'],q[key]['ci_familywise'])

@torch.inference_mode()
def numerical_support_audit(model,a,c):
    """Post-hoc numeric diagnostic; original interventions and primary counts stay fixed.

    Conv subtraction has tiny FP32 tails outside the mathematical three-position support.
    Evaluate the affine difference directly, cancelling bias analytically, and compare
    every original predicted token with both this difference and its matched noise.
    """
    phi=model.vqvae.vq.quant_resi[1.0];check('support.width3_affine',isinstance(phi,torch.nn.Conv1d) and phi.kernel_size==(3,))
    vq=model.vqvae.vq;mx_tail=0.;mx_ratio=0.;mx_diff=0.;changed={'patched':0,'random':0};N=len(c['native'])
    for lo in range(0,N,16):
        sl=slice(lo,min(lo+16,N));n=len(c['native'][sl]);codes=torch.tensor(a['codes'][sl,-89:]);z=torch.tensor(a['zfinal'][sl]);tp=CELL[c['cell'][sl]];emb=vq.embeddings[11](codes);support=(np.abs(np.arange(89)[None]-tp[:,None])<=1);mask=torch.tensor(support)
        with Engine(model,c['prefix'][sl]) as engine:
            base=engine.contribution(11,codes)
            for d in range(1,10):
                p=codes.clone();p[torch.arange(n),torch.tensor(tp)]=int(c['alphabet'][d-1]);de=(vq.embeddings[11](p)-emb).transpose(1,2);r=phi.resi_ratio
                # Direct FP64 contraction avoids the optimized FP32 convolution's tiny numerical tails.
                patches=torch.nn.functional.pad(de.double(),(1,1)).unfold(2,3,1)
                direct=torch.einsum('bcpt,oct->bop',patches,phi.weight.double())
                clean=(de.double()*(1-r)+direct*r).float().transpose(1,2).contiguous()
                actual=engine.contribution(11,p)-base;norm=actual.norm(dim=2);outside=norm.masked_fill(mask,0);ratio=outside.norm(dim=1)/norm.norm(dim=1).clamp_min(1e-30)
                mx_tail=max(mx_tail,float(outside.max()));mx_ratio=max(mx_ratio,float(ratio.max()));mx_diff=max(mx_diff,float((clean-actual).abs().max()))
                check(f'support.analytic_zero.{lo}.{d}',torch.all(clean[~mask]==0));close(f'support.recorded_norms.{lo}.{d}',norm.numpy(),c['delta_norm_position'][sl,d-1],1e-6)
                unit=np.stack([np.random.default_rng(Q['seed']+(lo+j)*100003+d*7919).normal(size=(89,256)) for j in range(n)]);unit/=np.linalg.norm(unit,axis=2)[:,:,None];noise=torch.tensor(unit,dtype=torch.float32)*clean.norm(dim=2)[:,:,None]
                for name,state in [('patched',z+clean),('random',z+noise)]:
                    ids=model.vqvae.decoder(state).argmax(-1).numpy();changed[name]+=int(np.any(ids!=c[name][sl,d-1],1).sum())
    write(D/'results/numerical_support_audit.json',dict(status='post_hoc_diagnostic',n_sources=N,requests_including_native_digit=N*9,maximum_tail_position_norm=mx_tail,maximum_tail_total_norm_ratio=mx_ratio,maximum_elementwise_delta_difference=mx_diff,readouts_changed_by_exact_support_computation=changed,explanation='Original matched controls include all actual FP32 delta norms, including tiny subtraction tails. Analytic affine difference cancels bias and has exact three-position support. Primary frozen results are not replaced.'))
    check('support.no_patched_readout_change',changed['patched']==0);check('support.no_control_readout_change',changed['random']==0)

@torch.inference_mode()
def replay():
    model=SudokuModel(ROOT/'artifacts/sudoku/hard');a=np.load(D/'results/hard_readout.npz');pred=model.predict(torch.tensor(a['prefix'][:16]));check('replay.native_codes',np.array_equal(pred.codes.numpy(),a['codes'][:16]));check('replay.native_state',np.array_equal(pred.reconstruction.numpy(),a['zfinal'][:16]));check('replay.batched',np.array_equal(pred.ids.numpy(),a['batched_native'][:16]));mapping=np.array(read('discovery_tables.json')['hard']['mapping']);calls=[];hook=model.vqvae.decoder.register_forward_pre_hook(lambda *_:calls.append(1))
    try:
        for i in range(16):
            before=len(calls);out=decode_certified(model,pred.reconstruction[i:i+1],pred.codes[i:i+1],a['prefix'][i:i+1],mapping);check(f'policy_replay.{i}.ids',np.array_equal(out,a['hybrid'][i:i+1]));check(f'policy_replay.{i}.calls',len(calls)-before==int(not a['accept'][i]))
    finally:hook.remove()
    c=np.load(D/'results/writing_interventions.npz');n=16;z=pred.reconstruction;codes=pred.codes[:,-89:];pos=c['cell'][:n];tp=CELL[pos]
    with Engine(model,a['prefix'][:n]) as e:
        r=e.contribution(11,codes)
        for d in range(1,10):
            p=codes.clone();p[torch.arange(n),torch.tensor(tp)]=int(c['alphabet'][d-1]);delta=e.contribution(11,p)-r;norm=delta.norm(dim=2);unit=np.stack([np.random.default_rng(Q['seed']+j*100003+d*7919).normal(size=(89,256)) for j in range(n)]);unit/=np.linalg.norm(unit,axis=2)[:,:,None];noise=torch.tensor(unit,dtype=torch.float32)*norm[:,:,None]
            for name,zz in [('patched',z+delta),('random',z+noise)]:check(f'writing_replay.{name}.{d}',np.array_equal(model.vqvae.decoder(zz).argmax(-1).numpy(),c[name][:n,d-1]))
    check('replay.cache_clean',all(not b.attn.caching and b.attn.cached_k is None for b in model.net.transformer.blocks))
    numerical_support_audit(model,a,c)

def timing():
    raw=read('benchmark_raw.json');q=read('benchmark.json');N=Q['benchmark_n'];check('timing.full_pairs',len(raw['full'])==N*Q['benchmark_repeats']);check('timing.stage_pairs',len(raw['stage'])==N);check('timing.one_thread',raw['threads']==1)
    for name,rows in [('end_to_end',raw['full']),('readout_only',raw['stage'])]:
        logs=np.array([np.mean([np.log(r['native_ns']/r['certified_ns']) for r in rows if r['puzzle']==i]) for i in range(N)]);b=bootstrap(logs);close('timing.'+name+'.mean',b['mean'],q[name]['log_speed_ratio']['mean']);close('timing.'+name+'.ci',b['ci95'],q[name]['log_speed_ratio']['ci95']);check('timing.'+name+'.ABBA',all(r['first_arm'] in [0,1] and r['native_ns']>0 and r['certified_ns']>0 for r in rows))
def publication():
    from editorial_audit import audit
    editorial=audit();check('editorial.unchanged_scientific_files',editorial['all_original_numerical_artifacts_and_frozen_rules_unchanged']);check('editorial.no_new_test_data',not editorial['new_experimental_data_generated'])
    p=np.load(D/'results/discovery_pilot.npz');old=np.load(ROOT/'results/hard.npz');ref=old['reference'][p['indices']];a=read('pilot_format_audit.json');check('pilot_format.all_tokens',a['exact_89_tokens']==int(np.all(p['terminal_only']==ref,1).sum()));check('pilot_format.cells',a['exact_81_cells']==int(np.all(p['terminal_only'][:,CELL]==ref[:,CELL],1).sum()));check('pilot_format.separators',a['exact_8_separators']==int(np.all(p['terminal_only'][:,SEP]==ref[:,SEP],1).sum()))
    decision=read('primary_decision.json');w=read('writing_interventions.json');h=read('hard_readout.json');check('primary.protocol',decision['protocol_sha256']==sha(D/'protocol.json'));check('primary.writing_decision',decision['writing']['criterion_satisfied']==(w['paired_advantage']['ci_familywise'][0]>0));check('primary.coverage_decision',decision['coverage']['criterion_satisfied']==(h['coverage']['ci_familywise'][0]>.30));check('primary.joint_decision',decision['both_co_primary_criteria_satisfied']==(decision['writing']['criterion_satisfied'] and decision['coverage']['criterion_satisfied']))
    with zipfile.ZipFile(D/'sources_recherche_positive.zip') as z:
        check('archive.crc',z.testzip() is None)
        for p in (D/'scripts').glob('*.py'):check('archive.current.'+p.name,z.read('scripts/'+p.name)==p.read_bytes())
    sys.path.insert(0,str(ROOT/'generalisation_architecturale/.deps'));import pymupdf
    doc=pymupdf.open(D/'PAPER.pdf');text='\n'.join(p.get_text() for p in doc);check('pdf.no_bad_glyph','\ufffd' not in text);check('pdf.pages',len(doc)>=4);check('pdf.title','Terminal Alphabet' in text)
def main():
    inputs();fresh_and_checker();readout_scores();causal_scores();replay();timing();publication();write(D/'results/verification.json',dict(status='passed',n_checks=len(checks),checks=checks,scope='Software/data assertions and same-agent verification, not independent scientific replication or proof against p-hacking. 4096 assertions check four properties for each of 1024 new puzzles.',replay='16 fresh puzzles: native codes/states/batched outputs, certified policy/call counts and all 9 writing targets with matched random perturbations. Primary statistics recomputed by a separate verification implementation; fresh puzzles rechecked with two solvers.',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    paths=[]
    for folder in ['scripts','sources','results','data','figures']:paths.extend(p for p in (D/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p))
    paths.extend(p for p in D.iterdir() if p.is_file());exclude={'results/final_manifest.json','results/progress.json'}
    write(D/'results/final_manifest.json',{str(p.relative_to(D)):sha(p) for p in sorted(set(paths)) if str(p.relative_to(D)) not in exclude and p.name!='.DS_Store'});log('positive_verified',checks=len(checks))
if __name__=='__main__':
    if '--numerical-only' in sys.argv:
        numerical_support_audit(SudokuModel(ROOT/'artifacts/sudoku/hard'),np.load(D/'results/hard_readout.npz'),np.load(D/'results/writing_interventions.npz'))
    else:main()

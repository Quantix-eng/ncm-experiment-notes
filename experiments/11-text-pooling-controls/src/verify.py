"""Verify final evidence with streaming hashes and a small native replay."""
from base import *
from text_audit import load_model,occupancy
from sudoku_levels import compare,generate
import gc,re,zipfile

checks=[]
def filehash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def check(name,value):
    if not bool(value):raise AssertionError(name)
    checks.append(name)
def read(name):return json.loads((D/'results'/name).read_text())
def close(a,b,tol=1e-10):return np.allclose(a,b,atol=tol,rtol=tol)

@torch.inference_mode()
def main(finalize=False):
    for root in [ROOT,ROOT/'directions_mecanistes']:
        manifest=json.loads((root/'results/final_manifest.json').read_text())
        for path,digest in manifest.items():check(str(root.name)+':unchanged:'+path,filehash(root/path)==digest)
    previous=read('protocol_frozen.json')
    for name,path in [('protocol_sha256',D/'protocol.json'),('objectives_sha256',D/'OBJECTIFS_ET_PROTOCOLE.md'),('previous_manifest_sha256',ROOT/'results/final_manifest.json'),('split_sha256',ROOT/'results/splits.json')]:
        check('frozen:'+name,filehash(path)==previous[name])
    current=D/'results/final_manifest.json'
    if current.exists() and not finalize:
        for path,digest in json.loads(current.read_text()).items():check('study_unchanged:'+path,filehash(D/path)==digest)
    for mname in ['assets_manifest.json','text_corpus_manifest.json']:
        for path,v in read(mname).items():
            h=filehash(D/path);check('asset:'+path,h==v['sha256'])
            if v['official_lfs_sha256']:check('official_lfs:'+path,h==v['official_lfs_sha256'])
    for task in ['tinystories','lm1b']:
        report=read(task+'_audit.json');data=read(task+'_data.json');a=np.load(D/f'results/{task}_audit.npz');inputs=np.load(D/f'data/{task}_selected.npz');x=inputs['tokens'];mask=inputs['mask'];offset=len(data['random_indices'])
        check(task+':state_frozen',report['state_before']==report['state_after'])
        check(task+':native_encode_parity',report['native_encode_parity'])
        check(task+':direct_distances',all(r['assignment_disagreements']==0 for r in report['distance_checks']))
        check(task+':random_indices',np.array_equal(inputs['source_indices'][:offset],data['random_indices']))
        check(task+':long_indices',np.array_equal(inputs['source_indices'][offset:],data['long_indices']))
        check(task+':no_random_resample',len(set(data['random_indices']))==offset)
        check(task+':long_condition',np.all(mask[offset:].sum(1)>=96))
        check(task+':mask',np.array_equal(mask,x!=data['pad_id']))
        for st,ix in [('random',np.arange(offset)),('long',np.arange(offset,len(x)))]:
            pred=a[st+'_decoded'];mm=mask[ix];accuracy=((pred==x[ix])&mm).sum()/mm.sum()
            check(task+':'+st+':reconstruction',close(accuracy,report[st]['reconstruction_nonpad_accuracy']))
            for r in report[st]['levels']:
                code=a[st+f"_codes_{r['level']}"];occ=occupancy(code,r['vocab']);saved=r['occupancy']
                for key in ['used','marginal_entropy','positional_entropy','modal_fraction','modal_position_fraction','unique_code_sequences','empirical_constant_level']:
                    check(f"{task}:{st}:{r['level']}:{key}",close(occ[key],saved[key]))
                check(f"{task}:{st}:{r['level']}:histogram",np.array_equal(occ['counts'],saved['counts']))
        control=read(task+'_controls.json');ar=np.load(D/f'results/{task}_controls.npz')
        for name in ['within_four','global_permutation']:
            native=ar['original_tokens'];variant=ar[name+'_tokens']
            check(task+':'+name+':global_histogram',np.array_equal(np.sort(native,axis=1),np.sort(variant,axis=1)))
            if name=='within_four':check(task+':block_histograms',np.array_equal(np.sort(native.reshape(-1,32,4),axis=-1),np.sort(variant.reshape(-1,32,4),axis=-1)))
            for key in ['raw_pool32','norm_pool32','first_input']:
                base=ar['original_'+key].astype(float);new=ar[name+'_'+key].astype(float);rel=np.linalg.norm(new-base)/max(np.linalg.norm(base),1e-30)
                check(task+':'+name+':'+key,close(rel,control[name][key+'_relative_l2']))
        model,_=load_model(task);lengths=list(model.vq.levels);ends=np.r_[0,np.cumsum(lengths)]
        for name in ['within_four','global_permutation']:
            for k,(s,e) in enumerate(zip(ends[:-1],ends[1:])):
                rate=(ar[name+'_codes'][:,s:e]!=ar['original_codes'][:,s:e]).mean()
                check(f'{task}:{name}:level{k+1}:disagreement',close(rate,control[name]['per_level_code_disagreement'][k]))
        # Replay first eight exact contexts without reproducing the whole audit.
        expected=np.concatenate([a[f'random_codes_{k+1}'][:8] for k in range(12)],axis=1)
        actual=torch.cat([model.encode(torch.tensor(row[None])) for row in x[:8]]).numpy()
        batch4=torch.cat([model.encode(torch.tensor(x[i:i+4])) for i in range(0,8,4)]).numpy()
        check(task+':native_replay_8',np.array_equal(actual,expected))
        check(task+':batch1_batch4_8',np.array_equal(actual,batch4))
        for i in range(2):
            ids=model.decode(torch.tensor(expected[i:i+1])).argmax(-1).numpy()[0]
            check(task+f':decode_replay:{i}',np.array_equal(ids,a['random_decoded'][i]))
        del model;gc.collect()
    audit=read('all_levels_audit.json');screen=read('sudoku_screen.json');selected=read('sudoku_selection_frozen.json');test=read('sudoku_test.json')
    va=np.array(SPLIT['validation'])[:Q['sudoku_screen_n']];te=np.array(SPLIT['test']);cache=np.load(ROOT/'results/hard.npz');screen_arrays=np.load(D/'results/sudoku_screen.npz');test_arrays=np.load(D/'results/sudoku_test.npz')
    check('screen:validation_only',np.array_equal(screen_arrays['indices'],va) and set(va).isdisjoint(te))
    rank=sorted(range(1,13),key=lambda k:(screen[f'fixed_{k}']['status_changes'],screen[f'fixed_{k}']['changed_grids'],k))
    check('selection:ranking',rank==selected['ranking'] and rank[:3]==selected['levels'])
    check('selection:screen_frozen',filehash(D/'results/sudoku_screen.json')==selected['screen_sha256'])
    templates=[];start=0
    for r in audit['hard']['generated_train']:
        c=cache['codes'][SPLIT['train'],start:start+r['length']];mode=np.stack([np.bincount(c[:,i],minlength=16) for i in range(r['length'])]).argmax(1)
        check('template:train_only:'+str(r['level']),np.array_equal(mode,r['template']));templates.append(mode);start+=r['length']
    def verify_metrics(tag,r,ids,codes,c,ix):
        result=compare(ids,codes,c['decoded'][ix,-1],c['codes'][ix],c['reference'][ix],c['prefix'][ix],r.get('generator_calls'))
        for key in ['changed_grids','changed_codes','status_changes','introduced','rescued']:
            check(tag+':'+key,result[key]==r[key])
        check(tag+':solved',result['metrics']['solved']==r['metrics']['solved'])
        check(tag+':paired_bootstrap',close(result['delta_success']['mean'],r['delta_success']['mean']) and close(result['delta_success']['ci95'],r['delta_success']['ci95']))
        check(tag+':wilson',close(result['status_change_wilson95'],r['status_change_wilson95']))
    for name,r in screen.items():verify_metrics('screen:'+name,r,screen_arrays[name+'_ids'],screen_arrays[name+'_codes'],cache,va)
    for corpus,rs in test.items():
        c=cache if corpus=='test' else np.load(ROOT/'results/external.npz');ix=te if corpus=='test' else np.arange(256)
        check(corpus+':indices',np.array_equal(test_arrays[corpus+'_indices'],ix))
        for name,r in rs.items():
            ids=test_arrays[corpus+'_'+name+'_ids'];codes=test_arrays[corpus+'_'+name+'_codes'];verify_metrics(corpus+':'+name,r,ids,codes,c,ix)
            check(corpus+':'+name+':calls',r['generator_calls']==12-len(r['fixed_levels']))
    # Independent rerun of the 9-prediction intervention on eight held-out puzzles.
    model=SudokuModel(ROOT/'artifacts/sudoku/hard');ids,codes,calls=generate(model,cache['prefix'][te[:8]],templates,fixed=[k-1 for k in selected['levels']])
    check('bypass:replay8:ids',np.array_equal(ids,test_arrays['test_joint_three_ids'][:8]))
    check('bypass:replay8:codes',np.array_equal(codes,test_arrays['test_joint_three_codes'][:8]) and calls==9)
    del model;gc.collect()
    sys.path.insert(0,str(D/'.deps'));import pymupdf
    pdf=pymupdf.open(D/'PAPER.pdf');check('pdf:pages',len(pdf)>=5)
    alltext='\n'.join(page.get_text() for page in pdf)
    check('pdf:core_sections',all(s in alltext for s in ['Abstract','conditional invariance','Cross-domain','Limitations','References']))
    check('pdf:no_replacement_glyph','\ufffd' not in alltext)
    check('pdf:page_text_present',all(len(page.get_text())>100 for page in pdf))
    write(D/'results/pdf_inspection.json',dict(pages=len(pdf),words=len(alltext.split()),renderer='PyMuPDF',page_sizes=[list(p.rect) for p in pdf]))
    for doc in ['PAPER.md','RAPPORT.md','README.md']:
        for target in re.findall(r'\]\(([^)]+)\)',(D/doc).read_text()):
            if not target.startswith('http'):check(doc+':link:'+target,(D/target).exists())
    with zipfile.ZipFile(D/'sources_generalisation.zip') as archive:
        check('archive:crc',archive.testzip() is None)
        for path in (D/'scripts').glob('*.py'):check('archive:current:'+path.name,archive.read(str(path.relative_to(ROOT)))==path.read_bytes())
    exclude={'results/verification.json','results/final_manifest.json','results/progress.json','results/pdf_inspection.json','logs/verify.log'}
    files=[p for sub in ['scripts','sources','results','figures','data','artifacts','logs'] for p in (D/sub).rglob('*') if p.is_file() and '__pycache__' not in str(p) and '.part' not in p.name and str(p.relative_to(D)) not in exclude]
    files+=list(D.glob('*.md'))+list(D.glob('*.tex'))+list(D.glob('*.pdf'))+[D/'run.sh',D/'protocol.json',D/'sources_generalisation.zip']
    manifest={str(p.relative_to(D)):filehash(p) for p in sorted(set(files))};write(D/'results/final_manifest.json',manifest)
    write(D/'results/verification.json',dict(status='passed',n_checks=len(checks),checks=checks,manifest_files=len(manifest),manifest_sha256=filehash(D/'results/final_manifest.json'),
      scope='All saved code counts, interventions and primary intervals; native text replay 8 contexts/task; batch1-vs4 8 contexts/task; independent Sudoku bypass replay 8 test puzzles. PDF structure inspected programmatically; rendered pages inspected separately.',
      utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print(json.dumps(dict(status='passed',checks=len(checks),manifest_files=len(manifest))),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--finalize',action='store_true',help='Create fresh hashes after an intentional experiment/report rebuild; parent/input checks still apply.')
    args=parser.parse_args();main(finalize=args.finalize)

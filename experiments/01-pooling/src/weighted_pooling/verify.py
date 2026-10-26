"""Recompute saved repair evidence and independently replay the actual patched path."""
from repair import *
import gc,re,zipfile
checks=[]
def read(name):return json.loads((D/'results'/name).read_text())
def check(name,value):
    if not bool(value):raise AssertionError(name)
    checks.append(name)
def close(a,b,tol=1e-10):return np.allclose(a,b,atol=tol,rtol=tol)
def out(ar,name):
    return {key:ar[name+'_'+key] for key in ['ids','codes','u5','native_u5','raw5','raw_native5']}

@torch.inference_mode()
def main(finalize=False):
    for root in [ROOT,ROOT/'directions_mecanistes',ROOT/'generalisation_architecturale']:
        for path,h in json.loads((root/'results/final_manifest.json').read_text()).items():check('previous:'+root.name+':'+path,filehash(root/path)==h)
    for path,h in json.loads((D/'sources/previous_manifests.json').read_text()).items():check('previous_manifest:'+path,filehash(ROOT/path)==h)
    frozen=read('protocol_frozen.json')
    for key,path in [('protocol_sha256',D/'protocol.json'),('objectives_sha256',D/'OBJECTIFS_ET_PROTOCOLE.md'),('split_sha256',ROOT/'results/splits.json')]:check('protocol:'+key,filehash(path)==frozen[key])
    manifest=D/'results/final_manifest.json'
    if manifest.exists() and not finalize:
        for path,h in json.loads(manifest.read_text()).items():check('unchanged:'+path,filehash(D/path)==h)
    screen=read('screen.json');selection=read('selection_frozen.json');selected=selection['selected'];held=read('heldout.json');parity=read('parity.json')
    val=np.array(SPLIT['validation'])[:Q['validation_n']];test=np.array(SPLIT['test'])
    check('split:validation_vs_test',set(val).isdisjoint(test));check('selection:indices',np.array_equal(val,selection['validation_indices']))
    check('selection:frozen_screen',filehash(D/'results/screen.json')==selection['screen_sha256'])
    eligible=[n for n,r in screen.items() if n not in ['native','weighted_0','fixed_bias_0.25'] and r['introduced']==0 and r['occupancy']['positional_entropy']>screen['native']['occupancy']['positional_entropy']+1e-12]
    rank=lambda n:(-screen[n]['occupancy']['positional_entropy'],screen[n]['input_change_rms'],n)
    expected=min(eligible,key=rank) if eligible else min([n for n in screen if n not in ['native','weighted_0','fixed_bias_0.25']],key=rank)
    check('selection:criterion',selected==expected and eligible==selection['eligible'])
    def verify_summary(tag,r,oo,native,reference):
        recomputed=summary(oo,reference,native)
        for k in ['n','exact','token_errors','cell_errors','introduced','rescued','changed_grids','changed_codes','changed_codes5','raw_variance','input_variance','input_change_rms']:
            check(tag+':'+k,close(recomputed[k],r[k]))
        for k in ['codes_used','marginal_entropy','positional_entropy','unique_sequences','counts']:
            check(tag+':occupancy:'+k,close(recomputed['occupancy'][k],r['occupancy'][k]))
        check(tag+':bootstrap',close(recomputed['delta_exact']['ci95'],r['delta_exact']['ci95']) and close(recomputed['delta_exact']['mean'],r['delta_exact']['mean']))
        check(tag+':wilson',close(recomputed['exact_wilson95'],r['exact_wilson95']))
        check(tag+':code_range',np.all((oo['codes']>=0)&(oo['codes']<16)))
        check(tag+':earlier_levels_unchanged',np.array_equal(oo['codes'][:,:13],native['codes'][:,:13]))
    ar=np.load(D/'results/screen.npz');check('screen:indices',np.array_equal(ar['indices'],val));native=out(ar,'native')
    for name,r in screen.items():verify_summary('screen:'+name,r,out(ar,name),native,ar['reference'])
    check('sham:all_codes',np.array_equal(ar['weighted_0_codes'],ar['native_codes']))
    check('sham:all_ids',np.array_equal(ar['weighted_0_ids'],ar['native_ids']))
    check('bias:variance_not_information',close(screen['fixed_bias_0.25']['input_variance'],screen['native']['input_variance'],1e-8))
    cache=np.load(ROOT/'results/hard.npz');check('native:previous_validation_codes',np.array_equal(native['codes'],cache['encoded'][val]))
    del ar,native;gc.collect()
    for corpus,rs in held.items():
        ar=np.load(D/f'results/{corpus}.npz');native=out(ar,'native')
        for name,r in rs.items():verify_summary(corpus+':'+name,r,out(ar,name),native,ar['reference'])
        if corpus!='external':
            c=np.load(ROOT/f"results/{corpus.split('_')[0]}.npz");check(corpus+':native_reference',np.array_equal(native['codes'],c['encoded'][test]) and np.array_equal(native['ids'],c['reconstructed'][test,-1]))
        del ar,native;gc.collect()
    noise=read('noise.json');ar=np.load(D/'results/noise.npz');clean_codes=np.load(D/'results/hard_test.npz');code_changes=read('noise_code_changes.json')
    check('noise:test_indices',np.array_equal(ar['indices'],test[:Q['robustness']['n_puzzles']]))
    for sigma,rs in noise.items():
        successes={}
        for name in ['native',selected]:
            ids=ar[sigma+'_'+name+'_ids'];successes[name]=np.all(ids==ar['reference'][None],axis=-1)
            check('noise:'+sigma+':'+name+':mean',close(successes[name].mean(),rs[name]['mean_exact']))
            check('noise:'+sigma+':'+name+':seedmean',close(successes[name].mean(0),rs[name]['per_puzzle_seedmean']))
            check('noise:'+sigma+':'+name+':changed_codes',int((ar[sigma+'_'+name+'_codes']!=clean_codes[name+'_codes'][None,:Q['robustness']['n_puzzles']]).sum())==code_changes[sigma][name])
        delta=paired_mean_ci(successes[selected].mean(0)-successes['native'].mean(0))
        check('noise:'+sigma+':paired_ci',close(delta['mean'],rs['paired_delta']['mean']) and close(delta['ci95'],rs['paired_delta']['ci95']))
    del ar;gc.collect()
    for kind,rs in read('symmetries.json').items():
        ar=np.load(D/f'results/symmetry_{kind}.npz');ref=ar['reference'];native=out(ar,'native')
        check('symmetry:'+kind+':valid_sudoku',all(np.all(np.sort(ref[:,CELL][:,u],axis=1)==np.arange(1,10)) for u in UNITS))
        for name,r in rs.items():verify_summary('symmetry:'+kind+':'+name,r,out(ar,name),native,ref)
        del ar,native;gc.collect()
    compensation=read('compensation.json');comp_ar=np.load(D/'results/compensation.npz');ref=comp_ar['reference']
    for name in ['native','reencoded','hybrid']:
        ids=comp_ar[name+'_ids'];check('compensation:'+name+':exact',int(np.all(ids==ref,axis=1).sum())==compensation[name]['exact'])
        check('compensation:'+name+':errors',int((ids!=ref).sum())==compensation[name]['token_errors'])
    check('compensation:fixed_other_codes',np.array_equal(comp_ar['native_codes'][:,:13],comp_ar['hybrid_codes'][:,:13]) and np.array_equal(comp_ar['native_codes'][:,22:],comp_ar['hybrid_codes'][:,22:]))
    check('compensation:common_level4',np.array_equal(comp_ar['native_decoded4'],comp_ar['reencoded_decoded4']))
    delta=(comp_ar['reencoded_decoded5'][:,CELL]==ref[:,CELL]).mean(1)-(comp_ar['native_decoded5'][:,CELL]==ref[:,CELL]).mean(1);ci_result=paired_mean_ci(delta)
    check('compensation:paired_delta',close(ci_result['mean'],compensation['paired_delta_level5_cell_accuracy']['mean']) and close(ci_result['ci95'],compensation['paired_delta_level5_cell_accuracy']['ci95']))
    check('compensation:amendment',filehash(D/'AMENDEMENT_COMPENSATION.md')==compensation['amendment_sha256'])
    model=SudokuModel(ROOT/'artifacts/sudoku/hard');before=statehash(model.vqvae);p,s=boards('hard');sample=test[:16]
    check('weighted:preserves_constants',close(pool(torch.ones(2,89,256),'weighted_1').numpy(),1,1e-7))
    ar=np.load(D/'results/hard_test.npz');counts=[]
    with patched(model.vqvae.vq,selected,counter=counts):
        encoded=model.vqvae.encode(torch.tensor(s[sample]));ids=model.vqvae.decode(encoded).argmax(-1).numpy()
    check('patched_native:replay16:codes',np.array_equal(encoded.numpy(),ar[selected+'_codes'][:16]))
    check('patched_native:replay16:ids',np.array_equal(ids,ar[selected+'_ids'][:16]))
    check('patched_native:method_called',len(counts)==1)
    # One original-size noisy batch reproduces the paired stress-test evidence.
    scale=read('train_scale.json');rms=scale['encoder_rms'];noise_ar=np.load(D/'results/noise.npz');seed=Q['robustness']['seeds'][0]
    for name in ['native',selected]:
        oo=run(model,s[sample],name,noise_sigma=.2*rms,noise_seed=seed,indices=sample)
        check('noise:replay16:'+name+':codes',np.array_equal(oo['codes'],noise_ar['0.2_'+name+'_codes'][0,:16]))
        check('noise:replay16:'+name+':ids',np.array_equal(oo['ids'],noise_ar['0.2_'+name+'_ids'][0,:16]))
        check('noise:perturbation_active:'+name,np.sqrt(np.mean((oo['u5']-ar[name+'_u5'][:16])**2))>1e-3)
    del oo,noise_ar;gc.collect()
    # Verify the analytic swap formula, including fixed positional embeddings.
    control=read('order_controls.json');order_ar=np.load(D/'results/order_controls.npz');ref=order_ar['reference'];raw=model.vqvae.encoder.wte(torch.tensor(ref)).double()+model.vqvae.encoder.wpe(torch.arange(89)).double()[None]
    swapped=order_ar['swapped'];raw2=model.vqvae.encoder.wte(torch.tensor(swapped)).double()+model.vqvae.encoder.wpe(torch.arange(89)).double()[None]
    t=torch.linspace(-1,1,10,dtype=torch.float64);expected=(t[0]-t[1])/10*(model.vqvae.encoder.wte(torch.tensor(ref[:,1])).double()-model.vqvae.encoder.wte(torch.tensor(ref[:,0])).double())
    delta=pool(raw2,'weighted_1')-pool(raw,'weighted_1')
    check('weighted:swap_formula',close(delta[:,0].numpy(),expected.numpy(),1e-12) and close(delta[:,1:].numpy(),0,1e-12))
    check('weighted:order_sensitivity',control['weighted_1']['swap_rms']>1e-3)
    check('bias:order_invariance',control['fixed_bias_0.25']['swap_rms']<1e-6)
    for name in Q['variants']:
        delta=order_ar[name+'_swapped'].astype(float)-order_ar[name+'_raw'];rms_delta=np.sqrt(np.mean(delta**2))
        check('order_control:'+name,close(rms_delta,control[name]['swap_rms']))
    reference_gen=model.predict(torch.tensor(p[test[:8]]));counts=[]
    with patched(model.vqvae.vq,selected,counter=counts):modified=model.predict(torch.tensor(p[test[:8]]))
    check('generator:zero_pool_calls',len(counts)==0 and parity['generation_pool_calls']==0)
    check('generator:codes',torch.equal(reference_gen.codes,modified.codes) and parity['generator_codes_identical'])
    check('generator:logits',torch.equal(reference_gen.logits,modified.logits) and parity['generator_logits_identical'])
    check('tokenizer:state_unchanged',before==statehash(model.vqvae)==parity['tokenizer_state_before']==parity['tokenizer_state_after'])
    align=read('generator_alignment.json');generated=cache['codes'][test]
    for name,r in align.items():
        code=ar[name+'_codes'];check('alignment:'+name,close((generated[:,13:22]==code[:,13:22]).mean(),r['level5_match']) and close((generated==code).mean(),r['all_codes_match']))
    del model;gc.collect()
    sys.path.insert(0,str(ROOT/'generalisation_architecturale/.deps'));import pymupdf
    doc=pymupdf.open(D/'PAPER.pdf');text='\n'.join(page.get_text() for page in doc)
    normalized=' '.join(text.split());check('pdf:repair_content','Frozen-generator' in normalized and 'without retraining' in normalized and len(doc)>=8)
    check('pdf:no_replacement_glyph','\ufffd' not in text)
    write(D/'results/pdf_inspection.json',dict(pages=len(doc),words=len(text.split()),renderer='PyMuPDF'))
    for name in ['README.md','PAPER.md','SUPPLEMENT.md','RAPPORT.md']:
        for target in re.findall(r'\]\(([^)]+)\)',(D/name).read_text()):
            if not target.startswith('http'):check('link:'+target,(D/target).exists())
    with zipfile.ZipFile(D/'sources_correction.zip') as archive:
        check('source_archive:crc',archive.testzip() is None)
        for path in (D/'scripts').glob('*.py'):check('source_archive:'+path.name,archive.read(str(path.relative_to(ROOT)))==path.read_bytes())
    exclude={'results/verification.json','results/final_manifest.json','results/progress.json','results/pdf_inspection.json','logs/verify.log'}
    files=[path for folder in ['scripts','sources','results','figures','logs'] for path in (D/folder).rglob('*') if path.is_file() and '__pycache__' not in str(path) and str(path.relative_to(D)) not in exclude]
    files+=list(D.glob('*.md'))+list(D.glob('*.pdf'))+list(D.glob('*.tex'))+[D/'run.sh',D/'protocol.json',D/'sources_correction.zip']
    write(manifest,{str(path.relative_to(D)):filehash(path) for path in sorted(set(files))})
    write(D/'results/verification.json',dict(status='passed',n_checks=len(checks),checks=checks,manifest_sha256=filehash(manifest),manifest_files=len(set(files)),
      replay='16 modified native encodes/decodes; 16 paired noisy encodes per operator at sigma .2; 8 native generator comparisons; all saved metrics/paired confidence intervals; previous campaign hashes.',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print(json.dumps(dict(status='passed',checks=len(checks),manifest_files=len(set(files)))),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--finalize',action='store_true');args=parser.parse_args();main(finalize=args.finalize)

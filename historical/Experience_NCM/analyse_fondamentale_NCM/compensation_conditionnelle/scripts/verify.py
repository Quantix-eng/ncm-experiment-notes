"""Saved-score recalculation, oracle checks, causal replay and source identity."""
from base import *
from assets import filehash
from crossed import bootstrap,symkey
from runtime import Engine
from solvers import exact_cover_count,backtracking
import zipfile,argparse,gc

checks=[]
def check(name,value):
    assert bool(value),name
    checks.append(name)
def close(name,a,b,tol=1e-6):check(name,np.allclose(a,b,atol=tol,rtol=tol))
def read(n):return json.loads((D/'results'/n).read_text())
def plain(v):return json.loads(json.dumps(v,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))

def inputs():
    frozen=read('protocol_frozen.json');check('protocol_identity',filehash(D/'protocol.json')==frozen['protocol_sha256']);check('objectives_identity',filehash(D/'OBJECTIFS_ET_PROTOCOLE.md')==frozen['objectives_sha256'])
    original=json.loads((ROOT/'results/final_manifest.json').read_text());check('parent_input_manifest',filehash(ROOT/'results/input_manifest.json')==original['results/input_manifest.json']);old_inputs=json.loads((ROOT/'results/input_manifest.json').read_text());identity={}
    prior=json.loads((ROOT/'generalisation_architecturale/results/final_manifest.json').read_text())
    for name in ['results/hard.npz','results/external.npz','results/splits.json','scripts/common.py','scripts/solvers.py']:
        h=filehash(ROOT/name);check('parent.'+name,h==original[name]);identity['../'+name]=h
    for name in ['artifacts/sudoku/hard/config.yaml','artifacts/sudoku/hard/tokenizer.pt','artifacts/sudoku/hard/generator.pt']:
        h=filehash(ROOT/name);check('weights.'+name,h==old_inputs[name]);identity['../'+name]=h
    g=ROOT/'generalisation_architecturale';check('text_metadata.original',filehash(g/'sources/model_metadata.json')==prior['sources/model_metadata.json']);check('text_metadata.copy',filehash(D/'sources/model_metadata.json')==filehash(g/'sources/model_metadata.json'));meta=json.loads((D/'sources/model_metadata.json').read_text());by={r['path']:r for r in meta['tree']};h=filehash(g/'artifacts/lm1b/tokenizer.pt');check('text_tokenizer.official',h==by['lm1b/tokenizer.pt']['lfs']['oid']);identity['../generalisation_architecturale/artifacts/lm1b/tokenizer.pt']=h
    check('text_generator.official',filehash(D/'artifacts/lm1b-small/generator.pt')==by['lm1b/small/generator.pt']['lfs']['oid']);identity['artifacts/lm1b-small/generator.pt']=filehash(D/'artifacts/lm1b-small/generator.pt')
    check('text_config.official_copy',filehash(D/'artifacts/lm1b-small/config.yaml')==filehash(g/'artifacts/lm1b/small/config.yaml'))
    for name in ['artifacts/lm1b/small/config.yaml','data/lm1b_selected.npz']:
        h=filehash(g/name);check('text_input.'+name,h==prior[name]);identity['../generalisation_architecturale/'+name]=h
    for folder in ['models','utils']:
        for p in sorted((D/'sources'/folder).rglob('*.py')):
            rel=str(p.relative_to(D));check('text_source.'+rel,filehash(p)==filehash(g/rel)==prior[rel])
    # All six previous finalized campaign manifests stay unchanged; full traces are audited where used.
    previous=json.loads((D/'sources/previous_manifests.json').read_text())
    for rel,h in previous.items():check('previous_manifest.'+rel,filehash(ROOT/rel)==h)
    write(D/'results/input_manifest.json',identity)

def fresh_checks():
    data=json.loads((D/'data/fresh.json').read_text());check('fresh.count',len(data['rows'])==Q['fresh_n']);check('fresh.protocol',data['protocol_sha256']==filehash(D/'protocol.json'))
    excluded=set()
    for task in ['hard','easy','medium']:
        _,s=boards(task);excluded.update(symkey(r[CELL]) for r in s)
    excluded.update(symkey(r['solution']) for r in json.loads((ROOT/'data/external.json').read_text())['rows'])
    for i,r in enumerate(data['rows']):
        p=r['puzzle'];s=r['solution'];key=symkey(s);check(f'fresh.{i}.clues',sum(v!=0 for v in p)==30);check(f'fresh.{i}.unique_algorithmX',exact_cover_count(p,2)==1);check(f'fresh.{i}.MRV',backtracking(p,1)[0]==s);check(f'fresh.{i}.family',key not in excluded);excluded.add(key)

def scores():
    for name in Q['sudoku_sets']:
        a=np.load(D/f'results/{name}_crossed.npz');q=read(name+'_crossed.json');mask=a['prefix'][:,1:-1][:,CELL]==0;ref=a['reference'];loss={}
        for c in Q['conditions']:
            l=a[c+'_logits'].astype(float);ids=l.argmax(-1);check(name+'.'+c+'.argmax',np.array_equal(ids,a[c+'_ids']));mx=l.max(-1);lse=mx+np.log(np.exp(l-mx[:,:,None]).sum(-1));gold=np.take_along_axis(l,ref[:,:,None],axis=-1)[:,:,0];nl=(lse-gold)[:,CELL];v=np.array([row[m].mean() for row,m in zip(nl,mask)]);loss[c]=v;close(name+'.'+c+'.NLL',v,a[c+'_loss']);check(name+'.'+c+'.metrics',evaluate(ids,ref,a['prefix'])==q[c]['metrics'])
        for key,v in [('R',loss['Y10']-loss['Y11']),('I',loss['Y10']+loss['Y01']-loss['Y11']-loss['Y00'])]:
            b=bootstrap(v);close(name+'.'+key+'.mean',b['mean'],q['contrasts'][key]['mean']);close(name+'.'+key+'.ci95',b['ci95'],q['contrasts'][key]['ci95']);close(name+'.'+key+'.ci975',b['ci975'],q['contrasts'][key]['ci975'])
        if name=='hard_test':
            parent=np.load(ROOT/'results/hard.npz');check('hard.native_codes',np.array_equal(a['native_codes'],parent['codes'][a['indices']]));check('hard.native_ids',np.array_equal(a['Y00_ids'],parent['decoded'][a['indices'],-1]));old=np.load(ROOT/'dynamique_temporelle/results/interventions.npz');check('old_omit9.ids',np.array_equal(a['Y11_ids'][:128],old['omit_9_ids']));check('old_omit9.codes',np.array_equal(a['adapted_codes'][:128],old['omit_9_codes']))

@torch.inference_mode()
def replay_sudoku():
    model=SudokuModel(ROOT/'artifacts/sudoku/hard');a=np.load(D/'results/fresh_crossed.npz')
    with Engine(model,a['prefix'][:16]) as e:
        z8,c8=e.first8();z9,c9,_,_=e.step(8,z8);nat=z9.clone();adapt=z8.clone();frozen=z8.clone();cross=z9.clone();cn=c8+[c9];ca=c8+[c9]
        for k in range(9,12):
            nat,c,_,_=e.step(k,nat);frozen=frozen+e.contribution(k,c);cn.append(c);adapt,c,_,_=e.step(k,adapt);cross=cross+e.contribution(k,c);ca.append(c)
        for name,z in [('Y00',nat),('Y10',frozen),('Y11',adapt),('Y01',cross)]:check('replay.'+name+'.logits',np.array_equal(model.vqvae.decoder(z).numpy(),a[name+'_logits'][:16]))
        check('replay.native_codes',np.array_equal(torch.cat(cn,1).numpy(),a['native_codes'][:16]));check('replay.adapted_codes',np.array_equal(torch.cat(ca,1).numpy(),a['adapted_codes'][:16]))

@torch.inference_mode()
def text():
    from text_generation import describe,load,sample,decode_history,official_readout
    a=np.load(D/'results/lm1b_generation.npz');q=read('lm1b_generation.json');ids=a['decoded'];final=ids[:,-1];mask=np.ones_like(final,bool);r=describe(ids,final,mask)
    check('text.fixed_count',len(ids)==Q['text_n']);check('text.fixed_seeds',np.array_equal(a['seeds'],np.arange(Q['text_seed_base'],Q['text_seed_base']+Q['text_n'])))
    structure=read('lm1b_checkpoint_structure.json');check('text.loaded_state',structure['all_state_tensors_equal_checkpoint']);check('text.shared_biases',structure['shared_head_views_bound_to_loaded_biases'])
    for i in range(Q['text_n']):
        row=np.load(D/f'data/text_trajectories/{i:04}.npz');check(f'text.valid_trajectory.{i}',np.array_equal(row['ids'],ids[i:i+1]) and np.array_equal(row['codes'],a['codes'][i:i+1]) and int(row['seed'])==int(a['seeds'][i]))
    for k in range(11):
        recorded=q['all_positions']['transitions'][k];actual=r['transitions'][k];check('text.transition.'+str(k),recorded==plain(actual))
    close('text.agreement',r['agreement_by_level'],q['all_positions']['agreement_by_level'])
    # A second counting procedure avoids relying only on the reporting function.
    adopt=(ids[:,7]!=final)&(ids[:,8]==final);abandon=adopt&((ids[:,9]!=final)|(ids[:,10]!=final));v=q['all_positions']['transitions'][7]['adoption_then_abandonment'];check('text.9.independent_counts',v['numerator']==int(abandon.sum()) and v['denominator']==int(adopt.sum()))
    oracle=np.load(D/'results/lm1b_oracle.npz');oq=read('lm1b_oracle.json');od=describe(oracle['decoded'],oracle['reference'],oracle['reference']!=0);check('oracle_text.counts',plain(od)==oq['reference_nonpad'])
    vq,net,cfg,inference=load();seed=int(a['seeds'][0]);codes=sample(net,inference['level0_unigram'],seed);check('text.replay.codes',np.array_equal(codes.numpy(),a['codes'][:1]));h,z=decode_history(vq,codes);check('text.replay.history',np.array_equal(h,ids[:1]));check('text.replay.official_final',np.array_equal(official_readout(vq.decoder(z),cfg,inference),a['official_final'][:1]));c=vq.encode(torch.tensor(oracle['reference'][:1]));h,_=decode_history(vq,c);check('text.oracle_replay',np.array_equal(h,oracle['decoded'][:1]))

def publication():
    with zipfile.ZipFile(D/'sources_compensation.zip') as z:
        check('archive.crc',z.testzip() is None)
        for p in (D/'scripts').glob('*.py'):check('archive.current.'+p.name,z.read('scripts/'+p.name)==p.read_bytes())
    sys.path.insert(0,str(ROOT/'generalisation_architecturale/.deps'));import pymupdf
    doc=pymupdf.open(D/'PAPER.pdf');txt='\n'.join(p.get_text() for p in doc);check('pdf.no_bad_glyph','\ufffd' not in txt);check('pdf.pages',len(doc)>=5);check('pdf.title','Conditional Residual Compensation' in txt)

def main():
    inputs();fresh_checks();scores();replay_sudoku()
    from text_generation import release_unused_memory
    release_unused_memory();text();release_unused_memory();publication();write(D/'results/verification.json',dict(status='passed',n_checks=len(checks),checks=checks,replay='16 fresh Sudoku puzzles, four conditions with exact logits/codes; first LM1B native generation and all readouts, official final readout and oracle first row. All saved primary metrics recalculated.',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    paths=[]
    for folder in ['scripts','sources','results','figures','data']:
        paths.extend(p for p in (D/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p))
    paths.extend(p for p in D.iterdir() if p.is_file());excluded={'results/final_manifest.json','results/progress.json'}
    write(D/'results/final_manifest.json',{str(p.relative_to(D)):filehash(p) for p in sorted(set(paths)) if str(p.relative_to(D)) not in excluded and p.name!='.DS_Store'});log('compensation_verified',checks=len(checks))
if __name__=='__main__':main()

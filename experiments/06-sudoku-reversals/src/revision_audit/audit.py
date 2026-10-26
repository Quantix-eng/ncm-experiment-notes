"""Exploratory mechanism audit: reuse verified traces, replay an oracle codec batch.

No generator interventions or circuit-discovery experiments are performed here.
"""
import os,sys,json,hashlib,time
from pathlib import Path
D=Path(__file__).resolve().parents[1];ROOT=D.parent
os.environ['MPLCONFIGDIR']=str(D/'.cache/matplotlib')
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
sys.path.insert(0,str(ROOT/'scripts'))
from common import np,torch,CELL,SEP,SudokuModel,write

SEED=202610241;B=2000
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()

def describe(ids,ref,prefix):
    v=ids[:,:,CELL];correct=v==ref[:,None,CELL];blank=prefix[:,1:-1][:,CELL]==0;out={}
    for name,mask in [('all_cells',np.ones_like(blank)),('initially_blank',blank)]:
        fixed=(~correct[:,7])&correct[:,8]&mask;rev=fixed&(~correct[:,9:]).any(1);num=rev.sum(1);den=fixed.sum(1);rng=np.random.default_rng(SEED);ix=rng.integers(0,len(ids),(B,len(ids)));ratios=num[ix].sum(1)/den[ix].sum(1)
        out[name]=dict(corrected9=int(den.sum()),reverted_later=int(num.sum()),rate=float(num.sum()/den.sum()),ci95=np.quantile(ratios,[.025,.975]),n_puzzles=len(ids))
    out['cell_accuracy_by_level']=correct.mean((0,2));out['exact_grids_by_level']=np.all(ids==ref[:,None],axis=-1).sum(0);out['exact_81cells_by_level']=correct.all(-1).sum(0)
    out['transitions']=[dict(level=k+1,corrected=int((~correct[:,k-1]&correct[:,k]).sum()),degraded=int((correct[:,k-1]&~correct[:,k]).sum())) for k in range(1,12)]
    ex=[]
    for i,c in np.argwhere((~correct[:,7])&correct[:,8]&(~correct[:,9:]).any(1))[:5]:ex.append(dict(local_puzzle_index=int(i),cell_index=int(c),reference=int(ref[i,CELL[c]]),values8_to12=v[i,7:,c]))
    out['examples']=ex
    return out

def main():
    manifest=json.loads((ROOT/'results/final_manifest.json').read_text());inputs={};checks=[]
    assert digest(ROOT/'results/input_manifest.json')==manifest['results/input_manifest.json']
    manifest.update(json.loads((ROOT/'results/input_manifest.json').read_text()))
    names=['results/'+d+'.npz' for d in ['hard','easy','medium']]+['results/splits.json','scripts/extract.py','scripts/common.py','vendor/ncm_mac/cached.py','vendor/ncm_mac/sudoku.py','vendor/ncm_mac/_vendor/models/quant.py','vendor/ncm_mac/_vendor/models/vae.py','vendor/ncm_mac/_vendor/models/vqvae.py','artifacts/sudoku/hard/config.yaml','artifacts/sudoku/hard/tokenizer.pt','artifacts/sudoku/hard/generator.pt','sources/paper.txt']
    for name in names:
        h=digest(ROOT/name);assert h==manifest[name],name;inputs[name]=h;checks.append('input_identity:'+name)
    split=json.loads((ROOT/'results/splits.json').read_text());result={}
    for difficulty in ['hard','easy','medium']:
        a=np.load(ROOT/f'results/{difficulty}.npz');ix=np.array(split['test']) if difficulty=='hard' else np.arange(len(a['reference']));ref=a['reference'][ix];prefix=a['prefix'][ix];q={}
        for kind,key in [('generator','decoded'),('oracle_codec','reconstructed')]:
            ids=a[key][ix];q[kind]=describe(ids,ref,prefix)
            # Recompute the primary count with a separate explicit puzzle/level loop.
            num=den=0
            for pred,truth in zip(ids[:,:,CELL],ref[:,CELL]):
                changed=(pred[7]!=truth)&(pred[8]==truth);den+=int(changed.sum());num+=sum(bool(changed[c]) and any(pred[k,c]!=truth[c] for k in [9,10,11]) for c in range(81))
            assert (num,den)==(q[kind]['all_cells']['reverted_later'],q[kind]['all_cells']['corrected9']);checks.append(difficulty+'.'+kind+'.independent_count')
        result[difficulty]=dict(n=len(ix),scope='fixed hard test' if difficulty=='hard' else 'all official puzzles; shared solution families',**q)
    a=np.load(ROOT/'results/hard.npz');ix=np.array(split['test'])[:16];model=SudokuModel(ROOT/'artifacts/sudoku/hard');calls=[];handles=[]
    for module in list(model.net.transformer.blocks)+list(model.net.lm_heads):handles.append(module.register_forward_pre_hook(lambda *_:calls.append(1)))
    with torch.inference_mode():
        codes=model.vqvae.encode(torch.tensor(a['reference'][ix]));assert np.array_equal(codes.numpy(),a['encoded'][ix]);checks.append('oracle_replay.codes_16')
        snapshots,=model.vqvae.vq.idx_to_zhats(codes,return_zlevels=False)
        for k,z in enumerate(snapshots):
            ids=model.vqvae.decoder(z).argmax(-1).numpy();assert np.array_equal(ids,a['reconstructed'][ix,k]);checks.append(f'oracle_replay.level_{k+1}_16')
    for handle in handles:handle.remove()
    assert not calls;checks.append('oracle_replay.no_generator_blocks_or_heads_called')
    cfg=model.config.tokenizer.vqvae;result['architecture_audit']=dict(decoder_layers=int(cfg.dec_layers),encoder_layers=int(cfg.get('enc_layers',0)),fine_drop=dict(cfg.vector_quantizer_config.aux.fine_drop),no_decoder_feedback='cached.generate_cached does not call decoder; sudoku.predict decodes after all codes are generated',inference_quantization='generator logits argmax; nearest-codebook search is in encoding, not in native generator heads',warning='Configured fine dropout and implemented training branch do not establish the exact historical training exposure.')
    result['status']='exploratory existing-trace audit plus oracle replay; proposals in ANALYSE_ET_PISTES.md are not executed'
    write(D/'results/audit.json',result);write(D/'results/input_manifest.json',dict(parent_manifest_sha256=digest(ROOT/'results/final_manifest.json'),files=inputs))
    write(D/'results/verification.json',dict(status='passed',checks=checks,n_checks=len(checks),replay='16 hard test solutions; exact encoder codes and 12 partial decoder outputs, with zero generator-block/head invocations',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print(json.dumps(dict(checks=len(checks),hard=result['hard']['oracle_codec']['all_cells']),default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v)))

if __name__=='__main__':main()

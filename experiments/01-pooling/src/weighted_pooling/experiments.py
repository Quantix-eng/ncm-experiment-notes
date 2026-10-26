"""Finite validation screen, frozen selection and held-out repair controls."""
from repair import *
import gc

def read(name):return json.loads((D/'results'/name).read_text())
def store_arrays(path,outputs,**meta):
    arrays=dict(meta)
    for name,out in outputs.items():
        for key,value in out.items():arrays[name+'_'+key]=value
    np.savez_compressed(D/'results'/path,**arrays)

@torch.inference_mode()
def main():
    if not (D/'results/protocol_frozen.json').exists():freeze()
    write(D/'sources/previous_manifests.json',{str(r.relative_to(ROOT)):filehash(r) for r in [ROOT/'results/final_manifest.json',ROOT/'directions_mecanistes/results/final_manifest.json',ROOT/'generalisation_architecturale/results/final_manifest.json']})
    p,s=boards('hard');train=np.array(SPLIT['train'])[:256];val=np.array(SPLIT['validation'])[:Q['validation_n']];test=np.array(SPLIT['test'])
    model=SudokuModel(ROOT/'artifacts/sudoku/hard');before=statehash(model.vqvae);z=model.vqvae._encode_pre_quant(torch.tensor(s[train]));rms=float(z.double().square().mean().sqrt());del z
    write(D/'results/train_scale.json',dict(indices=train,encoder_rms=rms))
    native=run(model,s[val],'native');check=model.vqvae.encode(torch.tensor(s[val[:16]]));assert np.array_equal(native['codes'][:16],check.numpy())
    assert np.array_equal(native['ids'][:16],model.vqvae.decode(check).argmax(-1).numpy())
    outputs={};screen={}
    for name in Q['variants']:
        out=native if name=='native' else run(model,s[val],name);outputs[name]=out;screen[name]=summary(out,s[val],native)
        if name=='weighted_0':assert np.array_equal(out['codes'],native['codes']) and np.array_equal(out['ids'],native['ids'])
        write(D/'results/screen.json',screen);log('screen',variant=name,exact=screen[name]['exact'],codes5=screen[name]['occupancy']['codes_used'],h5=screen[name]['occupancy']['positional_entropy'])
    store_arrays('screen.npz',outputs,indices=val,reference=s[val])
    del outputs,native,out;gc.collect()
    eligible=[name for name,v in screen.items() if name not in ['native','weighted_0','fixed_bias_0.25'] and v['introduced']==0 and v['occupancy']['positional_entropy']>screen['native']['occupancy']['positional_entropy']+1e-12]
    ranking=lambda name:(-screen[name]['occupancy']['positional_entropy'],screen[name]['input_change_rms'],name)
    selected=min(eligible,key=ranking) if eligible else min([n for n in screen if n not in ['native','weighted_0','fixed_bias_0.25']],key=ranking)
    selection=dict(selected=selected,eligible=eligible,passes_validation_preservation=bool(eligible),validation_indices=val,screen_sha256=filehash(D/'results/screen.json'),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write(D/'results/selection_frozen.json',selection);log('selected',variant=selected,passes=bool(eligible))
    variants=list(dict.fromkeys(['native',selected,'weighted_1','max_1']));results={};saved={}
    for corpus in ['hard_test','external','easy_test','medium_test']:
        if corpus=='external':_,ss=external_boards();ix=np.arange(len(ss))
        else:_,allsol=boards(corpus.split('_')[0]);ix=test;ss=allsol[ix]
        if corpus.startswith(('easy','medium')):
            del model;gc.collect();model=SudokuModel(ROOT/f"artifacts/sudoku/{corpus.split('_')[0]}")
        current={};oo={};nat=run(model,ss,'native')
        for name in (['native',selected] if corpus.startswith(('easy','medium')) else variants):
            out=nat if name=='native' else run(model,ss,name);oo[name]=out;current[name]=summary(out,ss,nat)
            write(D/'results/heldout.json',{**results,corpus:current});log('heldout',corpus=corpus,variant=name,exact=current[name]['exact'],codes5=current[name]['occupancy']['codes_used'])
        results[corpus]=current;store_arrays(corpus+'.npz',oo,indices=ix,reference=ss)
    del model;gc.collect();model=SudokuModel(ROOT/'artifacts/sudoku/hard')
    # Source-paired, valid solution transformations; no task solving claim.
    n=Q['symmetries_n'];ss=s[test[:n]];cells=ss[:,CELL];mapping=np.r_[0,np.random.default_rng(Q['seed']).permutation(np.arange(1,10))]
    transforms=dict(digits=tokens(mapping[cells]),transpose=tokens(cells.reshape(-1,9,9).transpose(0,2,1).reshape(-1,81)));sym={}
    for kind,reference in transforms.items():
        oo={};nat=run(model,reference,'native');sym[kind]={}
        for name in ['native',selected]:
            out=nat if name=='native' else run(model,reference,name);oo[name]=out;sym[kind][name]=summary(out,reference,nat)
        store_arrays('symmetry_'+kind+'.npz',oo,indices=test[:n],reference=reference)
    write(D/'results/symmetries.json',sym)
    # Same independent noise draws for every variant; aggregate seeds within puzzle.
    n=Q['robustness']['n_puzzles'];ss=s[test[:n]];noise={};arrays={}
    for sigma in Q['robustness']['sigma_relative_to_train_encoder_rms']:
        current={};pred={}
        for name in ['native',selected]:
            allids=[];allcodes=[]
            for seed in Q['robustness']['seeds']:
                out=run(model,ss,name,noise_sigma=sigma*rms,noise_seed=seed,indices=test[:n]);allids.append(out['ids']);allcodes.append(out['codes'])
            ids=np.stack(allids);cc=np.stack(allcodes);success=np.all(ids==ss[None],axis=-1);pred[name]=success
            current[name]=dict(mean_exact=float(success.mean()),per_puzzle_seedmean=success.mean(0),token_accuracy=float((ids==ss[None]).mean()))
            arrays[f'{sigma}_{name}_ids']=ids;arrays[f'{sigma}_{name}_codes']=cc
        current['paired_delta']=paired_mean_ci(pred[selected].mean(0)-pred['native'].mean(0));noise[str(sigma)]=current
        write(D/'results/noise.json',noise);log('noise',sigma=sigma,delta=current['paired_delta']['mean'])
    np.savez_compressed(D/'results/noise.npz',indices=test[:n],reference=ss,**arrays)
    # Native patched encode must equal the explicit codec, generation may never call it.
    sample=test[:Q['generator_parity_n']];native_generation=model.predict(torch.tensor(p[sample]));calls=[]
    with patched(model.vqvae.vq,selected,counter=calls):
        enc=model.vqvae.encode(torch.tensor(s[sample]));encode_calls=len(calls);calls.clear();generation=model.predict(torch.tensor(p[sample]));generation_calls=len(calls)
    explicit=run(model,s[sample],selected)
    parity=dict(patched_encode_codes=bool(np.array_equal(enc.numpy(),explicit['codes'])),encode_pool_calls=encode_calls,generation_pool_calls=generation_calls,
      generator_codes_identical=bool(torch.equal(native_generation.codes,generation.codes)),generator_logits_identical=bool(torch.equal(native_generation.logits,generation.logits)),
      generator_ids_identical=bool(torch.equal(native_generation.ids,generation.ids)),tokenizer_state_before=before,tokenizer_state_after=statehash(model.vqvae))
    assert parity['patched_encode_codes'] and parity['generator_codes_identical'] and parity['generator_logits_identical'] and before==parity['tokenizer_state_after']
    write(D/'results/parity.json',parity);np.savez_compressed(D/'results/generator_parity.npz',indices=sample,native_codes=native_generation.codes.numpy(),patched_codes=generation.codes.numpy(),native_ids=native_generation.ids.numpy(),patched_ids=generation.ids.numpy(),patched_encoded=enc.numpy())
    # Generator-code agreement with the old and new solution encodings is descriptive.
    held=np.load(D/'results/hard_test.npz');cached=np.load(ROOT/'results/hard.npz');gen=cached['codes'][test];alignment={}
    for name in variants:
        code=held[name+'_codes'];alignment[name]=dict(level5_match=float((gen[:,13:22]==code[:,13:22]).mean()),all_codes_match=float((gen==code).mean()))
    write(D/'results/generator_alignment.json',alignment)
    # Paired raw-embedding swap: algebra test without GroupNorm/residual history.
    raw=model.vqvae.encoder.wte(torch.tensor(s[val[:32]]));pos=model.vqvae.encoder.wpe(torch.arange(89))[None];raw=raw+pos
    swapped=s[val[:32]].copy();swapped[:,[0,1]]=swapped[:,[1,0]];raw2=model.vqvae.encoder.wte(torch.tensor(swapped))+pos;controls={};ar={}
    for name in Q['variants']:
        u=pool(raw,name);v=pool(raw2,name);delta=(v-u).numpy();controls[name]=dict(swap_relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(u.numpy())),swap_rms=float(np.sqrt(np.mean(delta.astype(float)**2))))
        ar[name+'_raw']=u.numpy();ar[name+'_swapped']=v.numpy()
    write(D/'results/order_controls.json',controls);np.savez_compressed(D/'results/order_controls.npz',reference=s[val[:32]],swapped=swapped,**ar)
    log('experiments_complete',selected=selected)

if __name__=='__main__':main()

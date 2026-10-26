"""Regimes O/C/E on all specified families; no post-outcome selection."""
import platform
import sys
import time
from support import D,ROOT,Q,np,torch,SudokuModel,data,write,emit,freeze,check_freeze,sha
from runtime import Engine,append,perturbations


@torch.inference_mode()
def main():
    if sys.argv[1:] == ['--freeze']:
        freeze()
        emit('freeze_complete')
        return
    assert not sys.argv[1:]
    check_freeze()
    assert not (D/'results/native.npz').exists(), 'Use a fresh sibling directory'
    start = time.monotonic()
    rows,prefix,reference,masks,different = data()
    model = SudokuModel(ROOT/'artifacts/sudoku/hard')
    arrays = {k:[] for k in ('codes','z','residuals','logits')}
    parity = []
    for f in range(len(rows)):
        grouped = {k:[] for k in arrays}
        for a in range(3):
            with Engine(model,prefix[f,a:a+1]) as engine:
                native = engine.native()
                resumed = engine.continuation(native['z'][:,Q['cut_level_one_based']-1])
                assert torch.equal(resumed['z'],native['z'][:,-1])
                assert torch.equal(resumed['logits'],native['logits'][:,-1])
                assert torch.equal(resumed['codes'],native['codes'][:,int(engine.offsets[Q['cut_level_one_based']]):])
            pred = model.predict(torch.tensor(prefix[f,a:a+1]))
            snapshots, = model.vqvae.vq.idx_to_zhats(pred.codes,return_zlevels=False)
            assert torch.equal(pred.codes,native['codes'])
            assert torch.equal(pred.reconstruction,native['z'][:,-1])
            assert torch.equal(pred.logits,native['logits'][:,-1])
            assert torch.equal(torch.stack(snapshots,1),native['z'])
            parity.append(dict(family=f,variant=a,status='bitwise_native_and_sham_passed'))
            for k in grouped:
                grouped[k].append(native[k].numpy()[0])
        for k in arrays:
            arrays[k].append(np.stack(grouped[k]))
        emit('native_parity',done=f+1,total=len(rows))
    native = {k:np.stack(v) for k,v in arrays.items()}
    native.update(prefix=prefix,reference=reference,mask=masks,different=different,
                  levels=np.array(model.net.levels),offsets=np.r_[0,np.cumsum(model.net.levels)])
    np.savez_compressed(D/'results/native.npz',**native)
    write(D/'results/native_parity.json',dict(status='passed',entries=parity))
    C = {k:[] for k in ('delta','positions','desired_norm','tail_residuals','tail_codes','logits')}
    E = {k:[] for k in ('logits','tail_residuals','tail_codes')}
    cut = Q['cut_level_one_based']-1
    for f in range(len(rows)):
        cf = {k:[] for k in C}
        for a in range(2):
            z = torch.tensor(native['z'][f,a:a+1,cut])
            nr = torch.tensor(native['residuals'][f,a:a+1,Q['cut_level_one_based']:])
            delta,positions,desired = perturbations(z,f,masks[f])
            tails = []
            with Engine(model,prefix[f,a:a+1]) as engine:
                for d in delta:
                    tails.append(engine.continuation(z+d.unsqueeze(0)))
            logits = []
            for j,d in enumerate(delta):
                perturbed = z+d.unsqueeze(0)
                adapted = tails[j]
                wrong = tails[Q['wrong_tail_indices'][j]]
                states = [append(z,nr),append(perturbed,nr),adapted['z'],
                          append(z,adapted['residuals']),append(perturbed,wrong['residuals'])]
                values = torch.stack([model.vqvae.decoder(s)[0] for s in states])
                assert torch.equal(values[0],torch.tensor(native['logits'][f,a,-1]))
                assert torch.equal(values[2],adapted['logits'][0])
                logits.append(values.numpy())
            cf['delta'].append(delta.numpy())
            cf['positions'].append(positions)
            cf['desired_norm'].append(desired)
            cf['tail_residuals'].append(torch.cat([t['residuals'] for t in tails],0).numpy())
            cf['tail_codes'].append(torch.cat([t['codes'] for t in tails],0).numpy())
            cf['logits'].append(np.stack(logits))
        for k in C:
            C[k].append(np.stack(cf[k]))
        ef = {k:[] for k in E}
        for right in (1,2):
            variants = (0,right)
            pair_logits,pair_r,pair_codes = [],[],[]
            for receiver in variants:
                input_logits,input_r,input_codes = [],[],[]
                with Engine(model,prefix[f,receiver:receiver+1]) as engine:
                    for donor in variants:
                        z = torch.tensor(native['z'][f,donor:donor+1,cut])
                        regenerated = engine.continuation(z)
                        if receiver == donor:
                            assert torch.equal(regenerated['logits'][0],torch.tensor(native['logits'][f,donor,-1]))
                        fixed = [torch.tensor(native['residuals'][f,t:t+1,Q['cut_level_one_based']:]) for t in variants]
                        input_logits.append(torch.stack([model.vqvae.decoder(append(z,r))[0] for r in fixed]+[regenerated['logits'][0]]).numpy())
                        input_r.append(regenerated['residuals'][0].numpy())
                        input_codes.append(regenerated['codes'][0].numpy())
                pair_logits.append(np.stack(input_logits))
                pair_r.append(np.stack(input_r))
                pair_codes.append(np.stack(input_codes))
            pair_logits = np.stack(pair_logits)
            assert np.array_equal(pair_logits[0,:,:2],pair_logits[1,:,:2])
            ef['logits'].append(pair_logits)
            ef['tail_residuals'].append(np.stack(pair_r))
            ef['tail_codes'].append(np.stack(pair_codes))
        for k in E:
            E[k].append(np.stack(ef[k]))
        emit('C_and_E',done=f+1,total=len(rows))
    np.savez_compressed(D/'results/crossed.npz',**{k:np.stack(v) for k,v in C.items()})
    np.savez_compressed(D/'results/exchanges.npz',**{k:np.stack(v) for k,v in E.items()})
    check_freeze()
    write(D/'results/execution.json',dict(status='executed_pending_independent_script_audit',
          utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),elapsed_seconds=time.monotonic()-start,
          families=len(rows),native_entries=3*len(rows),C_anchor_trials=8*len(rows),E_conditions=24*len(rows),
          environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,platform=platform.platform(),device='cpu',inference_dtype='float32',threads=torch.get_num_threads()),
          frozen_manifest_sha256=sha(D/'results/protocol_frozen.json')))
    emit('inference_complete')


if __name__ == '__main__':
    main()

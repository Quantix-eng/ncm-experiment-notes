"""Separate score calculations, residual reconstruction and explicit neural replay."""
import json
from support import D,ROOT,Q,np,torch,CELL,SEP,SudokuModel,data,write,check_freeze,sha
from solvers import backtracking,exact_cover_count
import torch.nn.functional as F


@torch.inference_mode()
def reference_tail(model,prefix,start):
    """Replay without importing the experimental Engine or its step method."""
    net,vq = model.net,model.vqvae.vq
    p = torch.tensor(prefix,dtype=torch.long)
    ncontext = p.shape[1]
    net._set_kv_caching(True)
    try:
        x = net.transformer.wte(model.vqvae.encoder.wte(p))
        for block in net.transformer.blocks:
            x = block(x,attn_mask=torch.zeros(1,1,ncontext,ncontext),rope_cos_sin=(net.rope_cos_gen_0,net.rope_sin_gen_0))
        kv = [(b.attn.cached_k,b.attn.cached_v) for b in net.transformer.blocks]
        offsets = np.r_[0,np.cumsum(net.levels)]
        pos = net.transformer.wle(net.lvl_1L)
        z = start.clone()
        codes,residuals = [],[]
        for k in range(Q['cut_level_one_based'],12):
            length = int(net.levels[k])
            q = F.interpolate(z.transpose(1,2),size=length,mode='area').transpose(1,2)
            q = net.transformer.wte(q)+pos[:,int(offsets[k]):int(offsets[k+1])]
            for block,(pk,pv) in zip(net.transformer.blocks,kv):
                block.attn.cached_k,block.attn.cached_v = pk,pv
                q = block(q,attn_mask=torch.zeros(1,1,length,ncontext+length),
                          rope_cos_sin=(getattr(net,f'rope_cos_gen_{k}'),getattr(net,f'rope_sin_gen_{k}')))
            c = net.lm_heads[k](net.transformer.ln_f(q)).argmax(-1)
            r = vq.embeddings[k](c)
            if k != 11:
                r = vq._up_to_full(r,k,89)
            r = vq._apply_quant_resi(r,k)
            z.add_(r)
            codes.append(c)
            residuals.append(r)
        return torch.cat(codes,1).numpy(),torch.stack(residuals,1).numpy(),model.vqvae.decoder(z).numpy()
    finally:
        net._set_kv_caching(False)


def independent_nll(logits,reference,mask):
    x = torch.tensor(logits,dtype=torch.float64).log_softmax(-1).numpy()
    losses = []
    for row in x.reshape(-1,89,12):
        losses.append(-np.mean([row[int(CELL[j]),int(reference[CELL[j]])] for j in np.flatnonzero(mask)]))
    return np.array(losses).reshape(x.shape[:-2])


@torch.inference_mode()
def main():
    check_freeze()
    rows,prefix,ref,mask,different = data()
    def load(name):
        with np.load(D/f'results/{name}.npz') as archive:
            return {k:archive[k] for k in archive.files}
    native,c,e,metrics = [load(k) for k in ('native','crossed','exchanges','metrics')]
    assert np.array_equal(native['prefix'],prefix) and np.array_equal(native['reference'],ref)
    assert np.array_equal(native['mask'],mask) and np.array_equal(native['different'],different)
    summary = json.loads((D/'results/summary.json').read_text())
    model = SudokuModel(ROOT/'artifacts/sudoku/hard')
    cut = Q['cut_level_one_based']-1
    counts = dict(families=0,decoder_reconstructions=0,neural_tail_replays=0,native_replays=0,score_comparisons=0)
    for f,row in enumerate(rows):
        assert len(backtracking(row['base'],3)) == 2 and exact_cover_count(row['base'],3) == 2
        for puzzle,solution in [(row['input_a'],row['oracle_solutions'][0]),(row['input_b'],row['oracle_solutions'][1]),(row['input_invariant'],row['oracle_solutions'][0])]:
            assert backtracking(puzzle,3) == [solution] and exact_cover_count(puzzle,3) == 1
        counts['families'] += 1
        for a in range(2):
            z0 = torch.tensor(native['z'][f,a:a+1,cut])
            norm = np.linalg.norm(c['delta'][f,a].astype(np.float64),axis=(-1,-2))
            assert np.allclose(norm,c['desired_norm'][f,a],rtol=2e-6,atol=0)
            assert np.array_equal(c['delta'][f,a,0],-c['delta'][f,a,1])
            assert np.array_equal(c['delta'][f,a,2],-c['delta'][f,a,3])
            support0 = np.any(c['delta'][f,a,0]!=0,axis=1)
            support1 = np.any(c['delta'][f,a,2]!=0,axis=1)
            assert support0.sum() == support1.sum() == 1 and not np.any(support0&support1)
            nr = native['residuals'][f,a,Q['cut_level_one_based']:]
            for j in range(4):
                zp = z0+torch.tensor(c['delta'][f,a,j:j+1])
                r = c['tail_residuals'][f,a,j]
                wrong = c['tail_residuals'][f,a,Q['wrong_tail_indices'][j]]
                starts = [z0,zp,zp,z0,zp]
                tails = [nr,nr,r,r,wrong]
                for condition,(start,tail) in enumerate(zip(starts,tails)):
                    state = start.clone()
                    for contribution in tail:
                        state.add_(torch.tensor(contribution[None]))
                    decoded = model.vqvae.decoder(state).numpy()[0]
                    assert np.array_equal(decoded,c['logits'][f,a,j,condition]),(f,a,j,condition)
                    counts['decoder_reconstructions'] += 1
                if f in Q['verification_replay_families']:
                    codes,residuals,logits = reference_tail(model,prefix[f,a:a+1],zp)
                    assert np.array_equal(codes[0],c['tail_codes'][f,a,j])
                    assert np.array_equal(residuals[0],r)
                    assert np.array_equal(logits[0],c['logits'][f,a,j,2])
                    counts['neural_tail_replays'] += 1
            independent = independent_nll(c['logits'][f,a],ref[f,a],mask[f])
            assert np.allclose(independent,metrics['losses'][f,a],rtol=0,atol=1e-12)
            counts['score_comparisons'] += independent.size
        for pair,right in enumerate((1,2)):
            variants = (0,right)
            assert np.array_equal(e['logits'][f,pair,0,:,:2],e['logits'][f,pair,1,:,:2])
            for i,receiver in enumerate(variants):
                for j,donor in enumerate(variants):
                    z = torch.tensor(native['z'][f,donor:donor+1,cut])
                    tails = [native['residuals'][f,t,Q['cut_level_one_based']:] for t in variants]+[e['tail_residuals'][f,pair,i,j]]
                    for kind,tail in enumerate(tails):
                        state = z.clone()
                        for r in tail:
                            state.add_(torch.tensor(r[None]))
                        assert np.array_equal(model.vqvae.decoder(state).numpy()[0],e['logits'][f,pair,i,j,kind])
                        counts['decoder_reconstructions'] += 1
                    if receiver == donor:
                        assert np.array_equal(e['logits'][f,pair,i,j,2],native['logits'][f,donor,-1])
                    if f in Q['verification_replay_families']:
                        codes,residuals,logits = reference_tail(model,prefix[f,receiver:receiver+1],z)
                        assert np.array_equal(codes[0],e['tail_codes'][f,pair,i,j])
                        assert np.array_equal(residuals[0],e['tail_residuals'][f,pair,i,j])
                        assert np.array_equal(logits[0],e['logits'][f,pair,i,j,2])
                        counts['neural_tail_replays'] += 1
        independent_h = independent_nll(e['logits'][f,0],ref[f,0],different[f])-independent_nll(e['logits'][f,0],ref[f,1],different[f])
        assert np.allclose(independent_h,metrics['h'][f],atol=1e-12,rtol=0)
        counts['score_comparisons'] += independent_h.size
        aa,ab,ba,bb = independent_h[0,0,2],independent_h[0,1,2],independent_h[1,0,2],independent_h[1,1,2]
        u = ((ba-aa)+(bb-ab))/2
        v = ((ab-aa)+(bb-ba))/2
        interaction = bb-ba-ab+aa
        span = bb-aa
        assert np.allclose([u,v,interaction,span],[metrics[k][f] for k in ('input_effect','state_effect','interaction','native_span')],atol=1e-12,rtol=0)
        both = all(np.array_equal(native['logits'][f,a,-1].argmax(-1),ref[f,a]) for a in (0,1))
        if not both:
            expected_label = 'non_identifiable_native_anchor_failure'
        elif span <= .1:
            expected_label = 'non_identifiable_weak_native_preference_span'
        elif abs(interaction/span) > .2:
            expected_label = 'non_additive_interaction_not_a_mechanism_class'
        elif u/span < 0 or u/span > 1:
            expected_label = 'non_monotone_non_identifiable'
        elif u/span >= .8:
            expected_label = 'input_dominant_response_preference_in_this_test'
        elif u/span <= .2:
            expected_label = 'state_dominant_response_preference_in_this_test'
        else:
            expected_label = 'two_sensitivities_mixed_or_unmodeled'
        assert expected_label == summary['E']['per_family'][f]['functional_label']
        for out,base,key in [(e['logits'][f,1,1,0,2],native['logits'][f,2,-1],'input_plus_state_A_minus_native_plus_nll'),
                             (e['logits'][f,1,0,1,2],native['logits'][f,0,-1],'input_A_state_plus_minus_native_A_nll')]:
            delta_nll = independent_nll(out,ref[f,0],mask[f])-independent_nll(base,ref[f,0],mask[f])
            assert np.isclose(delta_nll,metrics['invariant_'+key][f],atol=1e-12,rtol=0)
        if f in Q['verification_replay_families']:
            for a in range(3):
                pred = model.predict(torch.tensor(prefix[f,a:a+1]))
                assert np.array_equal(pred.codes.numpy()[0],native['codes'][f,a])
                assert np.array_equal(pred.logits.numpy()[0],native['logits'][f,a,-1])
                assert np.array_equal(pred.reconstruction.numpy()[0],native['z'][f,a,-1])
                counts['native_replays'] += 1
    # Independent recomposition of bootstrap and C decisions.
    L = metrics['losses']
    endpoints = dict(D=L[...,1]-L[...,0],R=L[...,1]-L[...,2],I=L[...,1]+L[...,3]-L[...,2]-L[...,0],S=L[...,4]-L[...,2])
    ix = np.random.default_rng(Q['seed']).integers(0,16,(Q['bootstrap'],16))
    tail_probability = .05/(2*4)
    adjusted = {}
    for key,values in endpoints.items():
        assert np.allclose(values,metrics[key],atol=1e-12,rtol=0)
        averaged = np.array([float(values[f].mean()) for f in range(16)])
        bootstrap = np.array([averaged[row].mean() for row in ix])
        ci = np.quantile(bootstrap,[tail_probability,1-tail_probability])
        adjusted[key] = ci
        assert np.allclose(ci,summary['C'][key]['ci_adjusted'],atol=1e-12,rtol=0)
        assert np.isclose(averaged.mean(),summary['C'][key]['mean'],atol=1e-12,rtol=0)
    m = Q['nll_effect_margin_nats']
    expected = ('useful_specific_functional_recovery_supported_for_tested_interventions' if all(ci[0]>m for ci in adjusted.values())
                else 'mean_frozen_damage_equivalent_within_prespecified_margin' if adjusted['D'][0]>-m and adjusted['D'][1]<m
                else 'useful_specific_recovery_not_confirmed_mechanism_non_identifiable')
    assert expected == summary['C']['decision']
    assert np.allclose(metrics['input_effect']+metrics['state_effect'],metrics['native_span'],atol=1e-12)
    # Check O counts without relying on the analysis script.
    for a,name in enumerate(Q['variants']):
        correct = native['logits'][:,a].argmax(-1) == ref[:,a,None,:]
        solved = [int(correct[:,k].all(1).sum()) for k in range(12)]
        assert solved == summary['O'][name]['solved_by_level']
        for transition in range(11):
            corrections,degradations = 0,0
            for f in range(16):
                for cell in np.flatnonzero(mask[f]):
                    before,after = correct[f,transition,CELL[cell]],correct[f,transition+1,CELL[cell]]
                    corrections += int(not before and after)
                    degradations += int(before and not after)
            assert corrections == summary['O'][name]['corrections_by_transition'][transition]
            assert degradations == summary['O'][name]['degradations_by_transition'][transition]
    check_freeze()
    write(D/'results/verification.json',dict(status='passed',counts=counts,
          summary_sha256=sha(D/'results/summary.json'),frozen_inputs_unchanged=True,
          scope='Separate reconstruction/scoring and independent explicit tail loop by same agent; not an independent scientific replication'))
    print(json.dumps(dict(status='passed',counts=counts)),flush=True)


if __name__ == '__main__':
    main()

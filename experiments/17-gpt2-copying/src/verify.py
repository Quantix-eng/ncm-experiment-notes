"""Independent scalar scoring, residual recomposition and explicit sublayer replay."""
from support import *
import math


def scalar_loss(values, target):
    maximum = float(max(values))
    normalizer = math.log(math.fsum(math.exp(float(x)-maximum) for x in values))
    return maximum+normalizer-float(values[target])


@torch.inference_mode()
def verify():
    check_freeze()
    setup()
    net = model()
    native = np.load(D/'results/native.npz')
    crossed = np.load(D/'results/crossed.npz')
    exchanges = np.load(D/'results/exchanges.npz')
    metrics = np.load(D/'results/metrics.npz')
    summary = read('results/summary.json')
    refs = native['references']
    stored_full = native['full']
    native_tails = native['tails']
    c_logits, c_tails, c_states, deltas = [crossed[k] for k in ('logits','tails','states','deltas')]
    e_logits, e_tails, e_states = [exchanges[k] for k in ('logits','tails','states')]
    counts = dict(decoder_recompositions=0, explicit_C_regenerations=0,
                  explicit_E_regenerations=0, official_native_replays=0, scalar_C_losses=0)

    def add(z, contributions):
        value = z.clone()
        for j in range(6):
            value = contributions[2*j]+value
            value = value+contributions[2*j+1]
        return value

    def decode(full, value):
        actual = full[-1].clone()
        actual[0,-1] = value
        return torch.nn.functional.linear(net.transformer.ln_f(actual),net.lm_head.weight)[0,-1]

    def explicit(full, z):
        value = z.clone()
        tail = []
        for j in range(6,12):
            block = net.transformer.h[j]
            source = full[j].clone()
            source[0,-1] = value
            attention = block.attn(block.ln_1(source),use_cache=False)[0]
            middle = attention+source
            mlp = block.mlp(block.ln_2(middle))
            outcome = middle+mlp
            assert torch.equal(outcome[0,:-1],full[j+1,0,:-1])
            tail.extend((attention[0,-1].clone(),mlp[0,-1].clone()))
            value = outcome[0,-1].clone()
        return torch.stack(tail),value

    scalar_losses = np.empty((16,2,4,5))
    for f in range(16):
        for a in range(2):
            full = torch.from_numpy(stored_full[f,a].copy())
            z = full[6,0,-1]
            base_tail = torch.from_numpy(native_tails[f,a].copy())
            for q in range(4):
                delta = torch.from_numpy(deltas[f,a,q].copy())
                assert math.isclose(float(delta.double().norm()),.1*float(z.double().norm()),rel_tol=2e-6)
                support = 0 if q<2 else 1
                assert torch.count_nonzero(delta[(1-support)*384:(2-support)*384]) == 0
                assert np.array_equal(deltas[f,a,q^1],-deltas[f,a,q])
                adapted = torch.from_numpy(c_tails[f,a,q].copy())
                other = torch.from_numpy(c_tails[f,a,q^2].copy())
                starts = (z,z+delta,z+delta,z,z+delta)
                tails = (base_tail,base_tail,adapted,adapted,other)
                for k in range(5):
                    final = add(starts[k],tails[k])
                    assert np.array_equal(final.numpy(),c_states[f,a,q,k])
                    assert np.array_equal(decode(full,final).numpy(),c_logits[f,a,q,k])
                    scalar_losses[f,a,q,k] = scalar_loss(c_logits[f,a,q,k],int(refs[f,a]))
                    counts['scalar_C_losses'] += 1
                    counts['decoder_recompositions'] += 1
                if f < 2:
                    replay_tail,replay_state = explicit(full,z+delta)
                    assert torch.equal(replay_tail,adapted)
                    assert np.array_equal(replay_state.numpy(),c_states[f,a,q,2])
                    counts['explicit_C_regenerations'] += 1
    assert np.allclose(scalar_losses,metrics['C_losses'],atol=1e-12,rtol=0)
    scalar_contrasts = []
    for f in range(16):
        values = [[],[],[],[]]
        for a in range(2):
            for q in range(4):
                l0,l1,l2,l3,l4 = scalar_losses[f,a,q]
                for dest,value in zip(values,(l1-l0,l1-l2,l1+l3-l2-l0,l4-l2)):
                    dest.append(value)
        scalar_contrasts.append([math.fsum(v)/8 for v in values])
    assert np.allclose(scalar_contrasts,metrics['C_family_contrasts'],atol=1e-12,rtol=0)
    # Reproduce grouped bootstrap independently, without the helper in support.py.
    for k,key in enumerate(('D','R','I','S')):
        rng = np.random.default_rng(20261025+k)
        ix = rng.integers(0,16,size=(20000,16))
        samples = np.asarray(scalar_contrasts)[:,k][ix].mean(axis=1)
        assert np.allclose(np.quantile(samples,[.00625,.99375]),summary['C'][key]['ci_adjusted'],atol=1e-12,rtol=0)
    for f in range(16):
        for comparison, branches in enumerate(((0,1),(0,2))):
            for r in range(2):
                full = torch.from_numpy(stored_full[f,branches[r]].copy())
                for d in range(2):
                    z = torch.from_numpy(stored_full[f,branches[d],6,0,-1].copy())
                    tails = (torch.from_numpy(native_tails[f,branches[0]].copy()),
                             torch.from_numpy(native_tails[f,branches[1]].copy()),
                             torch.from_numpy(e_tails[f,comparison,r,d].copy()))
                    for policy in range(3):
                        final = add(z,tails[policy])
                        assert np.array_equal(final.numpy(),e_states[f,comparison,r,d,policy])
                        assert np.array_equal(decode(full,final).numpy(),e_logits[f,comparison,r,d,policy])
                        counts['decoder_recompositions'] += 1
                    if r == d:
                        assert np.array_equal(e_logits[f,comparison,r,d,2],native['logits'][f,branches[r],-1])
                    if f < 2:
                        rt,rz = explicit(full,z)
                        assert torch.equal(rt,tails[2])
                        assert np.array_equal(rz.numpy(),e_states[f,comparison,r,d,2])
                        counts['explicit_E_regenerations'] += 1
            for donor in range(2):
                for policy in range(2):
                    assert np.array_equal(e_logits[f,comparison,0,donor,policy],e_logits[f,comparison,1,donor,policy])
        if f < 2:
            for a in range(3):
                out = net(torch.tensor([native['tokens'][f,a].tolist()]),use_cache=False).logits[0,-1]
                assert np.array_equal(out.numpy(),native['logits'][f,a,-1])
                counts['official_native_replays'] += 1

    h = np.asarray([e_logits[f,0,...,int(refs[f,1])].astype(np.float64)-
                    e_logits[f,0,...,int(refs[f,0])].astype(np.float64) for f in range(16)])
    assert np.allclose(h,metrics['E_preferences'],atol=1e-12,rtol=0)
    native_predictions = native['logits'].argmax(-1)
    recomputed_labels = {}
    for f,row in enumerate(summary['E']['per_family']):
        aa,ab,ba,bb = h[f,0,0,2],h[f,0,1,2],h[f,1,0,2],h[f,1,1,2]
        u=((ba-aa)+(bb-ab))/2
        v=((ab-aa)+(bb-ba))/2
        w=bb-ba-ab+aa
        g=bb-aa
        for key,value in (('input_effect',u),('state_effect',v),('interaction',w),('native_span',g)):
            assert abs(row[key]-value)<1e-12
        assert abs(u+v-g)<1e-12
        ratio = u/g if abs(g)>1e-12 else None
        if not all(native_predictions[f,:2,-1] == refs[f,:2]):
            label='non_identifiable_native_anchor_failure'
        elif g<=.1:
            label='non_identifiable_weak_native_preference_span'
        elif abs(w/g)>.2:
            label='non_additive_interaction_not_a_mechanism_class'
        elif ratio<0 or ratio>1:
            label='non_monotone_non_identifiable'
        elif ratio>=.8:
            label='input_dominant_response_preference_in_this_test'
        elif ratio<=.2:
            label='state_dominant_response_preference_in_this_test'
        else:
            label='two_sensitivities_mixed_or_unmodeled'
        assert row['functional_label']==label
        recomputed_labels[label]=recomputed_labels.get(label,0)+1
    assert recomputed_labels==summary['E']['label_counts']
    for a,name in enumerate(('A','B','Aplus')):
        assert int((native_predictions[:,a,-1] == refs[:,a]).sum()) == summary['O'][name]['final_exact']
    for k,name in enumerate(('Y00','Y10','Y11','Y01','Y1wrong')):
        assert int((c_logits[...,k,:].argmax(-1) == refs[:,:2,None]).sum()) == summary['C']['quality'][name]['exact']
    for comparison,branches in enumerate(((0,1),(0,2))):
        for mode,reported in summary['E']['quality'][('answer_change','answer_preserve')[comparison]].items():
            exact_recipient=exact_donor=other=changed=0
            for f in range(16):
                for r in range(2):
                    d=1-r
                    policy={'fixed_left':0,'fixed_right':1,'regenerated':2,'frozen_recipient_native':r,'frozen_donor_native':d}[mode]
                    prediction=int(np.argmax(e_logits[f,comparison,r,d,policy]))
                    exact_recipient+=int(prediction==refs[f,branches[r]])
                    exact_donor+=int(prediction==refs[f,branches[d]])
                    other+=int(prediction not in (refs[f,branches[r]],refs[f,branches[d]]))
                    changed+=int(prediction!=native_predictions[f,branches[r],-1])
            assert reported==dict(n=32,recipient_exact=exact_recipient,donor_exact=exact_donor,other=other,changed_from_native_recipient=changed)
    historical = {}
    for directory in ('benchmark_ncm_reel','article_benchmark_coordination'):
        base=ROOT/'analyse_fondamentale_NCM'/directory
        manifest=json.loads((base/'results/final_manifest.json').read_text())
        for path,expected in manifest['files'].items():
            assert sha(base/path)==expected,(directory,path)
        historical[directory]=len(manifest['files'])
    check_freeze()
    write('results/verification.json',dict(status='passed',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        counts=counts,frozen_inputs_unchanged=True,historical_manifests_unchanged=historical,
        scope='Internal audit by same research agent; no independent scientific replication',
        labels=recomputed_labels))
    print(json.dumps(read('results/verification.json'),indent=2))


if __name__ == '__main__':
    verify()

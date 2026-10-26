"""Prespecified family-level scores; no labels from architecture names."""
from support import *


def classify(exact, g, interaction, ratio):
    if not all(exact):
        return 'non_identifiable_native_anchor_failure'
    if g <= .1:
        return 'non_identifiable_weak_native_preference_span'
    if abs(interaction) > .2:
        return 'non_additive_interaction_not_a_mechanism_class'
    if not 0 <= ratio <= 1:
        return 'non_monotone_non_identifiable'
    if ratio >= .8:
        return 'input_dominant_response_preference_in_this_test'
    if ratio <= .2:
        return 'state_dominant_response_preference_in_this_test'
    return 'two_sensitivities_mixed_or_unmodeled'


def analyse():
    check_freeze()
    rows = read('data/families.json')['rows']
    n = np.load(D/'results/native.npz')
    c = np.load(D/'results/crossed.npz')
    e = np.load(D/'results/exchanges.npz')
    refs = n['references']
    native_ids = n['logits'].argmax(-1)
    O = {}
    for a, name in enumerate(('A','B','Aplus')):
        correct = native_ids[:,a] == refs[:,a,None]
        O[name] = dict(final_exact=int(correct[:,-1].sum()), n=16,
                       exact_by_readout=correct.sum(axis=0).tolist(),
                       corrections_by_transition=((~correct[:,:-1]) & correct[:,1:]).sum(axis=0).tolist(),
                       degradations_by_transition=(correct[:,:-1] & (~correct[:,1:])).sum(axis=0).tolist())
    cp = logprob(c['logits'])
    losses = -np.take_along_axis(cp,refs[:,:2,None,None,None],axis=-1)[...,0]
    ids = c['logits'].argmax(-1)
    c_quality = {name:dict(n=128,exact=int((ids[...,k] == refs[:,:2,None]).sum()))
                 for k,name in enumerate(('Y00','Y10','Y11','Y01','Y1wrong'))}
    per_family = np.stack((losses[...,1]-losses[...,0], losses[...,1]-losses[...,2],
                           losses[...,1]+losses[...,3]-losses[...,2]-losses[...,0],
                           losses[...,4]-losses[...,2]),axis=-1).mean(axis=(1,2))
    C = {key:bootstrap(per_family[:,k],20261025+k) for k,key in enumerate(('D','R','I','S'))}
    lower = [C[key]['ci_adjusted'][0] for key in ('D','R','I','S')]
    if all(bound > .01 for bound in lower):
        decision = 'useful_specific_recovery_supported_at_fixed_margin'
    elif all(abs(bound) < .01 for bound in C['D']['ci_adjusted']):
        decision = 'mean_frozen_damage_equivalent_within_prespecified_margin'
    else:
        decision = 'insufficient_or_not_confirmed'
    C.update(decision=decision,margin_nats=.01,quality=c_quality,
             loss_by_condition={key:float(losses[...,i].mean()) for i,key in enumerate(c_quality)})
    del cp
    ep = logprob(e['logits'])
    h = np.asarray([ep[f,0,...,int(refs[f,1])]-ep[f,0,...,int(refs[f,0])] for f in range(16)])
    reg = h[...,2]
    u = .5*(reg[:,1,0]-reg[:,0,0]+reg[:,1,1]-reg[:,0,1])
    v = .5*(reg[:,0,1]-reg[:,0,0]+reg[:,1,1]-reg[:,1,0])
    w = reg[:,1,1]-reg[:,1,0]-reg[:,0,1]+reg[:,0,0]
    g = reg[:,1,1]-reg[:,0,0]
    descriptions, counts = [], {}
    for f,row in enumerate(rows):
        exact = (native_ids[f,:2,-1] == refs[f,:2]).tolist()
        ratio = float(u[f]/g[f]) if abs(g[f]) > 1e-12 else None
        interaction = float(w[f]/g[f]) if abs(g[f]) > 1e-12 else None
        label = classify(exact,g[f],interaction,ratio)
        counts[label] = counts.get(label,0)+1
        descriptions.append(dict(source_id=row['source_id'],native_exact_A=exact[0],native_exact_B=exact[1],
            input_effect=float(u[f]),state_effect=float(v[f]),interaction=float(w[f]),native_span=float(g[f]),
            input_ratio=ratio,interaction_ratio=interaction,functional_label=label,mechanistic_conclusion='non_identified'))
    e_ids = e['logits'].argmax(-1)
    quality = {}
    for comparison, branches in enumerate(((0,1),(0,2))):
        modes = {}
        for mode in ('fixed_left','fixed_right','regenerated','frozen_recipient_native','frozen_donor_native'):
            recipient_correct = donor_correct = other = changed = 0
            for f in range(16):
                for recipient in range(2):
                    donor = 1-recipient
                    policy = {'fixed_left':0,'fixed_right':1,'regenerated':2,
                              'frozen_recipient_native':recipient,'frozen_donor_native':donor}[mode]
                    prediction = int(e_ids[f,comparison,recipient,donor,policy])
                    target_recipient, target_donor = map(int,(refs[f,branches[recipient]],refs[f,branches[donor]]))
                    recipient_correct += int(prediction == target_recipient)
                    donor_correct += int(prediction == target_donor)
                    other += int(prediction not in (target_recipient,target_donor))
                    changed += int(prediction != native_ids[f,branches[recipient],-1])
            modes[mode] = dict(n=32,recipient_exact=recipient_correct,donor_exact=donor_correct,
                               other=other,changed_from_native_recipient=changed)
        quality[('answer_change','answer_preserve')[comparison]] = modes
    invariant = []
    for f in range(16):
        for recipient in range(2):
            baseline = ep[f,1,recipient,recipient,2]
            swapped = ep[f,1,recipient,1-recipient,2]
            invariant.append(dict(source_id=rows[f]['source_id'],recipient=recipient,
                nll_difference=float(baseline[int(refs[f,0])]-swapped[int(refs[f,0])]),
                probability_TV=float(.5*np.abs(np.exp(baseline)-np.exp(swapped)).sum()),
                argmax_changed=bool(e_ids[f,1,recipient,recipient,2] != e_ids[f,1,recipient,1-recipient,2])))
    E = dict(input_effect=bootstrap(u,20261125), state_effect=bootstrap(v,20261126),
             interaction=bootstrap(w,20261127),native_span=bootstrap(g,20261128),label_counts=counts,
             per_family=descriptions,quality=quality,answer_preserving=invariant)
    np.savez_compressed(D/'results/metrics.npz',C_losses=losses,C_family_contrasts=per_family,
                        E_preferences=h,u=u,v=v,w=w,g=g)
    write('results/summary.json',dict(status='Executed second learned architecture, no mechanistic ground truth',
          families=16,O=O,C=C,E=E,scope=read('protocol.json')['scope']))
    check_freeze()
    print(json.dumps(dict(O=O,C=C,E_labels=counts,E_quality=quality),indent=2))


if __name__ == '__main__':
    analyse()

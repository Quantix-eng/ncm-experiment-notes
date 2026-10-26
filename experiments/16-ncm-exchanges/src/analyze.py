"""Fixed estimands and decisions; no fitting or model calls."""
import json
from support import D,Q,np,data,write,nll,logprob,preference,evaluate,CELL,check_freeze


def interval(values,ix):
    values = np.asarray(values,dtype=np.float64)
    means = values[ix].mean(1)
    alpha = Q['C_familywise_alpha']/len(Q['C_joint_endpoints'])
    return dict(mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]),
                ci_adjusted=np.quantile(means,[alpha/2,1-alpha/2]),n_families=len(values))


def main():
    check_freeze()
    rows,prefix,ref,mask,different = data()
    n = len(rows)
    with np.load(D/'results/native.npz') as archive:
        native = {k:archive[k] for k in archive.files}
    with np.load(D/'results/crossed.npz') as archive:
        c = {k:archive[k] for k in archive.files}
    with np.load(D/'results/exchanges.npz') as archive:
        e = {k:archive[k] for k in archive.files}
    ids = native['logits'].argmax(-1)
    ix = np.random.default_rng(Q['seed']).integers(0,n,(Q['bootstrap'],n))
    O = {}
    for a,name in enumerate(Q['variants']):
        # Explicit per-source masks avoid advanced-index axis reordering.
        corrections = np.zeros((n,11),int)
        degradations = np.zeros((n,11),int)
        reversions = np.zeros((n,11),int)
        for f in range(n):
            good = ids[f,a][:,CELL] == ref[f,a,CELL]
            for k in range(11):
                new = ~good[k] & good[k+1] & mask[f]
                corrections[f,k] = new.sum()
                degradations[f,k] = (good[k] & ~good[k+1] & mask[f]).sum()
                reversions[f,k] = (new & np.any(~good[k+2:],axis=0)).sum() if k < 10 else 0
        O[name] = dict(final=evaluate(ids[:,a,-1],ref[:,a],prefix[:,a]),
                      solved_by_level=[int(np.all(ids[:,a,k]==ref[:,a],axis=1).sum()) for k in range(12)],
                      corrections_by_transition=corrections.sum(0),degradations_by_transition=degradations.sum(0),
                      later_reversions_of_corrections_by_transition=reversions.sum(0),
                      mechanistic_conclusion='non_identifiable_from_trajectories')
    losses = np.stack([[nll(c['logits'][f,a],ref[f,a],mask[f]) for a in range(2)] for f in range(n)])
    L00,L10,L11,L01,Lwrong = np.moveaxis(losses,-1,0)
    contrasts = dict(D=L10-L00,R=L10-L11,I=L10+L01-L11-L00,S=Lwrong-L11)
    C = {key:interval(v.mean((1,2)),ix) for key,v in contrasts.items()}
    margin = Q['nll_effect_margin_nats']
    if all(C[key]['ci_adjusted'][0] > margin for key in Q['C_joint_endpoints']):
        cdecision = 'useful_specific_functional_recovery_supported_for_tested_interventions'
    elif C['D']['ci_adjusted'][0] > -margin and C['D']['ci_adjusted'][1] < margin:
        cdecision = 'mean_frozen_damage_equivalent_within_prespecified_margin'
    else:
        cdecision = 'useful_specific_recovery_not_confirmed_mechanism_non_identifiable'
    C.update(decision=cdecision,margin_nats=margin,
             loss_by_condition=dict(zip(('Y00','Y10','Y11','Y01','Y1wrong'),losses.mean((0,1,2)))),
             by_sign={sign:{key:interval(v[:,:,indices].mean((1,2)),ix) for key,v in contrasts.items()}
                      for sign,indices in [('positive',[0,2]),('negative',[1,3])]},
             tail_code_changed_fraction=float((c['tail_codes']!=native['codes'][:,:2,None,int(native['offsets'][Q['cut_level_one_based']]):]).mean()),
             mechanistic_conclusion='No ground-truth mechanism class assigned')
    h = np.stack([preference(e['logits'][f,0],ref[f,0],ref[f,1],different[f]) for f in range(n)])
    reg = h[:,:,:,2]
    u = .5*((reg[:,1,0]-reg[:,0,0])+(reg[:,1,1]-reg[:,0,1]))
    v = .5*((reg[:,0,1]-reg[:,0,0])+(reg[:,1,1]-reg[:,1,0]))
    w = reg[:,1,1]-reg[:,1,0]-reg[:,0,1]+reg[:,0,0]
    g = reg[:,1,1]-reg[:,0,0]
    descriptions = []
    counts = {}
    for f,row in enumerate(rows):
        exact = [bool(np.all(ids[f,a,-1]==ref[f,a])) for a in range(2)]
        ratio = float(u[f]/g[f]) if abs(g[f])>1e-12 else None
        interaction = float(w[f]/g[f]) if abs(g[f])>1e-12 else None
        if not all(exact):
            label = 'non_identifiable_native_anchor_failure'
        elif g[f] <= Q['E_min_native_preference_span_nats']:
            label = 'non_identifiable_weak_native_preference_span'
        elif abs(interaction)>Q['E_max_additivity_residual_ratio']:
            label = 'non_additive_interaction_not_a_mechanism_class'
        elif ratio < 0 or ratio > 1:
            label = 'non_monotone_non_identifiable'
        elif ratio >= Q['E_input_dominance_ratio']:
            label = 'input_dominant_response_preference_in_this_test'
        elif ratio <= Q['E_state_dominance_ratio']:
            label = 'state_dominant_response_preference_in_this_test'
        else:
            label = 'two_sensitivities_mixed_or_unmodeled'
        counts[label] = counts.get(label,0)+1
        descriptions.append(dict(source_id=row['source_id'],native_exact_A=exact[0],native_exact_B=exact[1],
                                 native_span=float(g[f]),input_effect=float(u[f]),state_effect=float(v[f]),
                                 interaction=float(w[f]),input_ratio=ratio,interaction_ratio=interaction,
                                 functional_label=label,mechanistic_conclusion='non_identified'))
    invariant = dict(native_plus_minus_A_nll=[],input_plus_state_A_minus_native_plus_nll=[],
                     input_A_state_plus_minus_native_A_nll=[],input_plus_state_A_probability_TV=[],
                     input_A_state_plus_probability_TV=[])
    for f in range(n):
        nativea = native['logits'][f,0,-1]
        nativeplus = native['logits'][f,2,-1]
        pa = e['logits'][f,1,1,0,2]
        ap = e['logits'][f,1,0,1,2]
        invariant['native_plus_minus_A_nll'].append(float(nll(nativeplus,ref[f,0],mask[f])-nll(nativea,ref[f,0],mask[f])))
        invariant['input_plus_state_A_minus_native_plus_nll'].append(float(nll(pa,ref[f,0],mask[f])-nll(nativeplus,ref[f,0],mask[f])))
        invariant['input_A_state_plus_minus_native_A_nll'].append(float(nll(ap,ref[f,0],mask[f])-nll(nativea,ref[f,0],mask[f])))
        for key,out,base in [('input_plus_state_A_probability_TV',pa,nativeplus),('input_A_state_plus_probability_TV',ap,nativea)]:
            tv = .5*np.abs(np.exp(logprob(out))-np.exp(logprob(base))).sum(-1)[CELL][mask[f]].mean()
            invariant[key].append(float(tv))
    E = dict(input_effect=interval(u,ix),state_effect=interval(v,ix),interaction=interval(w,ix),
             native_span=interval(g,ix),label_counts=counts,per_family=descriptions,
             invariant={k:interval(vals,ix) for k,vals in invariant.items()},
             ratio_warning='u+v=g algebraically; ratios are not independent resources or causal mediation fractions')
    cut = Q['cut_level_one_based']-1
    zcut = native['z'][:,:,cut].astype(np.float64)
    lo,hi = zcut.min((0,1)),zcut.max((0,1))
    perturbed = (native['z'][:,:2,None,cut]+c['delta']).astype(np.float64)
    outside = ((perturbed<lo)|(perturbed>hi)).mean((-1,-2))
    norms = np.linalg.norm(c['delta'].astype(np.float64),axis=(-1,-2))
    relnorm = norms/np.linalg.norm(zcut[:,:2],axis=(-1,-2))[:,:,None]
    dist_ab = np.linalg.norm(zcut[:,1]-zcut[:,0],axis=(-1,-2))/np.linalg.norm(zcut[:,0],axis=(-1,-2))
    dist_ap = np.linalg.norm(zcut[:,2]-zcut[:,0],axis=(-1,-2))/np.linalg.norm(zcut[:,0],axis=(-1,-2))
    diagnostics = dict(perturbation_relative_norm_mean=float(relnorm.mean()),
                       envelope_coordinate_fraction_mean=float(outside.mean()),
                       envelope_coordinate_fraction_max=float(outside.max()),
                       latent_AB_relative_distance=interval(dist_ab,ix),latent_Aplus_relative_distance=interval(dist_ap,ix),
                       warning='Envelope is not a density/OOD test; swapped native states may be conditionally OOD')
    np.savez_compressed(D/'results/metrics.npz',losses=losses,**contrasts,h=h,input_effect=u,state_effect=v,interaction=w,
                        native_span=g,**{'invariant_'+k:np.asarray(vals) for k,vals in invariant.items()})
    write(D/'results/summary.json',dict(families=n,O=O,C=C,E=E,diagnostics=diagnostics,
          decision_scope='Pilot with frozen checkpoint; conditional functional effects only; no mechanism labels'))
    check_freeze()
    print('Fixed analyses and decisions written.')


if __name__ == '__main__':
    main()

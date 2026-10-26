"""Predefined calibration rules: observation access is enforced by projection."""
from support import Q
from controls import LABELS


def project(record, regime):
    keys = dict(O=['trajectory'], C=['trajectory', 'crossed'],
                E=['trajectory', 'crossed', 'no_tail_change', 'input_switch_distance_a',
                   'input_switch_distance_b', 'state_switch_distance_a', 'state_switch_distance_b', 'invariant_change'])[regime]
    return {k: record[k] for k in keys}


def naive(observations):
    assert set(observations) == {'trajectory'}
    return ['state_compensation']


def cautious(observations):
    assert set(observations) == {'trajectory'}
    return LABELS.copy()


def crossed(observations):
    eps = Q['zero_tolerance']
    trials = observations['crossed']
    if max(t['losses']['Y10']-t['losses']['Y00'] for t in trials) < eps:
        return ['oracle_replay', 'tolerant']
    if all(t['R'] > eps and t['I'] > eps and t['S'] > eps for t in trials):
        return ['state_compensation', 'input_recompute', 'mixed']
    return ['fixed_response']


def complete(observations):
    candidates = crossed(observations)
    eps = Q['zero_tolerance']
    if candidates == ['oracle_replay', 'tolerant']:
        return ['oracle_replay' if observations['no_tail_change'] < eps else 'tolerant']
    if candidates == ['fixed_response']:
        return candidates
    # Reciprocal entry/state switches are both required.
    if observations['input_switch_distance_a'] < eps and observations['state_switch_distance_b'] < eps:
        return ['state_compensation']
    if observations['input_switch_distance_b'] < eps and observations['state_switch_distance_a'] < eps:
        return ['input_recompute']
    return ['mixed']  # finite-library label; outside it: mixed OR unmodeled.


METHODS = dict(trajectory_naive=('O', naive), trajectory_cautious=('O', cautious),
               crossed_only=('C', crossed), complete_intervention=('E', complete))

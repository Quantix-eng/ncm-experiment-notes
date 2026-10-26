"""Explicit analytical SCMs. Model names and anchors never enter method records."""
from support import Q, np

LABELS = ['oracle_replay', 'tolerant', 'state_compensation', 'input_recompute', 'fixed_response', 'mixed']


def onehot(digits, classes):
    return np.eye(classes, dtype=np.float64)[digits]


class Control:
    def __init__(self, name, v, d0, alpha=None):
        self.name, self.v, self.d0, self.alpha = name, v, d0, alpha
        self.calls = 0

    def initial(self, answer):
        return (answer-self.v if self.name == 'tolerant' else answer.copy(), answer-self.v)

    def continuation(self, input_answer, state):
        self.calls += 1
        a, b = state
        if self.name == 'oracle_replay':
            return np.zeros_like(a)
        if self.name == 'tolerant':
            return self.v.copy()
        if self.name == 'state_compensation':
            return a-b
        if self.name == 'input_recompute':
            return input_answer-b
        if self.name == 'fixed_response':
            return self.v.copy() if np.max(np.abs(b-(a-self.v))) < Q['zero_tolerance'] else self.v-self.d0
        if self.name == 'mixed':
            return self.alpha*a+(1-self.alpha)*input_answer-b
        raise ValueError(self.name)

    def read(self, state, tail):
        a, b = state
        if self.name == 'oracle_replay':
            return a.copy()
        if self.name == 'tolerant':
            return a+tail
        return b+tail


def loss(scores, answer):
    return float(np.mean((scores-answer)**2))


def observe(source, name, alpha, rng):
    sa, sb = (onehot(source[k], source['classes']) for k in ('target_a', 'target_b'))
    positions = rng.choice(source['eligible'], 2, replace=False)
    directions = []
    for pos in positions:
        d = np.zeros_like(sa)
        correct = source['target_a'][int(pos)]
        d[pos, correct] = -1
        d[pos, (correct+1) % source['classes']] = 1
        directions.append(d)
    d0, d1 = directions
    model = Control(name, Q['native_residual_amplitude']*d0, d0, alpha)
    za, zb = model.initial(sa), model.initial(sb)
    ta, tb = model.continuation(sa, za), model.continuation(sb, zb)
    native = model.read(za, ta)
    assert np.max(np.abs(native-sa)) < Q['zero_tolerance']
    assert np.max(np.abs(model.read(za, model.continuation(sa, za))-native)) < Q['zero_tolerance']
    vectors = dict(answer_a=sa, answer_b=sb, direction0=d0, direction1=d1, v=model.v,
                   state_a_anchor=za[0], state_a_payload=za[1], state_b_anchor=zb[0],
                   state_b_payload=zb[1], tail_a=ta, tail_b=tb)
    trajectory = np.stack([native+offset*d0 for offset in Q['reader_schedule']])
    vectors['trajectory'] = trajectory
    crossed = []
    for direction_index, d in enumerate(directions):
        for sign in (1, -1):
            delta = sign*Q['perturbation_amplitude']*d
            wrong_delta = sign*Q['perturbation_amplitude']*directions[1-direction_index]
            zd = (za[0], za[1]+delta)
            td = model.continuation(sa, zd)
            tw = model.continuation(sa, (za[0], za[1]+wrong_delta))
            outputs = dict(Y00=native, Y10=model.read(zd, ta), Y11=model.read(zd, td),
                           Y01=model.read(za, td), Y1wrong=model.read(zd, tw))
            losses = {k: loss(y, sa) for k, y in outputs.items()}
            crossed.append(dict(direction=direction_index, sign=sign, losses=losses,
                                R=losses['Y10']-losses['Y11'],
                                I=losses['Y10']+losses['Y01']-losses['Y11']-losses['Y00'],
                                S=losses['Y1wrong']-losses['Y11']))
            for key, vector in dict(delta=delta, tail=td, wrong_tail=tw, **outputs).items():
                vectors[f'cross_{direction_index}_{sign}_{key}'] = vector
    no_tail = model.read(za, np.zeros_like(ta))
    input_switch = model.read(za, model.continuation(sb, za))
    state_switch = model.read(zb, model.continuation(sa, zb))
    invariant = model.read(za, model.continuation(sa, za))
    vectors.update(no_tail=no_tail, input_switch=input_switch, state_switch=state_switch, invariant=invariant)
    for input_index, answer in enumerate((sa, sb)):
        for state_index, state in enumerate((za, zb)):
            for tail_name, tail in (('A', ta), ('B', tb), ('regenerated', model.continuation(answer, state))):
                vectors[f'natural_{input_index}_{state_index}_{tail_name}'] = model.read(state, tail)
    record = dict(trajectory=trajectory.argmax(-1).tolist(), crossed=crossed,
                  native_loss=loss(native, sa), no_tail_change=loss(no_tail, native),
                  input_switch_distance_a=loss(input_switch, sa), input_switch_distance_b=loss(input_switch, sb),
                  state_switch_distance_a=loss(state_switch, sa), state_switch_distance_b=loss(state_switch, sb),
                  invariant_change=loss(invariant, native), continuation_calls=model.calls)
    return record, vectors

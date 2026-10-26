from common import *
import itertools


def orbit(v):
    return sorted({tuple(w[i:]+w[:i]) for w in (list(v),list(v)[::-1]) for i in range(4)})


if __name__=='__main__':
    assert not (D/'data/families.json').exists()
    groups={min(orbit(v)):orbit(v) for v in itertools.permutations(range(8),4)}
    assert len(groups)==210 and all(len(v)==8 for v in groups.values())
    keys=sorted(groups);rng=np.random.default_rng(Q['partition_seed']);rng.shuffle(keys)
    rows=[];train_x=[];train_y=[]
    for i,key in enumerate(keys):
        split='train' if i<140 else 'development' if i<170 else 'evaluation'
        options=groups[key];base=options[int(rng.integers(8))]
        variants=[examples(base,b,n) for b,n in ((0,0),(1,0),(0,1))]
        rows.append(dict(source_id=f'rotation-{i:03d}',split=split,canonical=list(key),orbit=options,
                         inputs=[v[0] for v in variants],targets=[v[1] for v in variants]))
        if split=='train':
            for v in options:
                for b,n in itertools.product((0,1),repeat=2):
                    x,y=examples(v,b,n);train_x.append(x);train_y.append(y)
    write('data/families.json',dict(seed=Q['partition_seed'],rows=rows))
    np.savez_compressed(D/'data/train.npz',x=np.asarray(train_x),y=np.asarray(train_y))
    print('210 source orbits; 4480 training examples, 30 development and 40 evaluation families.')

"""Tokenization-only preparation. Does not load model weights or generate outputs."""
from support import *

PAIRS = [('red','blue'), ('black','white'), ('green','yellow'), ('gold','silver'),
         ('cat','dog'), ('horse','bird'), ('fish','bear'), ('lion','tiger'),
         ('apple','orange'), ('bread','cheese'), ('water','milk'), ('coffee','tea'),
         ('summer','winter'), ('spring','autumn'), ('north','south'), ('east','west')]

if __name__ == '__main__':
    assert not (D/'data/families.json').exists()
    tok = GPT2TokenizerFast.from_pretrained(str(ROOT/'artifacts/gpt2'), local_files_only=True)
    rows = []
    for i, (a, b) in enumerate(PAIRS):
        prompts = [f'The code is {a}. The code is', f'The code is {b}. The code is',
                   f'My code is {a}. The code is']
        ids = [tok.encode(p, add_special_tokens=False) for p in prompts]
        targets = [tok.encode(' '+word, add_special_tokens=False) for word in (a,b,a)]
        assert len({len(s) for s in ids}) == 1
        assert all(len(s) == 1 for s in targets)
        assert len([j for j,(x,y) in enumerate(zip(ids[0],ids[1])) if x != y]) == 1
        assert ids[0][-1] == ids[1][-1] == ids[2][-1]
        rows.append(dict(source_id=f'copy-{i:03d}', words=[a,b], prompts=prompts,
                         tokens=ids, reference=[s[0] for s in targets]))
    assert len({len(s) for r in rows for s in r['tokens']}) == 1
    write('data/families.json', dict(status='Chosen before model outputs; tokenization-only checks', rows=rows))
    print('Prepared',len(rows),'families; sequence length',len(rows[0]['tokens'][0]))
